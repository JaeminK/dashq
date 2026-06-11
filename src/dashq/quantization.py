from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _packed_length(numel: int, nbits: int) -> int:
    if int(nbits) < 1 or int(nbits) > 8:
        raise ValueError(f"nbits must be between 1 and 8, got {nbits}.")
    values_per_word = max(1, 32 // int(nbits))
    return math.ceil(int(numel) / values_per_word)


def _build_unpack_impl(nbits: int):
    values_per_word = max(1, 32 // int(nbits))
    mask = (1 << int(nbits)) - 1

    def unpack(packed: torch.Tensor, numel: int) -> torch.Tensor:
        shifts = torch.arange(values_per_word, device=packed.device, dtype=torch.int64) * int(nbits)
        words = packed.to(torch.int64).unsqueeze(1)
        vals = torch.bitwise_and(torch.bitwise_right_shift(words, shifts.view(1, -1)), mask)
        return vals.reshape(-1)[:numel].to(torch.int32)

    return unpack


_UNPACK_IMPLS: dict[int, Any] = {}


def _unpack_int_values(packed: torch.Tensor, nbits: int, numel: int) -> torch.Tensor:
    nbits = int(nbits)
    if nbits < 1 or nbits > 8:
        raise ValueError(f"nbits must be between 1 and 8, got {nbits}.")
    if nbits not in _UNPACK_IMPLS:
        _UNPACK_IMPLS[nbits] = _build_unpack_impl(nbits)
    return _UNPACK_IMPLS[nbits](packed, int(numel))


class PackedQuantizedLinear(nn.Module):
    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        quant_in_features: int,
        nbits: int,
        group_size: int,
        num_groups: int,
        has_bias: bool,
        linear_dtype: torch.dtype = torch.bfloat16,
        scale_zero_dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        self.linear_dtype = linear_dtype
        self.in_features = int(in_features)
        self.quant_in_features = int(quant_in_features)
        self.out_features = int(out_features)
        self.nbits = int(nbits)
        self.group_size = int(group_size)
        self.num_groups = int(num_groups)
        self.numel = self.out_features * self.quant_in_features
        self.scale_zero_dtype = scale_zero_dtype
        self.meta = {
            "shape": (self.out_features, self.in_features),
            "nbits": self.nbits,
            "group_size": self.group_size,
            "packing": f"int{self.nbits}_packed_u32",
        }

        self.register_buffer(
            "W_q_packed",
            torch.empty(_packed_length(self.numel, self.nbits), dtype=torch.int32),
        )
        self.register_buffer(
            "scale",
            torch.empty((self.num_groups, 1), dtype=self.scale_zero_dtype),
        )
        self.register_buffer(
            "zero",
            torch.empty((self.num_groups, 1), dtype=self.scale_zero_dtype),
        )
        if has_bias:
            self.bias = nn.Parameter(torch.empty(self.out_features, dtype=self.linear_dtype), requires_grad=False)
        else:
            self.register_parameter("bias", None)
        self._compiled_dequantize_weight = None
        self._compile_dequantization_failed = False

    def _dequantize_weight_eager(self, dtype: torch.dtype) -> torch.Tensor:
        w_int = _unpack_int_values(self.W_q_packed, self.nbits, self.numel)
        w_int = w_int.view(self.num_groups, self.group_size).to(dtype)
        weight = (w_int - self.zero.to(dtype)) * self.scale.to(dtype)
        return weight.view(self.out_features, self.quant_in_features).to(dtype=dtype)

    @property
    def weight(self) -> torch.Tensor:
        """Dequantized weight for code paths reading `.weight` directly
        (e.g. NemotronH fused Mamba kernels)."""
        return self.dequantize_weight(self.linear_dtype)

    def compile_dequantization(self, **compile_kwargs: Any) -> bool:
        if not hasattr(torch, "compile"):
            return False
        try:
            self._compiled_dequantize_weight = torch.compile(
                self._dequantize_weight_eager,
                **compile_kwargs,
            )
        except Exception:
            self._compiled_dequantize_weight = None
            self._compile_dequantization_failed = True
            return False
        self._compile_dequantization_failed = False
        return True

    def dequantize_weight(self, dtype: torch.dtype) -> torch.Tensor:
        if self._compiled_dequantize_weight is not None and not self._compile_dequantization_failed:
            try:
                return self._compiled_dequantize_weight(dtype)
            except Exception:
                self._compiled_dequantize_weight = None
                self._compile_dequantization_failed = True
        return self._dequantize_weight_eager(dtype)

    def forward_linear(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.dequantize_weight(dtype=x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_linear(x)
