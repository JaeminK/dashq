from __future__ import annotations

import torch
import torch.nn as nn


def _has_tensorized_moe_projections(module: nn.Module) -> bool:
    required_attrs = ("gate_up_proj", "down_proj", "num_experts", "hidden_size", "expert_dim")
    return (
        all(hasattr(module, attr) for attr in required_attrs)
        and isinstance(module.gate_up_proj, torch.nn.Parameter)
        and isinstance(module.down_proj, torch.nn.Parameter)
    )


def _forward_tensorized_moe_with_linear_experts(
    self,
    hidden_states: torch.Tensor,
    router_indices=None,
    routing_weights=None,
) -> torch.Tensor:
    batch_size = hidden_states.shape[0]
    hidden_states = hidden_states.reshape(-1, self.hidden_size)
    next_states = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
    num_experts = self.num_experts

    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(router_indices, num_classes=num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        active_experts = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in active_experts:
        e_idx = expert_idx[0].item()
        if e_idx >= self.num_experts:
            continue
        _, token_idx = torch.where(expert_mask[e_idx])
        current_state = hidden_states[token_idx]
        gate_up = self.gate_up_proj_list[e_idx](current_state)
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]

        if hasattr(self, "limit") and self.limit is not None:
            gate = gate.clamp(min=None, max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)

        alpha_val = self.alpha if hasattr(self, "alpha") else 1.702
        glu = gate * torch.sigmoid(gate * alpha_val)
        gated_output = (up + 1) * glu
        out = self.down_proj_list[e_idx](gated_output)
        expert_routing_weight = routing_weights[token_idx, e_idx].unsqueeze(-1)
        weighted_output = out * expert_routing_weight
        next_states.index_add_(0, token_idx, weighted_output.to(hidden_states.dtype))

    return next_states.view(batch_size, -1, self.hidden_size)


def _build_linear_from_expert_weight(
    weight_slice: torch.Tensor,
    bias_slice: torch.Tensor | None,
    in_features: int,
    out_features: int,
) -> nn.Linear:
    linear = nn.Linear(
        in_features,
        out_features,
        bias=bias_slice is not None,
        device=weight_slice.device,
        dtype=weight_slice.dtype,
    )
    linear.weight = nn.Parameter(weight_slice.t().contiguous(), requires_grad=False)
    if bias_slice is not None:
        linear.bias = nn.Parameter(bias_slice.contiguous(), requires_grad=False)
    return linear


def expose_tensorized_moe_experts_as_linears(model: nn.Module) -> int:
    converted = 0
    converted_classes: set[type[nn.Module]] = set()
    candidates = [
        module for _, module in model.named_modules() if _has_tensorized_moe_projections(module)
    ]
    for module in candidates:
        device = module.gate_up_proj.device
        dtype = module.gate_up_proj.dtype
        num_experts = int(module.num_experts)
        hidden_size = int(module.hidden_size)
        expert_dim = int(module.expert_dim)

        gate_up_weight = module.gate_up_proj
        down_weight = module.down_proj
        gate_up_bias = getattr(module, "gate_up_proj_bias", None)
        down_bias = getattr(module, "down_proj_bias", None)

        gate_up_list = nn.ModuleList()
        down_list = nn.ModuleList()
        for expert_idx in range(num_experts):
            gate_bias = gate_up_bias[expert_idx] if isinstance(gate_up_bias, torch.nn.Parameter) else None
            down_expert_bias = down_bias[expert_idx] if isinstance(down_bias, torch.nn.Parameter) else None
            gate_up_list.append(
                _build_linear_from_expert_weight(
                    gate_up_weight[expert_idx],
                    gate_bias,
                    hidden_size,
                    2 * expert_dim,
                )
            )
            down_list.append(
                _build_linear_from_expert_weight(
                    down_weight[expert_idx],
                    down_expert_bias,
                    expert_dim,
                    hidden_size,
                )
            )

        module.gate_up_proj_list = gate_up_list.to(device=device, dtype=dtype)
        module.down_proj_list = down_list.to(device=device, dtype=dtype)
        del module.gate_up_proj
        if isinstance(gate_up_bias, torch.nn.Parameter):
            del module.gate_up_proj_bias
        del module.down_proj
        if isinstance(down_bias, torch.nn.Parameter):
            del module.down_proj_bias
        converted += 1
        converted_classes.add(type(module))

    for expert_cls in converted_classes:
        expert_cls.forward = _forward_tensorized_moe_with_linear_experts
    return converted
