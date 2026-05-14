from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

from dashq.moe import expose_tensorized_moe_experts_as_linears
from dashq.quantization import PackedQuantizedLinear


MODEL_LOADERS = {
    "causal_lm": AutoModelForCausalLM,
    "image_text_to_text": AutoModelForImageTextToText,
}


def _dtype_from_string(value: str | None, default: torch.dtype = torch.bfloat16) -> torch.dtype:
    if value is None:
        return default
    key = str(value).replace("torch.", "").lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if hasattr(torch, "float8_e5m2"):
        mapping["fp8"] = torch.float8_e5m2
        mapping["fp8_e5m2"] = torch.float8_e5m2
        mapping["float8_e5m2"] = torch.float8_e5m2
    if hasattr(torch, "float8_e4m3fn"):
        mapping.setdefault("fp8", torch.float8_e4m3fn)
        mapping["fp8_e4m3"] = torch.float8_e4m3fn
        mapping["float8_e4m3fn"] = torch.float8_e4m3fn
    return mapping.get(key, default)


def _load_dashq_config(snapshot_path: Path) -> dict[str, Any]:
    config_path = snapshot_path / "dashq_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing dashq_config.json in {snapshot_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("format") != "dashq-packed-linear":
        raise ValueError(f"Unsupported DASH-Q checkpoint format: {config.get('format')}")
    return config


def _checkpoint_keys(snapshot_path: Path) -> set[str]:
    index_path = snapshot_path / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as handle:
            index = json.load(handle)
        return set(index.get("weight_map", {}).keys())

    keys: set[str] = set()
    for path in snapshot_path.glob("*.safetensors"):
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys.update(handle.keys())
    if not keys:
        raise FileNotFoundError(f"No safetensors checkpoint files found in {snapshot_path}")
    return keys


def _set_module_by_name(root: nn.Module, name: str, module: nn.Module) -> None:
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    leaf = parts[-1]
    if leaf.isdigit():
        parent[int(leaf)] = module
    else:
        setattr(parent, leaf, module)


def _replace_quantized_modules(
    model: nn.Module,
    quantization_map: dict[str, dict[str, Any]],
    checkpoint_keys: set[str],
) -> int:
    replaced = 0
    for name, meta in quantization_map.items():
        has_bias = f"{name}.bias" in checkpoint_keys
        module = PackedQuantizedLinear(
            in_features=int(meta["in_features"]),
            out_features=int(meta["out_features"]),
            quant_in_features=int(meta.get("quant_in_features", meta["in_features"])),
            nbits=int(meta["nbits"]),
            group_size=int(meta["group_size"]),
            num_groups=int(meta["num_groups"]),
            has_bias=has_bias,
            linear_dtype=_dtype_from_string(meta.get("linear_dtype")),
            scale_zero_dtype=_dtype_from_string(meta.get("scale_zero_dtype"), torch.float16),
        )
        _set_module_by_name(model, name, module)
        replaced += 1
    return replaced


def _compile_quantized_dequantization(
    model: nn.Module,
    compile_kwargs: dict[str, Any] | None = None,
) -> int:
    compile_kwargs = compile_kwargs or {}
    compiled = 0
    for module in model.modules():
        if isinstance(module, PackedQuantizedLinear) and module.compile_dequantization(**compile_kwargs):
            compiled += 1
    return compiled


def load_quantized(
    repo_id_or_path: str | Path,
    *,
    cache_dir: str | Path | None = "/workspace/cache/models",
    device_map: str | dict[str, Any] | None = "auto",
    max_memory: dict[int | str, str] | None = None,
    dtype: torch.dtype | str | None = torch.bfloat16,
    model_class: str | None = None,
    token: str | bool | None = None,
    revision: str | None = None,
    trust_remote_code: bool = True,
    offload_folder: str | Path | None = None,
    compile_dequantization: bool = True,
    compile_dequantization_kwargs: dict[str, Any] | None = None,
) -> tuple[nn.Module, Any]:
    """Load a DASH-Q packed quantized checkpoint and tokenizer.

    `repo_id_or_path` can be a Hub model ID or a local snapshot directory.
    """
    repo_or_path = Path(repo_id_or_path)
    if repo_or_path.exists():
        snapshot_path = repo_or_path
    else:
        snapshot_path = Path(
            snapshot_download(
                repo_id=str(repo_id_or_path),
                cache_dir=cache_dir,
                revision=revision,
                token=token,
            )
        )

    dashq_config = _load_dashq_config(snapshot_path)
    resolved_model_class = model_class or dashq_config.get("model_class", "causal_lm")
    if resolved_model_class not in MODEL_LOADERS:
        raise ValueError(f"Unsupported model_class: {resolved_model_class}")

    config = AutoConfig.from_pretrained(
        snapshot_path,
        trust_remote_code=trust_remote_code,
    )
    loader = MODEL_LOADERS[resolved_model_class]
    with init_empty_weights():
        model = loader.from_config(config, trust_remote_code=trust_remote_code)
        expose_tensorized_moe_experts_as_linears(model)
        replaced = _replace_quantized_modules(
            model,
            dashq_config["quantized_modules"],
            _checkpoint_keys(snapshot_path),
        )
        if hasattr(model, "tie_weights"):
            model.tie_weights()
    if replaced != len(dashq_config["quantized_modules"]):
        raise RuntimeError("Failed to replace all DASH-Q quantized modules.")

    no_split_classes = list(getattr(model, "_no_split_modules", []) or [])
    no_split_classes.append("PackedQuantizedLinear")
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=snapshot_path,
        device_map=device_map,
        max_memory=max_memory,
        no_split_module_classes=sorted(set(no_split_classes)),
        dtype=dtype,
        offload_folder=offload_folder,
        strict=False,
    )
    if compile_dequantization:
        _compile_quantized_dequantization(model, compile_dequantization_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(snapshot_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
