from __future__ import annotations

import torch
import torch.nn as nn


def _tensorized_moe_dims(module: nn.Module) -> tuple[int, int] | None:
    if not (
        hasattr(module, "num_experts")
        and isinstance(getattr(module, "gate_up_proj", None), torch.nn.Parameter)
        and isinstance(getattr(module, "down_proj", None), torch.nn.Parameter)
    ):
        return None

    gate_up_shape = tuple(module.gate_up_proj.shape)
    down_shape = tuple(module.down_proj.shape)
    if len(gate_up_shape) != 3 or len(down_shape) != 3:
        return None

    num_experts = int(module.num_experts)
    if gate_up_shape[0] != num_experts or down_shape[0] != num_experts:
        return None

    if gate_up_shape[1] % 2 != 0:
        return None

    hidden_size = int(getattr(module, "hidden_size", getattr(module, "hidden_dim", gate_up_shape[2])))
    expert_dim = int(getattr(module, "expert_dim", getattr(module, "intermediate_dim", gate_up_shape[1] // 2)))
    if gate_up_shape != (num_experts, 2 * expert_dim, hidden_size):
        return None
    if down_shape != (num_experts, hidden_size, expert_dim):
        return None
    return hidden_size, expert_dim


def _has_tensorized_moe_projections(module: nn.Module) -> bool:
    return _tensorized_moe_dims(module) is not None


def _forward_tensorized_moe_with_linear_experts(
    self,
    hidden_states: torch.Tensor,
    top_k_index=None,
    top_k_weights=None,
    router_indices=None,
    routing_weights=None,
) -> torch.Tensor:
    input_shape = hidden_states.shape
    hidden_size = getattr(self, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(self, "hidden_dim")
    hidden_size = int(hidden_size)
    hidden_states = hidden_states.reshape(-1, hidden_size)
    next_states = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
    num_experts = self.num_experts
    expert_indices = top_k_index if top_k_index is not None else router_indices
    expert_weights = top_k_weights if top_k_weights is not None else routing_weights

    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(expert_indices, num_classes=num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        active_experts = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in active_experts:
        e_idx = expert_idx[0].item()
        if e_idx >= self.num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[e_idx])
        current_state = hidden_states[token_idx]
        gate_up = self.gate_up_proj_list[e_idx](current_state)
        gate, up = gate_up.chunk(2, dim=-1)
        if hasattr(self, "act_fn"):
            gated_output = self.act_fn(gate) * up
        else:
            if hasattr(self, "limit") and self.limit is not None:
                gate = gate.clamp(min=None, max=self.limit)
                up = up.clamp(min=-self.limit, max=self.limit)
            alpha_val = self.alpha if hasattr(self, "alpha") else 1.702
            gated_output = (up + 1) * (gate * torch.sigmoid(gate * alpha_val))
        out = self.down_proj_list[e_idx](gated_output)
        if expert_weights.shape == expert_indices.shape:
            expert_routing_weight = expert_weights[token_idx, top_k_pos].unsqueeze(-1)
        else:
            expert_routing_weight = expert_weights[token_idx, e_idx].unsqueeze(-1)
        weighted_output = out * expert_routing_weight
        next_states.index_add_(0, token_idx, weighted_output.to(hidden_states.dtype))

    return next_states.view(input_shape)


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
    linear.weight = nn.Parameter(weight_slice.contiguous(), requires_grad=False)
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
        dims = _tensorized_moe_dims(module)
        if dims is None:
            continue
        hidden_size, expert_dim = dims

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
    converted += _expose_nemotron_latent_moe_experts(model)
    return converted


def _nemotron_latent_moe_dims(module: nn.Module) -> tuple[int, int] | None:
    if not (
        hasattr(module, "num_experts")
        and isinstance(getattr(module, "up_proj", None), torch.nn.Parameter)
        and isinstance(getattr(module, "down_proj", None), torch.nn.Parameter)
        and getattr(module, "gate_up_proj", None) is None
    ):
        return None
    up_shape = tuple(module.up_proj.shape)
    down_shape = tuple(module.down_proj.shape)
    if len(up_shape) != 3 or len(down_shape) != 3:
        return None
    num_experts = int(module.num_experts)
    intermediate_dim, input_dim = up_shape[1], up_shape[2]
    if up_shape != (num_experts, intermediate_dim, input_dim):
        return None
    if down_shape != (num_experts, input_dim, intermediate_dim):
        return None
    return input_dim, intermediate_dim


def _forward_nemotron_moe_with_linear_experts(self, hidden_states, top_k_index, top_k_weights):
    final_hidden_states = torch.zeros_like(hidden_states, dtype=top_k_weights.dtype)
    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero().squeeze(-1)
    for expert_idx in expert_hit:
        expert_idx = expert_idx.item()
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        if token_idx.numel() == 0:
            continue
        current = hidden_states[token_idx]
        current = self.up_proj_list[expert_idx](current)
        current = self.act_fn(current)
        current = self.down_proj_list[expert_idx](current)
        current = current * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(0, token_idx, current.to(final_hidden_states.dtype))
    return final_hidden_states.to(hidden_states.dtype)


def _expose_nemotron_latent_moe_experts(model: nn.Module) -> int:
    """Expose NemotronH-style latent-MoE experts (non-gated up/down rank-3 tensors)
    as per-expert Linear modules. No-op for gated tensorized MoE models."""
    converted = 0
    converted_classes: set[type[nn.Module]] = set()
    for _, module in model.named_modules():
        dims = _nemotron_latent_moe_dims(module)
        if dims is None:
            continue
        input_dim, intermediate_dim = dims
        num_experts = int(module.num_experts)
        device = module.up_proj.device
        dtype = module.up_proj.dtype
        up_list = nn.ModuleList()
        down_list = nn.ModuleList()
        for expert_idx in range(num_experts):
            up_list.append(
                _build_linear_from_expert_weight(module.up_proj[expert_idx], None, input_dim, intermediate_dim)
            )
            down_list.append(
                _build_linear_from_expert_weight(module.down_proj[expert_idx], None, intermediate_dim, input_dim)
            )
        module.up_proj_list = up_list.to(device=device, dtype=dtype)
        module.down_proj_list = down_list.to(device=device, dtype=dtype)
        delattr(module, "up_proj")
        delattr(module, "down_proj")
        converted += 1
        converted_classes.add(type(module))
    for expert_cls in converted_classes:
        expert_cls.forward = _forward_nemotron_moe_with_linear_experts
    return converted
