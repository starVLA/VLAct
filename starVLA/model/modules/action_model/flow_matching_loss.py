import math
from typing import Optional

import torch


def _cfg_get(cfg, key: str, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        value = cfg.get(key, default)
        if value is not None:
            return value
    return getattr(cfg, key, default)


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _get_nested_config(config, *keys):
    node = config
    for key in keys:
        node = _cfg_get(node, key, None)
        if node is None:
            return None
    return node


def get_endpoint_wrap_loss_weight(config, default_when_enabled: float = 0.1) -> float:
    action_cfg = _get_nested_config(config, "framework", "action_model")
    trainer_cfg = _cfg_get(config, "trainer", None)

    for cfg in (action_cfg, trainer_cfg):
        explicit = _cfg_get(cfg, "endpoint_wrap_loss_weight", None)
        if explicit is not None:
            return float(explicit)
    if use_shortest_angular_joint_loss_diff(config):
        return default_when_enabled
    return 0.0


def use_shortest_angular_joint_loss_diff(config) -> bool:
    trainer_cfg = _cfg_get(config, "trainer", None)
    return _as_bool(_cfg_get(trainer_cfg, "shortest_angular_joint_loss_diff", False))


def blend_angular_wrap_abs_error(
    abs_error: torch.Tensor,
    raw_diff: torch.Tensor,
    angular_mask: torch.Tensor,
) -> torch.Tensor:
    if not angular_mask.any():
        return abs_error

    wrapped_abs = torch.abs(torch.remainder(raw_diff + math.pi, 2 * math.pi) - math.pi)
    angular_abs_error = 0.5 * wrapped_abs + 0.5 * abs_error
    return torch.where(angular_mask.unsqueeze(1), angular_abs_error, abs_error)


def _parse_dim_spec(dim_spec, action_dim: int) -> torch.Tensor:
    mask = torch.zeros(action_dim, dtype=torch.bool)
    if dim_spec is None:
        return mask

    if isinstance(dim_spec, str):
        raw_parts = [part.strip() for part in dim_spec.split(",") if part.strip()]
    else:
        raw_parts = list(dim_spec)

    for part in raw_parts:
        if isinstance(part, str) and ":" in part:
            start_s, end_s = part.split(":", 1)
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else action_dim
            mask[max(0, start): min(action_dim, end)] = True
        else:
            idx = int(part)
            if -action_dim <= idx < action_dim:
                mask[idx % action_dim] = True
    return mask


def resolve_angular_dim_mask(
    config,
    actions: torch.Tensor,
    angular_dim_mask: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    if angular_dim_mask is not None:
        return angular_dim_mask.to(device=actions.device, dtype=torch.bool)

    action_cfg = _get_nested_config(config, "framework", "action_model")
    trainer_cfg = _cfg_get(config, "trainer", None)
    explicit_dims = _cfg_get(action_cfg, "endpoint_wrap_dims", None)
    if explicit_dims is None:
        explicit_dims = _cfg_get(trainer_cfg, "endpoint_wrap_dims", None)

    action_dim = actions.shape[-1]
    if explicit_dims is not None:
        mask = _parse_dim_spec(explicit_dims, action_dim).to(actions.device)
        return mask if mask.any() else None

    if not _as_bool(_cfg_get(trainer_cfg, "shortest_angular_joint_loss_diff", False)):
        return None

    framework_cfg = _cfg_get(config, "framework", None)
    if _as_bool(_cfg_get(framework_cfg, "disjoint_action_layout", False)) and action_dim >= 20:
        mask = torch.zeros(action_dim, dtype=torch.bool, device=actions.device)
        mask[:12] = True
        return mask

    # Common dual-arm joint layout: joint angles first, two gripper dims last.
    if action_dim >= 14:
        mask = torch.zeros(action_dim, dtype=torch.bool, device=actions.device)
        mask[: action_dim - 2] = True
        return mask
    return None


def masked_mean(
    loss_tensor: torch.Tensor,
    *,
    action_valid_mask: Optional[torch.Tensor] = None,
    action_dim_mask: Optional[torch.Tensor] = None,
    sample_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    weighted_loss = loss_tensor
    loss_mask = torch.ones_like(loss_tensor)

    if action_valid_mask is not None:
        step_mask = action_valid_mask.to(
            device=loss_tensor.device, dtype=loss_tensor.dtype
        ).unsqueeze(-1)
        loss_mask = loss_mask * step_mask

    if action_dim_mask is not None:
        dim_mask = action_dim_mask.to(
            device=loss_tensor.device, dtype=loss_tensor.dtype
        )
        if dim_mask.ndim == 1:
            dim_mask = dim_mask.unsqueeze(0)
        loss_mask = loss_mask * dim_mask.unsqueeze(1)

    if sample_weights is not None:
        weighted_loss = weighted_loss * sample_weights.to(
            device=loss_tensor.device, dtype=loss_tensor.dtype
        ).view(-1, 1, 1)

    if action_valid_mask is None and action_dim_mask is None:
        return weighted_loss.mean()

    denom = loss_mask.sum()
    if denom.item() <= 0:
        return loss_tensor.sum() * 0.0
    return (weighted_loss * loss_mask).sum() / denom


def flow_matching_loss_with_endpoint_wrap(
    *,
    velocity_pred: torch.Tensor,
    velocity_target: torch.Tensor,
    noisy_actions: torch.Tensor,
    target_actions: torch.Tensor,
    t: torch.Tensor,
    config=None,
    action_valid_mask: Optional[torch.Tensor] = None,
    action_dim_mask: Optional[torch.Tensor] = None,
    angular_dim_mask: Optional[torch.Tensor] = None,
    sample_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    velocity_loss = masked_mean(
        (velocity_pred - velocity_target) ** 2,
        action_valid_mask=action_valid_mask,
        action_dim_mask=action_dim_mask,
        sample_weights=sample_weights,
    )

    angular_dim_mask = resolve_angular_dim_mask(config, target_actions, angular_dim_mask)
    use_diff_blend = use_shortest_angular_joint_loss_diff(config)
    wrap_weight = get_endpoint_wrap_loss_weight(config)
    if angular_dim_mask is None or not angular_dim_mask.any() or wrap_weight <= 0.0:
        return velocity_loss, velocity_loss.detach(), velocity_loss.new_zeros(())

    t_flat = t.reshape(t.shape[0])
    t_view = t_flat.view(-1, 1, 1).to(device=target_actions.device, dtype=target_actions.dtype)
    endpoint_pred = noisy_actions + (1.0 - t_view) * velocity_pred
    raw_diff = endpoint_pred - target_actions
    wrapped_diff = torch.remainder(raw_diff + math.pi, 2 * math.pi) - math.pi
    if use_diff_blend:
        wrap_loss_tensor = 0.5 * torch.abs(wrapped_diff) + 0.5 * torch.abs(raw_diff)
    else:
        wrap_loss_tensor = wrapped_diff ** 2

    min_t = 0.2  # skip endpoint constraint on the noisiest timesteps
    endpoint_valid_mask = action_valid_mask
    t_mask = (t_flat >= min_t).to(device=target_actions.device, dtype=target_actions.dtype)
    endpoint_valid_mask = (
        t_mask.unsqueeze(-1)
        if endpoint_valid_mask is None
        else endpoint_valid_mask * t_mask.unsqueeze(-1)
    )

    wrap_dim_mask = angular_dim_mask
    if action_dim_mask is not None:
        wrap_dim_mask = wrap_dim_mask.to(device=target_actions.device, dtype=torch.bool)
        adm = action_dim_mask.to(device=target_actions.device, dtype=torch.bool)
        wrap_dim_mask = wrap_dim_mask & adm

    endpoint_wrap_loss = masked_mean(
        wrap_loss_tensor,
        action_valid_mask=endpoint_valid_mask,
        action_dim_mask=wrap_dim_mask,
        sample_weights=sample_weights,
    )
    total_loss = velocity_loss + float(wrap_weight) * endpoint_wrap_loss
    return (
        total_loss,
        velocity_loss.detach(),
        endpoint_wrap_loss.detach(),
    )
