# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
QwenPI_v4 — True Pi0.5-style VLA: Qwen-VL (prefix) + Qwen3-0.6B Action Expert (suffix)

Architecture mirrors Physical Intelligence's pi0.5:

  ┌──────────────────────────────────────────┐
  │  Qwen-VL-4B  (prefix encoder)           │  ← replaces PaliGemma
  │  images + instruction + [STATE 256bins]  │
  │  → prefix_hidden [B, S, 2048]           │
  └───────────────────┬──────────────────────┘
                      │ prefix_proj (LN + Linear 2048→1024)
                      ▼
  ┌──────────────────────────────────────────┐
  │  Qwen3-0.6B Action Expert (modified)    │  ← replaces Gemma expert
  │                                          │
  │  action_in_proj(noisy_actions) → tokens  │
  │  timestep → sinusoidal → time_mlp → c   │
  │                                          │
  │  Per layer (28 layers, pretrained):      │
  │    AdaRMSNorm(c) → Self-Attn (bidir.)   │  ← pretrained weights kept
  │    AdaRMSNorm(c) → Cross-Attn → prefix  │  ← NEW, randomly initialized
  │    AdaRMSNorm(c) → SwiGLU MLP           │  ← pretrained weights kept
  │                                          │
  │  action_out_proj → velocity prediction   │
  └──────────────────────────────────────────┘

Key pi0.5 features:
  ✓ Action expert is a PRETRAINED language model (Qwen3-0.6B)
  ✓ RMSNorm → AdaRMSNorm for timestep conditioning
  ✓ Bidirectional self-attention among action tokens
  ✓ Cross-attention to VLM prefix
  ✓ State discretized into text (256 bins)
  ✓ Flow matching with velocity prediction
  ✓ Prefix computed ONCE at inference, reused for all denoising steps
"""

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from transformers import AutoModelForCausalLM

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.action_model.flow_matching_loss import flow_matching_loss_with_endpoint_wrap
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)


# ============================================================
# AdaRMSNorm — pi0.5 timestep injection
# ============================================================

def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal positional encoding for flow matching timestep.
    Args:
        timesteps: [B] continuous values in [0, 1].
        dim: Output embedding dimension.
    Returns:
        [B, dim] sinusoidal embedding.
    """
    half = dim // 2
    freq = -math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / half
    emb = timesteps[:, None].float() * freq.exp()[None, :]
    emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb.to(timesteps.dtype)


class AdaRMSNorm(nn.Module):
    """RMSNorm with adaptive scale + shift from timestep conditioning.

    Pi0.5 style:  output = RMSNorm(x) * (1 + scale(cond)) + shift(cond)

    Scale/shift projections are zero-initialized so at init the module
    behaves exactly as the original pretrained RMSNorm.
    """

    def __init__(self, hidden_dim: int, cond_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.ada_scale = nn.Linear(cond_dim, hidden_dim, bias=False)
        self.ada_shift = nn.Linear(cond_dim, hidden_dim, bias=False)
        # Zero init → at start, behaves like original RMSNorm
        nn.init.zeros_(self.ada_scale.weight)
        nn.init.zeros_(self.ada_shift.weight)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, H] input tensor.
            cond: [B, H_cond] timestep condition (broadcast over T).
        """
        norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        norm_x = norm_x * self.weight
        scale = self.ada_scale(cond).unsqueeze(1)
        shift = self.ada_shift(cond).unsqueeze(1)
        return norm_x * (1.0 + scale) + shift

    @classmethod
    def from_pretrained_rmsnorm(cls, rmsnorm: nn.Module, cond_dim: int) -> "AdaRMSNorm":
        """Create AdaRMSNorm from a pretrained Qwen3RMSNorm, preserving weights."""
        hidden_dim = rmsnorm.weight.shape[0]
        eps = rmsnorm.variance_epsilon if hasattr(rmsnorm, 'variance_epsilon') else 1e-6
        ada = cls(hidden_dim, cond_dim, eps)
        ada.weight.data.copy_(rmsnorm.weight.data)
        return ada


# ============================================================
# Cross-Attention layer (added to each expert block)
# ============================================================

class CrossAttention(nn.Module):
    """Multi-head cross-attention: action tokens attend to VLM prefix.

    Uses the same GQA config as Qwen3-0.6B (16 heads, 8 KV heads, head_dim=128).
    Randomly initialized — this is the only new component.
    """

    def __init__(self, hidden_dim: int, num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)

        # Small init so cross-attn starts near zero → pretrained behavior preserved
        nn.init.normal_(self.q_proj.weight, std=0.02)
        nn.init.normal_(self.k_proj.weight, std=0.02)
        nn.init.normal_(self.v_proj.weight, std=0.02)
        nn.init.zeros_(self.o_proj.weight)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: [B, T_action, H] action tokens
            key_value: [B, S_prefix, H] VLM prefix hidden states
        """
        B, T_q, _ = query.shape
        T_kv = key_value.shape[1]

        q = self.q_proj(query).view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).view(B, T_kv, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).view(B, T_kv, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # GQA: repeat KV heads
        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(B, T_q, -1)
        return self.o_proj(out)


# ============================================================
# Modified Qwen3 expert layer wrapper
# ============================================================

class Qwen3ExpertLayer(nn.Module):
    """Wraps a pretrained Qwen3DecoderLayer with pi0.5 modifications:

    1. Replace input_layernorm / post_attention_layernorm → AdaRMSNorm
    2. Add cross-attention sublayer with its own AdaRMSNorm
    3. Self-attention uses bidirectional mask (is_causal=False)
    4. Timestep condition injected via all AdaRMSNorm layers

    Original Qwen3 layer:
        x = x + self_attn(input_layernorm(x))
        x = x + mlp(post_attention_layernorm(x))

    Modified (pi0.5):
        x = x + self_attn_bidir(ada_norm_1(x, t))
        x = x + cross_attn(ada_norm_cross(x, t), prefix)
        x = x + mlp(ada_norm_2(x, t))
    """

    def __init__(self, original_layer: nn.Module, cond_dim: int):
        super().__init__()
        hidden_dim = original_layer.input_layernorm.weight.shape[0]

        # Keep pretrained self-attention and MLP weights
        self.self_attn = original_layer.self_attn
        self.mlp = original_layer.mlp

        # Replace RMSNorm → AdaRMSNorm (copy pretrained norm weights)
        self.input_layernorm = AdaRMSNorm.from_pretrained_rmsnorm(
            original_layer.input_layernorm, cond_dim
        )
        self.post_attention_layernorm = AdaRMSNorm.from_pretrained_rmsnorm(
            original_layer.post_attention_layernorm, cond_dim
        )

        # NEW: cross-attention to prefix (randomly initialized)
        num_heads = self.self_attn.num_heads if hasattr(self.self_attn, 'num_heads') else 16
        num_kv_heads = self.self_attn.num_key_value_heads if hasattr(self.self_attn, 'num_key_value_heads') else 8
        head_dim = self.self_attn.head_dim if hasattr(self.self_attn, 'head_dim') else 128

        self.cross_attn_norm = AdaRMSNorm(hidden_dim, cond_dim)
        self.cross_attn = CrossAttention(hidden_dim, num_heads, num_kv_heads, head_dim)

    def forward(
        self,
        x: torch.Tensor,               # [B, T, H] action tokens
        prefix_hidden: torch.Tensor,    # [B, S, H] VLM prefix
        time_cond: torch.Tensor,        # [B, H] timestep condition
        position_embeddings: tuple = None,  # (cos, sin) from rotary embedding
    ) -> torch.Tensor:
        # 1. Bidirectional self-attention (pretrained weights)
        normed = self.input_layernorm(x, time_cond)
        attn_out = self.self_attn(
            hidden_states=normed,
            position_embeddings=position_embeddings,  # (cos, sin) for RoPE
            attention_mask=None,  # no causal mask → bidirectional
        )
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]
        x = x + attn_out

        # 2. Cross-attention to prefix (new, zero-init o_proj → starts at ~0)
        x = x + self.cross_attn(self.cross_attn_norm(x, time_cond), prefix_hidden)

        # 3. MLP (pretrained weights)
        x = x + self.mlp(self.post_attention_layernorm(x, time_cond))

        return x


# ============================================================
# QwenPI_v4 Framework
# ============================================================

@FRAMEWORK_REGISTRY.register("QwenPI_v4")
class Qwen_PI_v4(baseframework):
    """
    True Pi0.5-style VLA: Qwen-VL prefix + pretrained Qwen3-0.6B action expert.

    Training:
        1. Qwen-VL encodes images + instruction + discretized state → prefix
        2. Flow matching: sample t, build x_t, compute velocity target
        3. Qwen3 expert denoises x_t conditioned on prefix + timestep(AdaRMSNorm)
        4. Loss = MSE(predicted_velocity, target_velocity)

    Inference:
        1. Qwen-VL encodes prefix (computed ONCE)
        2. x_0 = noise, iterate Euler steps through expert
        3. Return x_1 as predicted actions
    """

    def __init__(self, config=None, **kwargs):
        super().__init__()
        self.config = config

        # ── Prefix: Qwen-VL ──────────────────────────────────────────
        self.qwen_vl_interface = get_vlm_model(config=config)
        vlm_hidden_dim = self.qwen_vl_interface.model.config.hidden_size

        # ── Action config ────────────────────────────────────────────
        action_cfg = config.framework.action_model
        self.action_dim = action_cfg.action_dim
        self.action_horizon = action_cfg.future_action_window_size + 1
        self.num_inference_timesteps = action_cfg.num_inference_timesteps
        self.vlm_hidden_dim = vlm_hidden_dim

        # ── Load Qwen3-0.6B as action expert ─────────────────────────
        expert_model_path = getattr(
            action_cfg, "expert_model_path",
            "playground/Pretrained_models/Qwen3-0.6B"
        )
        logger.info(f"Loading Qwen3 action expert from: {expert_model_path}")
        raw_expert = AutoModelForCausalLM.from_pretrained(
            expert_model_path, dtype=torch.bfloat16
        )
        expert_hidden_dim = raw_expert.config.hidden_size  # 1024

        # ── Prefix projection (VLM 2048 → expert 1024) ──────────────
        if vlm_hidden_dim != expert_hidden_dim:
            self.prefix_proj = nn.Sequential(
                nn.LayerNorm(vlm_hidden_dim),
                nn.Linear(vlm_hidden_dim, expert_hidden_dim),
            )
        else:
            self.prefix_proj = nn.Identity()

        # ── Action projections (pi0.5 style) ─────────────────────────
        self.action_in_proj = nn.Linear(self.action_dim, expert_hidden_dim)
        self.action_out_proj = nn.Linear(expert_hidden_dim, self.action_dim)

        # ── Timestep MLP: sinusoidal → Linear → SiLU → Linear ───────
        self.time_mlp = nn.Sequential(
            nn.Linear(expert_hidden_dim, expert_hidden_dim),
            nn.SiLU(),
            nn.Linear(expert_hidden_dim, expert_hidden_dim),
        )

        # ── Position embedding for action tokens ────────────────────
        max_action_len = int(getattr(action_cfg, "max_seq_len", 256))
        self.action_pos_embed = nn.Embedding(max_action_len, expert_hidden_dim)
        nn.init.normal_(self.action_pos_embed.weight, std=0.02)

        # ── Convert Qwen3 layers → ExpertLayers with AdaRMSNorm ─────
        self.expert_layers = nn.ModuleList()
        for layer in raw_expert.model.layers:
            self.expert_layers.append(
                Qwen3ExpertLayer(layer, cond_dim=expert_hidden_dim)
            )
        # Final norm (also adaptive)
        self.expert_final_norm = AdaRMSNorm.from_pretrained_rmsnorm(
            raw_expert.model.norm, cond_dim=expert_hidden_dim
        )

        # Keep rotary embedding for computing (cos, sin) position embeddings
        self.rotary_emb = raw_expert.model.rotary_emb

        # Free unused parts of raw expert (embed_tokens, lm_head)
        del raw_expert
        torch.cuda.empty_cache()

        # ── Flow matching ────────────────────────────────────────────
        noise_alpha = float(getattr(action_cfg, "noise_beta_alpha", 1.5))
        noise_beta = float(getattr(action_cfg, "noise_beta_beta", 1.0))
        self.noise_s = float(getattr(action_cfg, "noise_s", 0.999))
        self.beta_dist = Beta(noise_alpha, noise_beta)

        self.expert_hidden_dim = expert_hidden_dim

        logger.info(
            f"QwenPI_v4 initialized: VLM={vlm_hidden_dim}, "
            f"Expert={expert_hidden_dim} ({len(self.expert_layers)} layers from {expert_model_path}), "
            f"action_dim={self.action_dim}, horizon={self.action_horizon}"
        )

    # ── State discretization (pi0.5: 256 bins into text) ─────────

    @staticmethod
    def state2str(state: np.ndarray) -> str:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        state = np.clip(state, -1.0, 1.0)
        bins = np.digitize(state, np.linspace(-1, 1, 257)[:-1]) - 1
        return " ".join(map(str, bins.tolist()))

    def _inject_state_into_instruction(
        self, instructions: List[str], states: Optional[list]
    ) -> List[str]:
        """Pi0.5 format: 'Task: <instr>, State: <bins>;\n'"""
        if states is None:
            return instructions
        result = []
        for instr, state in zip(instructions, states):
            s = np.asarray(state)
            if s.ndim > 1:
                s = s[-1]
            result.append(f"Task: {instr}, State: {self.state2str(s)};\n")
        return result

    # ── Timestep ─────────────────────────────────────────────────

    def _embed_timestep(self, t: torch.Tensor) -> torch.Tensor:
        emb = sinusoidal_embedding(t, self.expert_hidden_dim)
        return self.time_mlp(emb)

    # ── Prefix encoding ─────────────────────────────────────────

    def _encode_qwen_hidden(self, batch_images, instructions):
        """Qwen-VL forward → last hidden + projected prefix."""
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
        last_hidden = outputs.hidden_states[-1]
        prefix_proj_dtype = next(
            (param.dtype for param in self.prefix_proj.parameters()),
            last_hidden.dtype,
        )
        last_hidden = last_hidden.to(dtype=prefix_proj_dtype)
        prefix_hidden = self.prefix_proj(last_hidden)
        return last_hidden, prefix_hidden, qwen_inputs

    def _encode_prefix(self, batch_images, instructions) -> torch.Tensor:
        _, prefix_hidden, _ = self._encode_qwen_hidden(batch_images, instructions)
        return prefix_hidden

    # ── Flow matching utilities ──────────────────────────────────

    def _sample_time(self, batch_size, device, dtype):
        t = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.noise_s - t) / self.noise_s

    def _get_repeated_diffusion_steps(self) -> int:
        ac = getattr(getattr(self.config, "framework", None), "action_model", None)
        if ac is not None:
            v = getattr(ac, "repeated_diffusion_steps", None)
            if v is not None:
                return int(v)
        return 4

    # ── Expert forward ───────────────────────────────────────────

    def _expert_forward(
        self,
        noisy_actions: torch.Tensor,     # [B, T, D]
        prefix_hidden: torch.Tensor,     # [B, S, H]
        time_cond: torch.Tensor,         # [B, H]
    ) -> torch.Tensor:
        """Run noisy action tokens through the modified Qwen3 expert."""
        B, T, _ = noisy_actions.shape

        # Project actions → expert hidden dim + positional embedding
        x = self.action_in_proj(noisy_actions)
        pos_ids = torch.arange(T, device=x.device, dtype=torch.long)
        x = x + self.action_pos_embed(pos_ids).unsqueeze(0)

        # Compute RoPE position embeddings (cos, sin) — Qwen3 style
        position_ids = pos_ids.unsqueeze(0).expand(B, -1)
        position_embeddings = self.rotary_emb(x, position_ids)  # (cos, sin)

        # Forward through all expert layers
        for layer in self.expert_layers:
            x = layer(x, prefix_hidden, time_cond, position_embeddings)

        x = self.expert_final_norm(x, time_cond)
        return self.action_out_proj(x)

    # ================================================================
    # Training forward
    # ================================================================

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        batch_images = [ex["image"] for ex in examples]
        instructions = [ex["lang"] for ex in examples]
        actions = [ex["action"] for ex in examples]
        action_valid_masks = [ex.get("action_valid_mask") for ex in examples]
        states = [ex["state"] for ex in examples] if "state" in examples[0] else None

        # Pi0.5: state always discretized into text
        instructions = self._inject_state_into_instruction(instructions, states)

        # Step 1: Prefix
        _, prefix_hidden, _ = self._encode_qwen_hidden(batch_images, instructions)

        # Step 2: Flow matching
        with torch.autocast("cuda", dtype=torch.float32):
            actions_tensor = torch.as_tensor(
                np.asarray(actions), device=prefix_hidden.device, dtype=prefix_hidden.dtype
            )
            actions_target = actions_tensor[:, -self.action_horizon:, :]
            action_valid_mask = None
            if any(mask is not None for mask in action_valid_masks):
                action_valid_mask = torch.as_tensor(
                    np.asarray(
                        [
                            mask if mask is not None else np.ones(action.shape[0], dtype=np.float32)
                            for action, mask in zip(actions, action_valid_masks)
                        ]
                    ),
                    device=prefix_hidden.device,
                    dtype=prefix_hidden.dtype,
                )
                action_valid_mask = action_valid_mask[:, -self.action_horizon:]

            R = self._get_repeated_diffusion_steps()
            actions_target = actions_target.repeat(R, 1, 1)
            prefix_hidden_r = prefix_hidden.repeat(R, 1, 1)
            if action_valid_mask is not None:
                action_valid_mask = action_valid_mask.repeat(R, 1)
            B = actions_target.shape[0]

            t = self._sample_time(B, actions_target.device, actions_target.dtype)
            noise = torch.randn_like(actions_target)
            t_expand = t[:, None, None]
            x_t = (1 - t_expand) * noise + t_expand * actions_target
            velocity_target = actions_target - noise

            # Step 3: Expert
            time_cond = self._embed_timestep(t)
            v_pred = self._expert_forward(x_t, prefix_hidden_r, time_cond)

            loss, _, _ = flow_matching_loss_with_endpoint_wrap(
                velocity_pred=v_pred,
                velocity_target=velocity_target,
                noisy_actions=x_t,
                target_actions=actions_target,
                t=t,
                config=self.config,
                action_valid_mask=action_valid_mask,
            )

            output = {
                "action_loss": loss,
                "primary_action_loss": loss.detach(),
            }

        return output

    # ================================================================
    # Inference
    # ================================================================

    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs) -> dict:
        diffusion_samples = max(1, int(kwargs.get("diffusion_samples", 1) or 1))
        if not isinstance(examples, list):
            examples = [examples]

        from deployment.model_server.tools.image_tools import to_pil_preserve

        batch_images = [to_pil_preserve(ex["image"]) for ex in examples]
        instructions = [ex["lang"] for ex in examples]
        states = [ex["state"] for ex in examples] if "state" in examples[0] else None

        instructions = self._inject_state_into_instruction(instructions, states)

        image_size_buckets = getattr(self.config.datasets.vla_data, "image_size_buckets", None)
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if image_size_buckets is None and train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        _, prefix_hidden, _ = self._encode_qwen_hidden(batch_images, instructions)

        with torch.autocast("cuda", dtype=torch.float32):
            B = prefix_hidden.shape[0]
            device, dtype = prefix_hidden.device, prefix_hidden.dtype
            pred_actions = None

            dt = 1.0 / self.num_inference_timesteps

            for _ in range(diffusion_samples):
                x_t = torch.randn(B, self.action_horizon, self.action_dim, device=device, dtype=dtype)

                for step in range(self.num_inference_timesteps):
                    t_val = step / float(self.num_inference_timesteps)
                    t = torch.full((B,), t_val, device=device, dtype=dtype)
                    time_cond = self._embed_timestep(t)

                    # Expert forward (prefix_hidden reused every step)
                    v_pred = self._expert_forward(x_t, prefix_hidden, time_cond)
                    x_t = x_t + dt * v_pred

                pred_actions = x_t if pred_actions is None else pred_actions + x_t

            x_t = pred_actions / diffusion_samples

        return {"normalized_actions": x_t.detach().cpu().numpy()}


# ============================================================
# Standalone test
# ============================================================

if __name__ == "__main__":
    from omegaconf import OmegaConf
    from PIL import Image
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str,
                        default="./examples/LIBERO/train_files/starvla_cotrain_libero.yaml")
    args, _ = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3-VL-4B-Instruct"
    cfg.framework.action_model.expert_model_path = "./playground/Pretrained_models/Qwen3-0.6B"

    model = Qwen_PI_v4(cfg)
    print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image, image],
        "lang": "Pick up the red cup and place it on the plate.",
        "state": np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    out = model([sample, sample])
    print(f"Action Loss: {out['action_loss'].item():.6f}")

    pred = model.predict_action([sample])
    print(f"Predicted actions shape: {pred['normalized_actions'].shape}")
