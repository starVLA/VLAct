from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from deployment.model_server.tools.image_tools import to_pil_preserve
from torch.distributions import Beta
from transformers import AutoModelForCausalLM

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.QwenPI_v4 import AdaRMSNorm, Qwen3ExpertLayer, sinusoidal_embedding
from starVLA.model.modules.action_model.GR00T_ActionHeader import (
    FlowmatchingActionHead,
    get_action_model as get_gr00t_action_model,
)
from starVLA.model.modules.action_model.flow_matching_loss import (
    blend_angular_wrap_abs_error,
    flow_matching_loss_with_endpoint_wrap,
    masked_mean,
)
from starVLA.model.modules.action_model.MLP_ActionHeader import (
    get_action_model as get_oft_action_model,
)
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY, collate_fn_extend_dim
from starVLA.training.trainer_utils.trainer_tools import resize_images


MULTI_ROBOT_ACTION_SPECS = {
    "franka": {
        "action_dim": 7,
        "robo_info": "single arm, delta eef",
    },
    "oxe_droid": {
        "action_dim": 7,
        "robo_info": "single arm, delta eef",
    },
    "interna1_split_aloha": {
        "action_dim": 14,
        "robo_info": "dual arm, joint and gripper control",
    },
    "robocoin": {
        "action_dim": 14,
        "robo_info": "dual arm, joint and gripper control",
    },
}

AUTO_ANGULAR_JOINT_LOSS_GRIPPER_DIMS = {
    "interna1_split_aloha": 2,
    "robocoin": 2,
}

DISJOINT_ACTION_LAYOUT_DIM = 20
DISJOINT_DUAL_ARM_JOINT_SLICE = slice(0, 12)
DISJOINT_SINGLE_ARM_EEF_SLICE = slice(12, 18)
DISJOINT_LEFT_GRIPPER_DIM = 18
DISJOINT_RIGHT_GRIPPER_DIM = 19


def _cfg_bool(value) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


class QwenPIActionHead(nn.Module):
    def __init__(self, config, vlm_hidden_dim: int):
        super().__init__()
        self.config = config
        action_cfg = config.framework.action_model
        self.action_dim = action_cfg.action_dim
        self.action_horizon = action_cfg.future_action_window_size + 1
        self.num_inference_timesteps = action_cfg.num_inference_timesteps

        expert_model_path = getattr(
            action_cfg,
            "expert_model_path",
            "playground/Pretrained_models/Qwen3-0.6B",
        )
        raw_expert = AutoModelForCausalLM.from_pretrained(
            expert_model_path, dtype=torch.bfloat16
        )
        expert_hidden_dim = raw_expert.config.hidden_size
        self.expert_hidden_dim = expert_hidden_dim

        if vlm_hidden_dim != expert_hidden_dim:
            self.prefix_proj = nn.Sequential(
                nn.LayerNorm(vlm_hidden_dim),
                nn.Linear(vlm_hidden_dim, expert_hidden_dim),
            )
        else:
            self.prefix_proj = nn.Identity()

        self.action_in_proj = nn.Linear(self.action_dim, expert_hidden_dim)
        self.action_out_proj = nn.Linear(expert_hidden_dim, self.action_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(expert_hidden_dim, expert_hidden_dim),
            nn.SiLU(),
            nn.Linear(expert_hidden_dim, expert_hidden_dim),
        )

        max_action_len = int(getattr(action_cfg, "max_seq_len", 256))
        self.action_pos_embed = nn.Embedding(max_action_len, expert_hidden_dim)
        nn.init.normal_(self.action_pos_embed.weight, std=0.02)

        self.expert_layers = nn.ModuleList(
            Qwen3ExpertLayer(layer, cond_dim=expert_hidden_dim)
            for layer in raw_expert.model.layers
        )
        self.expert_final_norm = AdaRMSNorm.from_pretrained_rmsnorm(
            raw_expert.model.norm, cond_dim=expert_hidden_dim
        )
        self.rotary_emb = raw_expert.model.rotary_emb

        noise_alpha = float(getattr(action_cfg, "noise_beta_alpha", 1.5))
        noise_beta = float(getattr(action_cfg, "noise_beta_beta", 1.0))
        self.noise_s = float(getattr(action_cfg, "noise_s", 0.999))
        self.beta_dist = Beta(noise_alpha, noise_beta)
        del raw_expert

    def _embed_timestep(self, t: torch.Tensor) -> torch.Tensor:
        emb = sinusoidal_embedding(t, self.expert_hidden_dim)
        return self.time_mlp(emb)

    def _sample_time(self, batch_size, device, dtype):
        t = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.noise_s - t) / self.noise_s

    def _expert_forward(
        self,
        noisy_actions: torch.Tensor,
        prefix_hidden: torch.Tensor,
        time_cond: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, action_len, _ = noisy_actions.shape
        x = self.action_in_proj(noisy_actions)
        pos_ids = torch.arange(action_len, device=x.device, dtype=torch.long)
        x = x + self.action_pos_embed(pos_ids).unsqueeze(0)
        position_ids = pos_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.rotary_emb(x, position_ids)

        for layer in self.expert_layers:
            x = layer(x, prefix_hidden, time_cond, position_embeddings)

        x = self.expert_final_norm(x, time_cond)
        return self.action_out_proj(x)

    def forward(
        self,
        vl_embs: torch.Tensor,
        actions: torch.Tensor,
        action_valid_mask: Optional[torch.Tensor] = None,
        action_dim_mask: Optional[torch.Tensor] = None,
        angular_dim_mask: Optional[torch.Tensor] = None,
        action_loss_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        prefix_hidden = self.prefix_proj(vl_embs)
        batch_size = actions.shape[0]
        t = self._sample_time(batch_size, actions.device, actions.dtype)
        noise = torch.randn_like(actions)
        x_t = (1 - t[:, None, None]) * noise + t[:, None, None] * actions
        velocity_target = actions - noise
        velocity_pred = self._expert_forward(x_t, prefix_hidden, self._embed_timestep(t))

        loss, _, _ = flow_matching_loss_with_endpoint_wrap(
            velocity_pred=velocity_pred,
            velocity_target=velocity_target,
            noisy_actions=x_t,
            target_actions=actions,
            t=t,
            config=self.config,
            action_valid_mask=action_valid_mask,
            action_dim_mask=action_dim_mask,
            angular_dim_mask=angular_dim_mask,
            sample_weights=action_loss_weights,
        )
        return loss

    @torch.no_grad()
    def predict_action(self, vl_embs: torch.Tensor) -> torch.Tensor:
        prefix_hidden = self.prefix_proj(vl_embs)
        batch_size = prefix_hidden.shape[0]
        device, dtype = prefix_hidden.device, prefix_hidden.dtype
        actions = torch.randn(
            batch_size,
            self.action_horizon,
            self.action_dim,
            device=device,
            dtype=dtype,
        )
        dt = 1.0 / self.num_inference_timesteps
        for step in range(self.num_inference_timesteps):
            t = torch.full(
                (batch_size,),
                step / float(self.num_inference_timesteps),
                device=device,
                dtype=dtype,
            )
            actions = actions + dt * self._expert_forward(
                actions,
                prefix_hidden,
                self._embed_timestep(t),
            )
        return actions


@FRAMEWORK_REGISTRY.register("QwenHybrid_xrobot_padding")
class QwenHybrid_xrobot_padding(baseframework):
    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.disjoint_action_layout = _cfg_bool(
            getattr(config.framework, "disjoint_action_layout", False)
        )
        self.mask_padded_action_dims = _cfg_bool(
            getattr(config.framework, "mask_padded_action_dims", False)
        )
        if self.disjoint_action_layout:
            config.framework.action_model.action_dim = max(
                int(config.framework.action_model.action_dim),
                DISJOINT_ACTION_LAYOUT_DIM,
            )
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        self.config.framework.action_model.action_hidden_dim = self.qwen_vl_interface.model.config.hidden_size
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = (
            self.qwen_vl_interface.model.config.hidden_size
        )

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        self.single_arm_gripper_to_dim12 = (
            _cfg_bool(getattr(config.framework, "single_arm_gripper_to_dim12", False))
            and not self.disjoint_action_layout
        )
        self.use_robo_meta_prompt = _cfg_bool(getattr(config.framework, "use_robo_meta_prompt", False))
        self.single_arm_loss_weight = float(getattr(config.framework, "single_arm_loss_weight", 1.0))
        self.dual_arm_loss_weight = float(getattr(config.framework, "dual_arm_loss_weight", 1.0))

        heads_cfg = getattr(config.framework, "heads", "oft")
        self.heads = self._parse_heads(heads_cfg)
        if not self.heads:
            raise ValueError("framework.heads must contain at least one head")
        self.use_multi_head = len(self.heads) > 1
        self.inference_head = self.heads[0]
        self.head_loss_weights = self._parse_head_loss_weights(
            getattr(config.framework, "head_loss_weights", None)
        )

        self.oft_action_model = get_oft_action_model(config=self.config) if "oft" in self.heads else None
        self.gr00t_action_model: Optional[FlowmatchingActionHead] = (
            get_gr00t_action_model(config=self.config) if "gr00t" in self.heads else None
        )
        self.pi_action_model = (
            QwenPIActionHead(self.config, self.qwen_vl_interface.model.config.hidden_size)
            if "pi" in self.heads
            else None
        )

        self.action_token = "🔍"
        self.action_token_id = self.qwen_vl_interface.processor.tokenizer(
            self.action_token, add_special_tokens=False
        )["input_ids"][0]
        self.l1_loss = nn.L1Loss()
        trainer_cfg = getattr(self.config, "trainer", None)
        shortest_angular_joint_loss = (
            trainer_cfg.get("shortest_angular_joint_loss", False) if trainer_cfg is not None else False
        )
        self.use_shortest_angular_joint_loss = str(shortest_angular_joint_loss).lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        shortest_angular_joint_loss_diff = (
            trainer_cfg.get("shortest_angular_joint_loss_diff", False) if trainer_cfg is not None else False
        )
        self.use_shortest_angular_joint_loss_diff = str(shortest_angular_joint_loss_diff).lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> dict:
        examples = collate_fn_extend_dim(
            examples, max_dim=self.config.framework.action_model.action_dim
        )
        if self.disjoint_action_layout:
            for example in examples:
                example["action"] = self._map_action_to_disjoint_layout(
                    example["action"],
                    example["robot_tag"],
                )
        elif self.single_arm_gripper_to_dim12:
            for example in examples:
                if self._get_robot_spec(example["robot_tag"])["action_dim"] == 7:
                    example["action"] = self._map_single_arm_action_to_dim12(example["action"])

        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        robot_tags = [example.get("robot_tag") for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None
        action_valid_masks = [example.get("action_valid_mask") for example in examples]

        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        instructions = self._build_model_instructions(examples, instructions, states=state)
        qwen_inputs, last_hidden = self._encode_inputs(batch_images, instructions)

        with torch.autocast("cuda", dtype=torch.float32):
            actions_target = torch.tensor(
                np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
            )
            actions_target = actions_target[:, -self.chunk_len :, :]
            state_tensor = self._prepare_state(state, device=last_hidden.device, dtype=last_hidden.dtype)
            action_valid_mask = self._build_action_valid_mask_tensor(
                action_valid_masks,
                device=last_hidden.device,
                dtype=last_hidden.dtype,
            )
            if action_valid_mask is not None:
                action_valid_mask = action_valid_mask[:, -actions_target.shape[1] :]
            action_dim_mask = self._build_action_dim_mask_tensor(
                robot_tags,
                action_dim=actions_target.shape[-1],
                device=last_hidden.device,
                dtype=last_hidden.dtype,
            )
            angular_dim_mask = self._infer_shortest_angular_joint_mask(
                robot_tags,
                actions_target,
                enabled=self.use_shortest_angular_joint_loss_diff,
            )
            action_loss_weights = self._build_action_loss_weight_tensor(
                robot_tags,
                device=last_hidden.device,
                dtype=last_hidden.dtype,
            )

            total_loss = last_hidden.new_zeros(())
            output = {}

            if self.oft_action_model is not None:
                input_ids = qwen_inputs.get("input_ids", None)
                action_queries = self._gather_action_token_embeddings(
                    last_hidden,
                    input_ids,
                    action_token_id=self.action_token_id,
                    chunk_len=self.chunk_len,
                )
                oft_pred_actions = self.oft_action_model.predict_action(action_queries)
                oft_action_loss = self._compute_action_loss(
                    oft_pred_actions,
                    actions_target,
                    robot_tags,
                    action_valid_mask=action_valid_mask,
                    action_dim_mask=action_dim_mask,
                    action_loss_weights=action_loss_weights,
                )
                total_loss = total_loss + self.head_loss_weights.get("oft", 1.0) * oft_action_loss
                output["oft_loss"] = oft_action_loss.detach()

            if self.gr00t_action_model is not None:
                repeated_diffusion_steps = self._get_repeated_diffusion_steps()
                actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
                last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)
                state_repeated = (
                    state_tensor.repeat(repeated_diffusion_steps, 1, 1) if state_tensor is not None else None
                )
                action_valid_mask_repeated = (
                    action_valid_mask.repeat(repeated_diffusion_steps, 1)
                    if action_valid_mask is not None
                    else None
                )
                action_dim_mask_repeated = (
                    action_dim_mask.repeat(repeated_diffusion_steps, 1)
                    if action_dim_mask is not None
                    else None
                )
                angular_dim_mask_repeated = (
                    angular_dim_mask.repeat(repeated_diffusion_steps, 1)
                    if angular_dim_mask is not None
                    else None
                )
                action_loss_weights_repeated = (
                    action_loss_weights.repeat(repeated_diffusion_steps)
                    if action_loss_weights is not None
                    else None
                )
                gr00t_action_loss = self.gr00t_action_model(
                    last_hidden_repeated,
                    actions_target_repeated,
                    state_repeated,
                    action_valid_mask=action_valid_mask_repeated,
                    action_dim_mask=action_dim_mask_repeated,
                    angular_dim_mask=angular_dim_mask_repeated,
                    action_loss_weights=action_loss_weights_repeated,
                )
                total_loss = total_loss + self.head_loss_weights.get("gr00t", 1.0) * gr00t_action_loss
                output["gr00t_loss"] = gr00t_action_loss.detach()

            if "pi" in self.heads:
                repeated_diffusion_steps = self._get_repeated_diffusion_steps()
                actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
                action_valid_mask_repeated = (
                    action_valid_mask.repeat(repeated_diffusion_steps, 1)
                    if action_valid_mask is not None
                    else None
                )
                action_dim_mask_repeated = (
                    action_dim_mask.repeat(repeated_diffusion_steps, 1)
                    if action_dim_mask is not None
                    else None
                )
                angular_dim_mask_repeated = (
                    angular_dim_mask.repeat(repeated_diffusion_steps, 1)
                    if angular_dim_mask is not None
                    else None
                )
                action_loss_weights_repeated = (
                    action_loss_weights.repeat(repeated_diffusion_steps)
                    if action_loss_weights is not None
                    else None
                )
                pi_in = last_hidden.repeat(repeated_diffusion_steps, 1, 1)
                pi_action_loss = self._pi_forward(
                    pi_in,
                    actions_target_repeated,
                    action_valid_mask=action_valid_mask_repeated,
                    action_dim_mask=action_dim_mask_repeated,
                    angular_dim_mask=angular_dim_mask_repeated,
                    action_loss_weights=action_loss_weights_repeated,
                )
                total_loss = total_loss + self.head_loss_weights.get("pi", 1.0) * pi_action_loss
                output["pi_loss"] = pi_action_loss.detach()

        output["action_loss"] = total_loss
        return output

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict] = None,
        **kwargs: str,
    ) -> dict:
        if type(examples) is not list:
            examples = [examples]

        examples = collate_fn_extend_dim(
            examples, max_dim=self.config.framework.action_model.action_dim
        )
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        instructions = self._build_model_instructions(examples, instructions, states=state)
        qwen_inputs, last_hidden = self._encode_inputs(batch_images, instructions)
        state_tensor = self._prepare_state(state, device=last_hidden.device, dtype=last_hidden.dtype)

        with torch.autocast("cuda", dtype=torch.float32):
            if self.inference_head == "gr00t":
                if self.gr00t_action_model is None:
                    raise RuntimeError("gr00t is selected for inference but not enabled")
                pred_actions = self.gr00t_action_model.predict_action(last_hidden, state_tensor)
            elif self.inference_head == "oft":
                if self.oft_action_model is None:
                    raise RuntimeError("oft is selected for inference but not enabled")
                input_ids = qwen_inputs.get("input_ids", None)
                action_queries = self._gather_action_token_embeddings(
                    last_hidden,
                    input_ids,
                    action_token_id=self.action_token_id,
                    chunk_len=self.chunk_len,
                )
                pred_actions = self.oft_action_model.predict_action(action_queries)
            elif self.inference_head == "pi":
                if self.pi_action_model is None:
                    raise RuntimeError("pi is selected for inference but not enabled")
                pred_actions = self._pi_predict(last_hidden)
            else:
                raise KeyError(f"Unsupported inference head `{self.inference_head}`")

        normalized_actions = pred_actions.detach().cpu().numpy()
        if self.disjoint_action_layout:
            restored_actions = [
                self._restore_action_from_disjoint_layout(
                    normalized_actions[idx],
                    example["robot_tag"],
                )
                for idx, example in enumerate(examples)
            ]
            normalized_actions = self._stack_restored_actions(restored_actions)
        elif self.single_arm_gripper_to_dim12:
            restored_actions = []
            for idx, example in enumerate(examples):
                action = normalized_actions[idx]
                if self._get_robot_spec(example["robot_tag"])["action_dim"] == 7:
                    action = self._restore_single_arm_action_from_dim12(action)
                restored_actions.append(action)

            normalized_actions = self._stack_restored_actions(restored_actions)
        normalized_actions = self._slice_predictions_by_robot(examples, normalized_actions)
        return {"normalized_actions": normalized_actions}

    def _pi_forward(self, pi_in, actions, **kwargs):
        return self.pi_action_model(pi_in, actions, **kwargs)

    def _pi_predict(self, pi_in):
        return self.pi_action_model.predict_action(pi_in)

    def _encode_inputs(self, batch_images, instructions):
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]
        return qwen_inputs, last_hidden

    def _build_model_instructions(self, examples, instructions, states=None):
        instructions = self._add_robo_meta_tokens_to_instructions(examples, instructions)
        if "pi" in self.heads:
            instructions = self._inject_state_into_instruction(instructions, states)
        if "oft" not in self.heads:
            return instructions

        # When OFT is enabled, reuse a single encoder pass by keeping the
        # action-token prompt for all active heads.
        enhanced_instructions = []
        action_tokens = self.action_token * self.chunk_len
        prompt_suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."
        for instruction in instructions:
            enhanced_instructions.append(instruction + prompt_suffix)
        return enhanced_instructions

    @staticmethod
    def state2str(state: np.ndarray) -> str:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        state = np.clip(state, -1.0, 1.0)
        bins = np.digitize(state, np.linspace(-1, 1, 257)[:-1]) - 1
        return " ".join(map(str, bins.tolist()))

    def _inject_state_into_instruction(self, instructions: List[str], states: Optional[list]) -> List[str]:
        if states is None:
            return instructions
        enhanced_instructions = []
        for instruction, state in zip(instructions, states):
            state_arr = np.asarray(state)
            if state_arr.ndim > 1:
                state_arr = state_arr[-1]
            enhanced_instructions.append(f"Task: {instruction}, State: {self.state2str(state_arr)};\n")
        return enhanced_instructions

    def _prepare_state(self, state, device, dtype):
        if state is None:
            return None
        return torch.tensor(np.array(state), device=device, dtype=dtype)

    def _compute_action_loss(
        self,
        pred_actions: torch.Tensor,
        actions_target: torch.Tensor,
        robot_tags: Optional[List[Optional[str]]] = None,
        action_valid_mask: Optional[torch.Tensor] = None,
        action_dim_mask: Optional[torch.Tensor] = None,
        action_loss_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raw_diff = pred_actions - actions_target
        abs_error = torch.abs(raw_diff)
        angular_mask = self._infer_shortest_angular_joint_mask(
            robot_tags,
            pred_actions,
            enabled=self.use_shortest_angular_joint_loss,
        )
        if angular_mask is not None:
            abs_error = blend_angular_wrap_abs_error(abs_error, raw_diff, angular_mask)
        if action_loss_weights is not None:
            sample_weights = action_loss_weights.to(device=abs_error.device, dtype=abs_error.dtype).view(-1, 1, 1)
            abs_error = abs_error * sample_weights
        if action_valid_mask is None and action_dim_mask is None:
            return abs_error.mean()

        loss_mask = torch.ones_like(abs_error)
        if action_valid_mask is not None:
            step_mask = action_valid_mask.to(device=abs_error.device, dtype=abs_error.dtype).unsqueeze(-1)
            loss_mask = loss_mask * step_mask
        if action_dim_mask is not None:
            dim_mask = action_dim_mask.to(device=abs_error.device, dtype=abs_error.dtype).unsqueeze(1)
            loss_mask = loss_mask * dim_mask

        denom = loss_mask.sum()
        if denom.item() <= 0:
            return pred_actions.sum() * 0.0
        return (abs_error * loss_mask).sum() / denom

    def _build_action_loss_weight_tensor(
        self,
        robot_tags: Optional[List[Optional[str]]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if self.single_arm_loss_weight == 1.0 and self.dual_arm_loss_weight == 1.0:
            return None
        if not robot_tags:
            raise ValueError("Action loss weighting requires robot tags.")

        weights = []
        for robot_tag in robot_tags:
            spec = self._get_robot_spec(robot_tag)
            weight = self.single_arm_loss_weight if spec["action_dim"] == 7 else self.dual_arm_loss_weight
            weights.append(weight)
        return torch.tensor(weights, device=device, dtype=dtype)

    def _build_action_valid_mask_tensor(
        self,
        action_valid_masks,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if not action_valid_masks or all(mask is None for mask in action_valid_masks):
            return None

        normalized_masks = []
        for mask in action_valid_masks:
            if mask is None:
                normalized_masks.append(np.ones((self.chunk_len,), dtype=np.float32))
                continue
            arr = np.asarray(mask, dtype=np.float32).reshape(-1)
            normalized_masks.append(arr)
        return torch.tensor(np.stack(normalized_masks, axis=0), device=device, dtype=dtype)

    def _build_action_dim_mask_tensor(
        self,
        robot_tags: Optional[List[Optional[str]]],
        *,
        action_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if not self.disjoint_action_layout and not self.mask_padded_action_dims:
            return None
        if not robot_tags:
            raise ValueError("Action dimension masking requires robot tags.")

        masks = []
        for robot_tag in robot_tags:
            spec = self._get_robot_spec(robot_tag)
            mask = np.zeros((action_dim,), dtype=np.float32)
            if self.disjoint_action_layout:
                if spec["action_dim"] == 7:
                    mask[DISJOINT_SINGLE_ARM_EEF_SLICE] = 1.0
                    mask[DISJOINT_LEFT_GRIPPER_DIM] = 1.0
                else:
                    mask[DISJOINT_DUAL_ARM_JOINT_SLICE] = 1.0
                    mask[DISJOINT_LEFT_GRIPPER_DIM] = 1.0
                    mask[DISJOINT_RIGHT_GRIPPER_DIM] = 1.0
            elif self.single_arm_gripper_to_dim12 and spec["action_dim"] == 7:
                mask[:6] = 1.0
                mask[12] = 1.0
            else:
                mask[: spec["action_dim"]] = 1.0
            masks.append(mask)
        return torch.tensor(np.stack(masks, axis=0), device=device, dtype=dtype)

    def _infer_shortest_angular_joint_mask(
        self,
        robot_tags: Optional[List[Optional[str]]],
        pred_actions: torch.Tensor,
        *,
        enabled: bool,
    ) -> Optional[torch.Tensor]:
        if not enabled or not robot_tags:
            return None

        batch_size, _, action_dim = pred_actions.shape
        masks = []
        for idx in range(batch_size):
            tag = str(robot_tags[idx]).lower() if idx < len(robot_tags) and robot_tags[idx] is not None else ""
            gripper_dims = AUTO_ANGULAR_JOINT_LOSS_GRIPPER_DIMS.get(tag)
            mask = torch.zeros(action_dim, dtype=torch.bool, device=pred_actions.device)
            if gripper_dims is not None and self.disjoint_action_layout:
                joint_start = DISJOINT_DUAL_ARM_JOINT_SLICE.start or 0
                joint_stop = min(DISJOINT_DUAL_ARM_JOINT_SLICE.stop, action_dim)
                if joint_start < joint_stop:
                    mask[joint_start:joint_stop] = True
            elif gripper_dims is not None and action_dim > gripper_dims:
                mask[: action_dim - gripper_dims] = True
            masks.append(mask)

        angular_mask = torch.stack(masks, dim=0)
        if not angular_mask.any():
            return None
        return angular_mask

    def _get_repeated_diffusion_steps(self) -> int:
        repeated_diffusion_steps = getattr(self.config.framework.action_model, "repeated_diffusion_steps", None)
        if repeated_diffusion_steps is not None:
            return int(repeated_diffusion_steps)

        trainer_cfg = getattr(self.config, "trainer", None)
        if trainer_cfg is not None:
            trainer_steps = trainer_cfg.get("repeated_diffusion_steps", None)
            if trainer_steps is not None:
                return int(trainer_steps)
        return 4

    @staticmethod
    def _parse_heads(heads_cfg) -> List[str]:
        if heads_cfg is None:
            return ["oft"]
        if isinstance(heads_cfg, str):
            raw_heads = heads_cfg.split(",")
        else:
            raw_heads = list(heads_cfg)

        heads = []
        for head in raw_heads:
            name = str(head).strip().lower()
            if name in ("pi_v3", "pi_v4"):
                name = "pi"
            if not name:
                continue
            if name not in {"oft", "gr00t", "pi"}:
                raise ValueError(f"Unsupported head `{head}`.")
            if name not in heads:
                heads.append(name)
        return heads

    @staticmethod
    def _parse_head_loss_weights(weights_cfg) -> dict:
        if weights_cfg is None:
            return {}
        if isinstance(weights_cfg, str):
            raw_items = [item.strip() for item in weights_cfg.split(",") if item.strip()]
            weights = {}
            for item in raw_items:
                name, value = item.split(":", 1)
                name = name.strip().lower()
                if name in ("pi_v3", "pi_v4"):
                    name = "pi"
                weights[name] = float(value)
            return weights
        weights = {}
        for name, value in dict(weights_cfg).items():
            name = str(name).lower()
            if name in ("pi_v3", "pi_v4"):
                name = "pi"
            weights[name] = float(value)
        return weights

    def _add_robo_meta_tokens_to_instructions(self, examples, instructions):
        if not self.use_robo_meta_prompt:
            return instructions
        enhanced_instructions = []
        for example, instruction in zip(examples, instructions):
            robot_tag = example["robot_tag"]
            robot_spec = self._get_robot_spec(robot_tag)
            robo_info = "Robot name: {}. Action Dim: {}. Robot info: {}".format(
                robot_tag, robot_spec["action_dim"], robot_spec["robo_info"]
            )
            prompt_suffix = (
                f" Please predict the next {self.chunk_len} robot actions for the robot "
                f"{robo_info}."
            )
            enhanced_instructions.append(instruction + prompt_suffix)
        return enhanced_instructions

    def _slice_predictions_by_robot(self, examples, normalized_actions: np.ndarray) -> np.ndarray:
        sliced_actions = []
        output_dims = []
        for idx, example in enumerate(examples):
            action_dim = self._get_robot_spec(example["robot_tag"])["action_dim"]
            output_dims.append(action_dim)
            sliced_actions.append(normalized_actions[idx, :, :action_dim])

        if len(set(output_dims)) == 1:
            return np.stack(sliced_actions, axis=0)

        max_dim = max(output_dims)
        padded_actions = []
        for action in sliced_actions:
            if action.shape[-1] == max_dim:
                padded_actions.append(action)
                continue
            pad_width = max_dim - action.shape[-1]
            padded_actions.append(
                np.concatenate(
                    [action, np.zeros((action.shape[0], pad_width), dtype=action.dtype)],
                    axis=-1,
                )
            )
        return np.stack(padded_actions, axis=0)

    def _map_single_arm_action_to_dim12(self, actions: np.ndarray) -> np.ndarray:
        if actions.shape[-1] < 13:
            raise ValueError(
                "single_arm_gripper_to_dim12 requires padded action dim >= 13, "
                f"got {actions.shape[-1]}"
            )
        remapped = np.zeros_like(actions)
        remapped[..., :6] = actions[..., :6]
        remapped[..., 12] = actions[..., 6]
        return remapped

    def _restore_single_arm_action_from_dim12(self, actions: np.ndarray) -> np.ndarray:
        if actions.shape[-1] < 13:
            raise ValueError(
                "single_arm_gripper_to_dim12 requires predicted action dim >= 13, "
                f"got {actions.shape[-1]}"
            )
        restored = np.zeros(actions.shape[:-1] + (7,), dtype=actions.dtype)
        restored[..., :6] = actions[..., :6]
        restored[..., 6] = actions[..., 12]
        return restored

    def _map_action_to_disjoint_layout(self, actions: np.ndarray, robot_tag: str) -> np.ndarray:
        if actions.shape[-1] < DISJOINT_ACTION_LAYOUT_DIM:
            raise ValueError(
                "disjoint_action_layout requires padded action dim >= "
                f"{DISJOINT_ACTION_LAYOUT_DIM}, got {actions.shape[-1]}"
            )

        robot_spec = self._get_robot_spec(robot_tag)
        remapped = np.zeros_like(actions)
        if robot_spec["action_dim"] == 7:
            remapped[..., DISJOINT_SINGLE_ARM_EEF_SLICE] = actions[..., :6]
            remapped[..., DISJOINT_LEFT_GRIPPER_DIM] = actions[..., 6]
        else:
            remapped[..., DISJOINT_DUAL_ARM_JOINT_SLICE] = actions[..., :12]
            remapped[..., DISJOINT_LEFT_GRIPPER_DIM] = actions[..., 12]
            remapped[..., DISJOINT_RIGHT_GRIPPER_DIM] = actions[..., 13]
        return remapped

    def _restore_action_from_disjoint_layout(self, actions: np.ndarray, robot_tag: str) -> np.ndarray:
        if actions.shape[-1] < DISJOINT_ACTION_LAYOUT_DIM:
            raise ValueError(
                "disjoint_action_layout requires predicted action dim >= "
                f"{DISJOINT_ACTION_LAYOUT_DIM}, got {actions.shape[-1]}"
            )

        robot_spec = self._get_robot_spec(robot_tag)
        if robot_spec["action_dim"] == 7:
            restored = np.zeros(actions.shape[:-1] + (7,), dtype=actions.dtype)
            restored[..., :6] = actions[..., DISJOINT_SINGLE_ARM_EEF_SLICE]
            restored[..., 6] = actions[..., DISJOINT_LEFT_GRIPPER_DIM]
            return restored

        restored = np.zeros(actions.shape[:-1] + (14,), dtype=actions.dtype)
        restored[..., :12] = actions[..., DISJOINT_DUAL_ARM_JOINT_SLICE]
        restored[..., 12] = actions[..., DISJOINT_LEFT_GRIPPER_DIM]
        restored[..., 13] = actions[..., DISJOINT_RIGHT_GRIPPER_DIM]
        return restored

    @staticmethod
    def _stack_restored_actions(restored_actions: list[np.ndarray]) -> np.ndarray:
        max_dim = max(action.shape[-1] for action in restored_actions)
        return np.stack(
            [
                action
                if action.shape[-1] == max_dim
                else np.concatenate(
                    [
                        action,
                        np.zeros((action.shape[0], max_dim - action.shape[-1]), dtype=action.dtype),
                    ],
                    axis=-1,
                )
                for action in restored_actions
            ],
            axis=0,
        )

    @staticmethod
    def _get_robot_spec(robot_tag: str) -> dict:
        if robot_tag not in MULTI_ROBOT_ACTION_SPECS:
            raise KeyError(
                f"Unsupported robot_tag `{robot_tag}` for QwenHybrid_xrobot_padding. "
                f"Expected one of {sorted(MULTI_ROBOT_ACTION_SPECS)}."
            )
        return MULTI_ROBOT_ACTION_SPECS[robot_tag]

    def _gather_action_token_embeddings(
        self,
        last_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        action_token_id=None,
        chunk_len=None,
    ) -> torch.Tensor:
        if action_token_id is None:
            raise ValueError("action_token_id cannot be None")
        if chunk_len is None:
            chunk_len = self.chunk_len

        device = input_ids.device
        batch_size, seq_len, hidden_dim = last_hidden.shape

        if isinstance(action_token_id, (list, tuple, set)):
            id_list = torch.tensor(list(action_token_id), device=device, dtype=input_ids.dtype)
            mask = torch.isin(input_ids, id_list)
        else:
            mask = input_ids == action_token_id

        counts = mask.sum(dim=1)
        if (counts < chunk_len).any():
            insufficient = (counts < chunk_len).nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(
                f"Samples do not contain enough action tokens for chunk_len={chunk_len}: {insufficient}"
            )

        idx = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        masked_pos = torch.where(mask, idx, torch.full_like(idx, -1))
        topk_pos = masked_pos.topk(k=chunk_len, dim=-1).values
        selected_pos = topk_pos.sort(dim=-1).values
        expanded_index = selected_pos.unsqueeze(-1).expand(-1, -1, hidden_dim)
        return last_hidden.gather(dim=1, index=expanded_index)
