from collections import deque
from typing import Optional
import cv2 as cv
import numpy as np
from typing import Dict
from pathlib import Path

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from examples.SimplerEnv.eval_files.adaptive_ensemble import AdaptiveEnsembler
from starVLA.model.tools import read_mode_config


POLICY_SETUP_TO_UNNORM_CANDIDATES = {
    "franka": ("franka", "vla_arena_franka", "libero_franka"),
    "vla_arena_franka": ("vla_arena_franka", "franka", "libero_franka"),
}


class ModelClient:
    """
    StarVLA WebSocket policy client adapted for VLA-Arena environments.

    Connects to the starVLA deployment server and provides step-by-step
    inference for VLA-Arena simulation environments (robosuite-based,
    mounted Franka Panda, 7-DOF delta-EEF actions).
    """

    def __init__(
        self,
        policy_ckpt_path: str,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "franka",
        horizon: int = 0,
        action_ensemble: bool = True,
        action_ensemble_horizon: Optional[int] = 3,
        image_size: list[int] = [224, 224],
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha: float = 0.1,
        host: str = "127.0.0.1",
        port: int = 10093,
    ) -> None:
        self.client = WebsocketClientPolicy(host, port)
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key

        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")

        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.image_size = image_size
        self.horizon = horizon
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        self.ov2_codec_cfg = self.get_ov2_codec_cfg(policy_ckpt_path)
        self.codec_frame_offsets = tuple(self.ov2_codec_cfg.get("frame_offsets", [-2, -1, 0])) if self.ov2_codec_cfg else ()
        self.codec_frame_history = deque(maxlen=max((abs(int(x)) for x in self.codec_frame_offsets), default=0) + 1)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(
                self.action_ensemble_horizon, self.adaptive_ensemble_alpha
            )
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        self.action_norm_stats = self.get_action_stats(
            self.unnorm_key,
            policy_ckpt_path=policy_ckpt_path,
            policy_setup=self.policy_setup,
        )
        self.action_unnorm_mode = self.get_action_unnorm_mode(
            policy_ckpt_path=policy_ckpt_path
        )
        self.action_chunk_size = self.get_action_chunk_size(
            policy_ckpt_path=policy_ckpt_path
        )

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        self.codec_frame_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

    def step(
        self,
        example: dict,
        step: int = 0,
        **kwargs,
    ) -> dict[str, np.ndarray]:
        """
        Perform one step of inference for VLA-Arena.

        :param example: dict with keys "image" (list of np.ndarray HxWxC uint8)
                        and "lang" (str instruction)
        :param step: current timestep (used for action chunking)
        :return: dict with "raw_action" containing world_vector, rotation_delta,
                 open_gripper
        """
        task_description = example.get("lang", None)
        images = example["image"]  # list of images

        if task_description is not None and task_description != self.task_description:
            self.reset(task_description)

        images = [self._resize_image(image) for image in images]
        example = dict(example)
        example["image"] = images
        if self.ov2_codec_cfg:
            self.codec_frame_history.append(images)
            example["videos"] = self._build_ov2_codec_clips()
        example.setdefault("robot_tag", self._canonicalize_policy_setup(self.policy_setup))

        vla_input = {
            "examples": [example],
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }

        action_chunk_size = self.action_chunk_size
        if step % action_chunk_size == 0:
            response = self.client.predict_action(vla_input)
            try:
                normalized_actions = response["data"]["normalized_actions"]  # B, chunk, D
            except KeyError:
                print(f"Response data: {response}")
                raise KeyError(
                    f"Key 'normalized_actions' not found in response: {response['data'].keys()}"
                )

            normalized_actions = normalized_actions[0]
            self.raw_actions = self.unnormalize_actions(
                normalized_actions=normalized_actions,
                action_norm_stats=self.action_norm_stats,
                action_unnorm_mode=self.action_unnorm_mode,
            )

        raw_actions = self.raw_actions[step % action_chunk_size][None]

        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),  # [0,1]; 1=open, 0=close
        }

        return {"raw_action": raw_action}

    @staticmethod
    def unnormalize_actions(
        normalized_actions: np.ndarray,
        action_norm_stats: Dict[str, np.ndarray],
        action_unnorm_mode: str = "min_max",
    ) -> np.ndarray:
        use_q99 = (
            action_unnorm_mode == "q99"
            and "q01" in action_norm_stats
            and "q99" in action_norm_stats
        )

        if use_q99:
            action_low = np.array(action_norm_stats["q01"], dtype=np.float32)
            action_high = np.array(action_norm_stats["q99"], dtype=np.float32)
            if "min" in action_norm_stats and "max" in action_norm_stats:
                fallback_low = np.array(action_norm_stats["min"], dtype=np.float32)
                fallback_high = np.array(action_norm_stats["max"], dtype=np.float32)
                invalid_q99 = np.abs(action_high - action_low) < 1e-6
                action_low = np.where(invalid_q99, fallback_low, action_low)
                action_high = np.where(invalid_q99, fallback_high, action_high)
        else:
            action_low = np.array(action_norm_stats["min"], dtype=np.float32)
            action_high = np.array(action_norm_stats["max"], dtype=np.float32)

        mask = action_norm_stats.get(
            "mask", np.ones_like(action_low, dtype=bool)
        )
        normalized_actions = np.clip(normalized_actions, -1, 1)
        # Binarize gripper channel
        normalized_actions[:, 6] = np.where(
            normalized_actions[:, 6] < 0.5, 0, 1
        )
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        return actions

    @staticmethod
    def get_action_stats(unnorm_key: str, policy_ckpt_path, policy_setup: str = "franka") -> dict:
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)
        unnorm_key = ModelClient._check_unnorm_key(
            norm_stats, unnorm_key, policy_setup=policy_setup
        )
        return norm_stats[unnorm_key]["action"]

    @staticmethod
    def get_action_unnorm_mode(policy_ckpt_path) -> str:
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, _ = read_mode_config(policy_ckpt_path)
        vla_data_cfg = ((model_config.get("datasets") or {}).get("vla_data") or {})
        mode_hints = (
            str(vla_data_cfg.get("data_mix", "")),
            str(vla_data_cfg.get("pretrained_stats_path", "")),
        )
        if any("q99" in hint.lower() for hint in mode_hints):
            return "q99"
        return "min_max"

    @staticmethod
    def get_action_chunk_size(policy_ckpt_path) -> int:
        model_config, _ = read_mode_config(policy_ckpt_path)
        return model_config["framework"]["action_model"]["future_action_window_size"] + 1

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        return cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)

    def _build_ov2_codec_clips(self) -> list[list[np.ndarray]]:
        history = list(self.codec_frame_history)
        if not history:
            return []
        num_views = len(history[-1])
        clips = []
        for view_idx in range(num_views):
            clip = []
            for offset in self.codec_frame_offsets:
                frame_idx = max(0, min(len(history) - 1, len(history) - 1 + int(offset)))
                clip.append(history[frame_idx][view_idx])
            clips.append(clip)
        return clips

    @staticmethod
    def get_ov2_codec_cfg(policy_ckpt_path) -> dict:
        model_config, _ = read_mode_config(Path(policy_ckpt_path))
        vla_data_cfg = ((model_config.get("datasets") or {}).get("vla_data") or {})
        codec_cfg = vla_data_cfg.get("ov2_codec") or {}
        enabled = codec_cfg.get("enabled", False)
        if isinstance(enabled, str):
            enabled = enabled.strip().lower() not in {"", "0", "false", "none", "no"}
        return codec_cfg if enabled else {}

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key, policy_setup: str = "franka"):
        available_keys = list(norm_stats.keys())
        if not available_keys:
            raise ValueError("No norm_stats found in checkpoint.")

        candidate_keys = []
        if unnorm_key is not None:
            candidate_keys.append(unnorm_key)
        else:
            candidate_keys.extend(
                POLICY_SETUP_TO_UNNORM_CANDIDATES.get(
                    policy_setup,
                    (policy_setup, ModelClient._canonicalize_policy_setup(policy_setup)),
                )
            )
            if len(available_keys) == 1:
                candidate_keys.append(available_keys[0])

        for candidate in dict.fromkeys(candidate_keys):
            if candidate in norm_stats:
                return candidate

        if unnorm_key is None:
            raise AssertionError(
                "Model trained on multiple datasets; unable to infer unnorm_key "
                f"for policy_setup '{policy_setup}'. Available keys: {available_keys}"
            )
        raise AssertionError(
            f"unnorm_key '{unnorm_key}' not found; available: {available_keys}"
        )

    @staticmethod
    def _canonicalize_policy_setup(policy_setup: str) -> str:
        if policy_setup in ("franka", "vla_arena_franka", "libero_franka"):
            return "franka"
        return policy_setup
