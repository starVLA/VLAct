import json
from collections import deque
from pathlib import Path
from typing import Dict, Optional

import cv2 as cv
import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from starVLA.model.tools import read_mode_config

try:
    from examples.SimplerEnv.eval_files.adaptive_ensemble import AdaptiveEnsembler
except ImportError:
    AdaptiveEnsembler = None


ROBOTWIN_ARM_ACTION_INDICES = tuple(range(12))


class ModelClient:
    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "robotwin",
        horizon: int = 0,
        action_ensemble: bool = False,
        action_ensemble_horizon: Optional[int] = 3,
        image_size: Optional[list[int]] = None,
        image_size_buckets: Optional[list[list[int]]] = None,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha: float = 0.1,
        host: str = "127.0.0.1",
        port: int = 5694,
        action_mode: str = "abs",
        action_postprocess: str = "none",
        robot_tag: Optional[str] = None,
    ) -> None:
        self.client = WebsocketClientPolicy(host, port)
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key
        self.action_mode = action_mode
        self.action_postprocess = action_postprocess.lower()
        self.robot_tag = robot_tag
        self.joint_action_unnorm_mode = self._infer_joint_action_unnorm_mode(policy_ckpt_path)

        print(
            f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key}, "
            f"action_mode: {action_mode}, action_postprocess: {action_postprocess}, "
            f"joint_action_unnorm_mode: {self.joint_action_unnorm_mode}, robot_tag: {robot_tag} ***"
        )

        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.image_size = image_size
        self.image_size_buckets = image_size_buckets or []
        self.horizon = horizon
        self.action_ensemble = action_ensemble and (AdaptiveEnsembler is not None)
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon

        self.initial_state = None
        self.prev_action = None
        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        self.action_norm_stats = self.get_action_stats(
            self.unnorm_key, policy_ckpt_path=policy_ckpt_path, action_mode=action_mode
        )
        self.action_chunk_size = self.get_action_chunk_size(policy_ckpt_path=policy_ckpt_path)
        self.state_norm_stats = self.get_state_stats(self.unnorm_key, policy_ckpt_path=policy_ckpt_path)
        self.raw_actions = None

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0
        self.raw_actions = None
        self.initial_state = None
        self.prev_action = None

    def step(self, example: dict, step: int = 0) -> np.ndarray:
        state = example.get("state", None)

        if self.action_mode in ["delta", "rel"] and self.initial_state is None:
            if state is None:
                raise ValueError(f"action_mode='{self.action_mode}' requires state to be provided in example")
            self.initial_state = np.array(state).copy()

        task_description = example.get("lang", None)
        if example is not None and task_description != self.task_description:
            self.reset(task_description)
            if self.action_mode in ["delta", "rel"] and state is not None:
                self.initial_state = np.array(state).copy()

        images = [self._resize_image(image) for image in example["image"]]
        example["image"] = images
        example_copy = example.copy()
        example_copy.pop("state")
        if self.robot_tag is not None:
            example_copy["robot_tag"] = self.robot_tag
        vla_input = {
            "examples": [example_copy],
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }

        if step % self.action_chunk_size == 0 or self.raw_actions is None:
            response = self.client.predict_action(vla_input)
            try:
                normalized_actions = response["data"]["normalized_actions"][0]
            except KeyError:
                print(f"Response data: {response}")
                raise KeyError(f"Key 'normalized_actions' not found in response data: {response['data'].keys()}")

            raw_actions = self.unnormalize_actions(
                normalized_actions=normalized_actions,
                action_norm_stats=self.action_norm_stats,
                joint_action_unnorm_mode=self.joint_action_unnorm_mode,
            )
            if self.action_mode == "delta":
                self.raw_actions = self._delta_to_absolute(raw_actions)
            elif self.action_mode == "rel":
                self.raw_actions = self._rel_to_absolute(raw_actions)
            else:
                self.raw_actions = raw_actions
            self.raw_actions = self._postprocess_actions(self.raw_actions)

        current_action = self.raw_actions[step % self.action_chunk_size]
        if self.action_mode == "delta":
            self.prev_action = current_action.copy()
        current_action = current_action[[0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13]]
        return current_action

    @staticmethod
    def normalize_state(state: dict[str, np.ndarray], state_norm_stats: Dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        mask = np.array([True] * 12 + [False, False], dtype=bool)
        state_high, state_low = np.array(state_norm_stats["max"]), np.array(state_norm_stats["min"])
        normalized_state = np.where(
            mask,
            (state - state_low) / (state_high - state_low) * 2 - 1,
            state,
        )
        normalized_state = np.where(~mask, (normalized_state > 0.5).astype(normalized_state.dtype), normalized_state)
        return normalized_state

    @staticmethod
    def unnormalize_actions(
        normalized_actions: np.ndarray,
        action_norm_stats: Dict[str, np.ndarray],
        joint_action_unnorm_mode: str = "stats",
    ) -> np.ndarray:
        mask = np.array(action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool)), dtype=bool)
        raw_actions = np.array(normalized_actions, copy=True)
        clipped_actions = np.clip(raw_actions, -1, 1)
        actions = np.array(clipped_actions, copy=True)

        if np.any(mask):
            if joint_action_unnorm_mode == "wrap":
                actions[..., mask] = (raw_actions[..., mask] + np.pi) % (2 * np.pi) - np.pi
            else:
                action_high = np.array(action_norm_stats["max"])
                action_low = np.array(action_norm_stats["min"])
                actions[..., mask] = (
                    0.5
                    * (clipped_actions[..., mask] + 1)
                    * (action_high[mask] - action_low[mask])
                    + action_low[mask]
                )
        return actions

    def _delta_to_absolute(self, delta_actions: np.ndarray) -> np.ndarray:
        abs_actions = np.zeros_like(delta_actions)
        mask = self.action_norm_stats.get("mask", np.ones(delta_actions.shape[-1], dtype=bool))
        base = self.prev_action if self.prev_action is not None else self.initial_state
        for i in range(len(delta_actions)):
            abs_actions[i] = np.where(mask, delta_actions[i] + base, delta_actions[i])
            base = abs_actions[i]
        return abs_actions

    def _rel_to_absolute(self, rel_actions: np.ndarray) -> np.ndarray:
        abs_actions = np.zeros_like(rel_actions)
        mask = self.action_norm_stats.get("mask", np.ones(rel_actions.shape[-1], dtype=bool))
        for i in range(len(rel_actions)):
            abs_actions[i] = np.where(mask, rel_actions[i] + self.initial_state, rel_actions[i])
        return abs_actions

    def _postprocess_actions(self, actions: np.ndarray) -> np.ndarray:
        if self.action_postprocess == "wrap":
            wrapped_actions = np.array(actions, copy=True)
            wrapped_actions[..., ROBOTWIN_ARM_ACTION_INDICES] = (
                (wrapped_actions[..., ROBOTWIN_ARM_ACTION_INDICES] + np.pi) % (2 * np.pi) - np.pi
            )
            return wrapped_actions
        return actions

    @staticmethod
    def get_action_stats(unnorm_key: str, policy_ckpt_path, action_mode: str = "abs") -> dict:
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)
        unnorm_key = ModelClient._check_unnorm_key(norm_stats, unnorm_key)
        stats = norm_stats[unnorm_key]
        if action_mode in stats:
            mode_stats = stats[action_mode]
            return mode_stats.get("action", mode_stats)
        if "action" in stats:
            if action_mode != "abs":
                print(f"[WARNING] Statistics file only has abs mode, but {action_mode} was requested. Using abs stats.")
            return stats["action"]
        raise ValueError(f"Invalid statistics file format for key: {unnorm_key}")

    @staticmethod
    def get_state_stats(unnorm_key: str, policy_ckpt_path) -> dict:
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)
        unnorm_key = ModelClient._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["state"]

    @staticmethod
    def get_action_chunk_size(policy_ckpt_path):
        model_config, _ = read_mode_config(policy_ckpt_path)
        return model_config["framework"]["action_model"]["future_action_window_size"] + 1

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        target_size = self._get_target_image_size(image)
        return cv.resize(image, tuple(target_size), interpolation=cv.INTER_AREA)

    def _get_target_image_size(self, image: np.ndarray) -> list[int]:
        if self.image_size_buckets:
            height, width = image.shape[:2]
            aspect_ratio = width / float(height)
            target_size = min(
                self.image_size_buckets,
                key=lambda size: abs((size[0] / float(size[1])) - aspect_ratio),
            )
            return [int(target_size[0]), int(target_size[1])]
        if self.image_size is not None:
            return [int(self.image_size[0]), int(self.image_size[1])]
        return [224, 224]

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        if unnorm_key is None:
            unnorm_key = next(iter(norm_stats.keys()))
        if unnorm_key not in norm_stats:
            unnorm_key = next(iter(norm_stats.keys()))
        return unnorm_key

    @staticmethod
    def _infer_joint_action_unnorm_mode(policy_ckpt_path) -> str:
        return "wrap" if "wrap" in str(policy_ckpt_path).lower() else "stats"


def get_model(usr_args):
    policy_ckpt_path = usr_args.get("policy_ckpt_path")
    host = usr_args.get("host", "127.0.0.1")
    port = usr_args.get("port", 5694)
    unnorm_key = usr_args.get("unnorm_key", None)
    action_mode = usr_args.get("action_mode", "abs")
    action_postprocess = usr_args.get("action_postprocess", "none")
    robot_tag = usr_args.get("robot_tag", None)
    image_size = usr_args.get("image_size")
    image_size_buckets = None

    if policy_ckpt_path is None:
        raise ValueError("policy_ckpt_path must be provided in config")

    if image_size is None:
        image_size, image_size_buckets = _get_model_image_size(policy_ckpt_path)

    model = ModelClient(
        policy_ckpt_path=policy_ckpt_path,
        host=host,
        port=port,
        unnorm_key=unnorm_key,
        action_mode=action_mode,
        action_postprocess=action_postprocess,
        robot_tag=robot_tag,
        image_size=image_size,
        image_size_buckets=image_size_buckets,
    )
    model.control_frequency = _get_control_frequency(policy_ckpt_path)
    if model.control_frequency is not None:
        print(f"*** control_frequency_prompt: {model.control_frequency}Hz ***")
    return model


def reset_model(model):
    model.reset(task_description="")


def _get_model_image_size(policy_ckpt_path) -> tuple[list[int], Optional[list[list[int]]]]:
    model_config, _ = read_mode_config(Path(policy_ckpt_path))
    vla_data_cfg = ((model_config.get("datasets") or {}).get("vla_data") or {})

    image_size_buckets = vla_data_cfg.get("image_size_buckets") or []
    if image_size_buckets:
        image_size_buckets = [[int(size[0]), int(size[1])] for size in image_size_buckets]
        print(f"*** image_size_buckets: {image_size_buckets} (aspect-ratio matched) ***")
        return image_size_buckets[0], image_size_buckets
    else:
        image_size = [224, 224]

    image_size = [int(image_size[0]), int(image_size[1])]
    print(f"*** image_size: {image_size} ***")
    return image_size, None


def _get_control_frequency(policy_ckpt_path) -> Optional[int | float]:
    policy_ckpt_path = Path(policy_ckpt_path)
    current_ckpt_path = policy_ckpt_path

    while True:
        model_config, _ = read_mode_config(current_ckpt_path)
        vla_data_cfg = ((model_config.get("datasets") or {}).get("vla_data") or {})
        if vla_data_cfg.get("unified_norm", False):
            break

        pretrained_stats_path = vla_data_cfg.get("pretrained_stats_path")
        if not pretrained_stats_path:
            return None

        next_ckpt_path = Path(str(pretrained_stats_path))
        if not next_ckpt_path.is_absolute():
            next_ckpt_path = (current_ckpt_path.parents[4] / next_ckpt_path).resolve()
        if not next_ckpt_path.exists():
            return None
        current_ckpt_path = next_ckpt_path

    model_config, _ = read_mode_config(policy_ckpt_path)
    data_root_dir = (((model_config.get("datasets") or {}).get("vla_data") or {}).get("data_root_dir"))
    if not data_root_dir:
        return None

    data_root_path = Path(str(data_root_dir))
    if not data_root_path.is_absolute():
        data_root_path = (policy_ckpt_path.parents[4] / data_root_path).resolve()

    info_files = sorted(data_root_path.glob("*/meta/info.json"))
    if not info_files:
        return None

    with open(info_files[0], "r") as f:
        fps = json.load(f).get("fps")
    if fps is None:
        return None

    fps = float(fps)
    return int(fps) if fps.is_integer() else fps


def _format_instruction(instruction: str, control_frequency: Optional[int | float]) -> str:
    if control_frequency is None or "Control frequency:" in instruction:
        return instruction
    return f"{instruction} Control frequency: {control_frequency}Hz."


def eval(TASK_ENV, model, observation):
    instruction = _format_instruction(
        str(TASK_ENV.get_instruction()),
        getattr(model, "control_frequency", None),
    )

    head_img = observation["observation"]["head_camera"]["rgb"]
    left_img = observation["observation"]["left_camera"]["rgb"]
    right_img = observation["observation"]["right_camera"]["rgb"]
    images = [head_img, left_img, right_img]

    state = observation["joint_action"]["vector"]
    example = {
        "lang": instruction,
        "image": images,
        "state": state,
    }

    action = model.step(example, step=TASK_ENV.take_action_cnt)
    TASK_ENV.take_action(action)
