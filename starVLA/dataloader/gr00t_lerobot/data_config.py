# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#


import ast
from abc import ABC, abstractmethod

from omegaconf import OmegaConf

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform, ModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.concat import ConcatTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionSignFlipTransform,
    StateActionSinCosTransform,
    StateActionToTensor,
    StateActionTransform,
)
from starVLA.dataloader.gr00t_lerobot.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResizeBucketedSizes,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)
# from gr00t.model.transforms import GR00TTransform


class BaseDataConfig(ABC):
    @abstractmethod
    def modality_config(self) -> dict[str, ModalityConfig]:
        pass

    @abstractmethod
    def transform(self) -> ModalityTransform:
        pass


_DELTA_EEF_30FPS_ABS_LIMITS = [0.05, 0.05, 0.05, 0.1, 0.1, 0.1]
_DELTA_EEF_REFERENCE_FPS = 30.0


def _materialize_config_value(value):
    """Convert OmegaConf or tracked config wrappers into plain Python values."""
    wrapped_cfg = getattr(value, "_cfg", None)
    if wrapped_cfg is not None and OmegaConf.is_config(wrapped_cfg):
        value = wrapped_cfg

    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)

    if isinstance(value, list):
        return [_materialize_config_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_materialize_config_value(item) for item in value)

    if value is not None and hasattr(value, "__len__") and hasattr(value, "__getitem__") and not isinstance(
        value, (dict, str, bytes)
    ):
        try:
            return [_materialize_config_value(value[i]) for i in range(len(value))]
        except Exception:
            pass

    return value


def _resolve_image_size(data_cfg, default=(224, 224)):
    image_size = data_cfg.get("image_size", list(default)) if data_cfg else list(default)
    image_size = _materialize_config_value(image_size)
    if image_size is None:
        image_size = list(default)
    if len(image_size) != 2:
        raise ValueError(f"image_size must have 2 elements, got {image_size}")
    width, height = int(image_size[0]), int(image_size[1])
    return width, height


def _build_symmetric_min_max_stats(max_abs_values):
    max_values = [float(v) for v in max_abs_values]
    min_values = [-float(v) for v in max_values]
    return {
        "min": min_values,
        "max": max_values,
    }


def _build_delta_eef_velocity_defaults() -> dict[str, object]:
    position_limits_1fps = [
        float(v * _DELTA_EEF_REFERENCE_FPS) for v in _DELTA_EEF_30FPS_ABS_LIMITS[:3]
    ]
    rotation_limits_1fps = [
        float(v * _DELTA_EEF_REFERENCE_FPS) for v in _DELTA_EEF_30FPS_ABS_LIMITS[3:]
    ]
    return {
        "action_target_mode": "delta_eef_velocity",
        "action_mode": "delta",
        "action_mode_reference": "action",
        "action_mode_apply_keys": [
            "action.eef_position",
            "action.eef_rotation",
        ],
        "gripper_norm_mode": "min_max",
        "manual_action_normalization_statistics": {
            "action.eef_position": _build_symmetric_min_max_stats(position_limits_1fps),
            "action.eef_rotation": _build_symmetric_min_max_stats(rotation_limits_1fps),
            "action.gripper_position": {
                "min": [0.0],
                "max": [1.0],
            },
        },
        "delta_eef_position_abs_limit": max(position_limits_1fps),
        "delta_eef_rotation_abs_limit": max(rotation_limits_1fps),
    }


def _build_strict_franka_manual_velocity_defaults(
    *,
    position_action_key: str,
    rotation_action_key: str,
    gripper_action_key: str,
    filter_invalid_droid_task: bool = False,
) -> dict[str, object]:
    defaults = {
        "action_target_mode": "delta_eef_velocity",
        "action_type": "delta_eef",
        "action_mode": "delta",
        "action_mode_reference": "action",
        "action_mode_apply_keys": [
            position_action_key,
            rotation_action_key,
        ],
        "gripper_norm_mode": "min_max",
        "manual_action_normalization_statistics": {
            position_action_key: _build_symmetric_min_max_stats([0.5, 0.5, 0.5]),
            rotation_action_key: _build_symmetric_min_max_stats([1.0, 1.0, 1.0]),
            gripper_action_key: {
                "min": [0.0],
                "max": [1.0],
            },
        },
        "filter_delta_eef_steps": True,
        "filter_delta_eef_trajectory": False,
        "delta_eef_position_abs_limit": 0.5,
        "delta_eef_rotation_abs_limit": 1.0,
        "delta_eef_valid_ratio": 0.5,
    }
    if filter_invalid_droid_task:
        defaults["filter_invalid_droid_task"] = True
    return defaults


def _resolve_image_size_buckets(data_cfg):
    raw_buckets = data_cfg.get("image_size_buckets", None) if data_cfg else None
    raw_buckets = _materialize_config_value(raw_buckets)
    if raw_buckets is None:
        return None

    if isinstance(raw_buckets, str):
        raw_buckets = ast.literal_eval(raw_buckets)

    if not isinstance(raw_buckets, (list, tuple)) or not raw_buckets:
        raise ValueError(f"image_size_buckets must be a non-empty list, got {raw_buckets}")

    buckets = []
    for bucket in raw_buckets:
        if not isinstance(bucket, (list, tuple)) or len(bucket) != 2:
            raise ValueError(f"each image size bucket must have 2 elements, got {bucket}")
        width, height = int(bucket[0]), int(bucket[1])
        if width <= 0 or height <= 0:
            raise ValueError(f"image size bucket must be positive, got {(width, height)}")
        buckets.append((width, height))
    return buckets


def _build_configured_video_resize_transforms(data_cfg=None, video_keys: list[str] | None = None) -> list:
    if data_cfg is None or video_keys is None:
        return []

    image_size_buckets = _resolve_image_size_buckets(data_cfg)
    if image_size_buckets is not None:
        return [
            VideoResizeBucketedSizes(
                apply_to=video_keys,
                target_sizes=image_size_buckets,
                backend="albumentations",
                interpolation="linear",
            ),
        ]

    if data_cfg.get("image_size", None) is None:
        return []

    resize_width, resize_height = _resolve_image_size(data_cfg)
    return [
        VideoToTensor(apply_to=video_keys),
        VideoResize(apply_to=video_keys, height=resize_height, width=resize_width, interpolation="linear"),
        VideoToNumpy(apply_to=video_keys),
    ]



###########################################################################################

class OxeDroidDataConfig:
    video_keys = [
        "video.exterior_image_1",
        "video.exterior_image_2",
        "video.wrist_image",
    ]
    state_keys = [
        "state.joint_position",
        "state.gripper_position",
    ]
    action_keys = [
        "action.eef_position",
        "action.eef_rotation",
        "action.gripper_position",
    ]
    language_keys = ["annotation.language.language_instruction"]
    observation_indices = [0]
    action_indices = list(range(16))

    def dataset_defaults(self) -> dict[str, str]:
        return {
            "gripper_norm_mode": "min_max",
        }

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self, data_cfg=None):
        resize_width, resize_height = _resolve_image_size(data_cfg)
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoResize(apply_to=self.video_keys, height=resize_height, width=resize_width, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.joint_position": "min_max",
                    "state.gripper_position": "min_max",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.eef_position": "min_max",
                    "action.eef_rotation": "min_max",
                    "action.gripper_position": "min_max",
                },
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


###########################################################################################


class OxeBridgeDataConfig:
    video_keys = [
        "video.image_0",
    ]
    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.roll",
        "state.pitch",
        "state.yaw",
        "state.pad",
        "state.gripper",
    ]
    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            # VideoToTensor(apply_to=self.video_keys),
            # VideoCrop(apply_to=self.video_keys, scale=0.95),
            # VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            # VideoColorJitter(
            #     apply_to=self.video_keys,
            #     brightness=0.3,
            #     contrast=0.4,
            #     saturation=0.5,
            #     hue=0.08,
            # ),
            # VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.x": "q99",
                    "state.y": "q99",
                    "state.z": "q99",
                    "state.roll": "q99",
                    "state.pitch": "q99",
                    "state.yaw": "q99",
                    "state.pad": "q99",
                    "state.gripper": "binary",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.x": "q99",
                    "action.y": "q99",
                    "action.z": "q99",
                    "action.roll": "q99",
                    "action.pitch": "q99",
                    "action.yaw": "q99",
                    "action.gripper": "binary",
                },
            ),
            # concat transforms
            # ConcatTransform(
            #     # video_concat_order=self.video_keys,
            #     state_concat_order=self.state_keys,
            #     action_concat_order=self.action_keys,
            # ),
            # GR00TTransform(
            #     state_horizon=len(self.observation_indices),
            #     action_horizon=len(self.action_indices),
            #     max_state_dim=64,
            #     max_action_dim=32,
            # ),
        ]

        return ComposedModalityTransform(transforms=transforms)


###########################################################################################

class OxeRT1DataConfig:
    video_keys = [
        "video.image",
    ]
    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.rx",
        "state.ry",
        "state.rz",
        "state.rw",
        "state.gripper",
    ]
    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            # VideoToTensor(apply_to=self.video_keys),
            # VideoCrop(apply_to=self.video_keys, scale=0.95),
            # VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            # VideoColorJitter(
            #     apply_to=self.video_keys,
            #     brightness=0.3,
            #     contrast=0.4,
            #     saturation=0.5,
            #     hue=0.08,
            # ),
            # VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.x": "q99",
                    "state.y": "q99",
                    "state.z": "q99",
                    "state.rx": "q99",
                    "state.ry": "q99",
                    "state.rz": "q99",
                    "state.rw": "q99",
                    "state.gripper": "binary",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.x": "q99",
                    "action.y": "q99",
                    "action.z": "q99",
                    "action.roll": "q99",
                    "action.pitch": "q99",
                    "action.yaw": "q99",
                    "action.gripper": "binary",
                },
            ),
            # concat transforms
            # ConcatTransform(
            #     # video_concat_order=self.video_keys,
            #     state_concat_order=self.state_keys,
            #     action_concat_order=self.action_keys,
            # ),
            # GR00TTransform(
            #     state_horizon=len(self.observation_indices),
            #     action_horizon=len(self.action_indices),
            #     max_state_dim=64,
            #     max_action_dim=32,
            # ),
        ]

        return ComposedModalityTransform(transforms=transforms)


###########################################################################################


class SingleFrankaRobotiqDeltaEefDataConfig:
    video_keys = [
        "video.base_view",
        "video.ego_view",
    ]
    state_keys = [
        "state.eef_position",
        "state.eef_rotation",
    ]
    action_keys = [
        "action.delta_eef_position",
        "action.delta_eef_rotation",
        "action.gripper_close",
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.eef_position": "min_max",
                    "state.eef_rotation": "min_max",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.delta_eef_position": "min_max",
                    "action.delta_eef_rotation": "min_max",
                    "action.gripper_close": "binary",
                },
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)

###########################################################################################

class Libero4in1DataConfig:
    video_keys = [
        "video.primary_image",
        "video.wrist_image",
    ]
    
    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.roll",
        "state.pitch",
        "state.yaw",
        "state.pad",
        "state.gripper",
    ]
    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    ]
    
    language_keys = ["annotation.human.action.task_description"]

    observation_indices = [0]
    action_indices = list(range(8))
    state_indices = list(range(-16, 0))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.state_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
            apply_to=self.action_keys,
            normalization_modes={
                "action.x": "min_max",
                "action.y": "min_max",
                "action.z": "min_max",
                "action.roll": "min_max",
                "action.pitch": "min_max",
                "action.yaw": "min_max",
            },
        ),
        ]

        return ComposedModalityTransform(transforms=transforms)

###########################################################################################


class VLAArenaFrankaDataConfig:
    video_keys = [
        "video.primary_image",
        "video.wrist_image",
    ]
    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.roll",
        "state.pitch",
        "state.yaw",
        "state.gripper",
    ]
    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(8))
    state_indices = list(range(-16, 0))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.state_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        return {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

    def transform(self, data_cfg=None):
        transforms = [
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.x": "min_max",
                    "action.y": "min_max",
                    "action.z": "min_max",
                    "action.roll": "min_max",
                    "action.pitch": "min_max",
                    "action.yaw": "min_max",
                },
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


class SingleFrankaRobotiqDeltaJointsDataConfig:
    video_keys = [
        "video.base_view",
        "video.ego_view",
    ]
    state_keys = [
        "state.joints",
    ]
    action_keys = [
        "action.delta_joints",
        "action.gripper_close",
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.joints": "min_max",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.delta_joints": "min_max",
                    "action.gripper_close": "binary",
                },
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


###########################################################################################

class FourierGr1ArmsWaistDataConfig:
    video_keys = ["video.ego_view"]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        "state.waist",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
        "action.waist",
    ]
    language_keys = ["annotation.human.coarse_action"]
    observation_indices = [0]
    action_indices = list(range(16))


    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            # VideoToTensor(apply_to=self.video_keys),
            # VideoCrop(apply_to=self.video_keys, scale=0.95),
            # VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            # VideoColorJitter(
            #     apply_to=self.video_keys,
            #     brightness=0.3,
            #     contrast=0.4,
            #     saturation=0.5,
            #     hue=0.08,
            # ),
            # VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionSinCosTransform(apply_to=self.state_keys),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            # ConcatTransform(
            #     video_concat_order=self.video_keys,
            #     state_concat_order=self.state_keys,
            #     action_concat_order=self.action_keys,
            # ),
        ]
        return ComposedModalityTransform(transforms=transforms)


class OxeDroidExterior1WristDataConfig(OxeDroidDataConfig):
    video_keys = [
        "video.exterior_image_1",
        "video.wrist_image",
    ]


class OxeDroidExterior2WristDataConfig(OxeDroidDataConfig):
    video_keys = [
        "video.exterior_image_2",
        "video.wrist_image",
    ]


class OxeDroidExterior1WristData50Config(OxeDroidExterior1WristDataConfig):
    action_indices = list(range(50))


class OxeDroidExterior2WristData50Config(OxeDroidExterior2WristDataConfig):
    action_indices = list(range(50))


def _build_oxe_droid_no_jitter_transform(data_cfg, video_keys, state_keys, action_keys):
    transforms = [
        *_build_configured_video_resize_transforms(data_cfg=data_cfg, video_keys=video_keys),
        StateActionToTensor(apply_to=state_keys),
        StateActionTransform(
            apply_to=state_keys,
            normalization_modes={
                "state.joint_position": "min_max",
                "state.gripper_position": "min_max",
            },
        ),
        StateActionToTensor(apply_to=action_keys),
        StateActionTransform(
            apply_to=action_keys,
            normalization_modes={
                "action.eef_position": "min_max",
                "action.eef_rotation": "min_max",
                "action.gripper_position": "min_max",
            },
        ),
    ]
    return ComposedModalityTransform(transforms=transforms)


class OxeDroidExterior1WristManualVelocityData50Config(OxeDroidExterior1WristData50Config):
    def dataset_defaults(self) -> dict[str, object]:
        defaults = dict(super().dataset_defaults())
        defaults.update(_build_delta_eef_velocity_defaults())
        return defaults

    def transform(self, data_cfg=None):
        return _build_oxe_droid_no_jitter_transform(
            data_cfg=data_cfg,
            video_keys=self.video_keys,
            state_keys=self.state_keys,
            action_keys=self.action_keys,
        )


class OxeDroidExterior2WristManualVelocityData50Config(OxeDroidExterior2WristData50Config):
    def dataset_defaults(self) -> dict[str, object]:
        defaults = dict(super().dataset_defaults())
        defaults.update(_build_delta_eef_velocity_defaults())
        return defaults

    def transform(self, data_cfg=None):
        return _build_oxe_droid_no_jitter_transform(
            data_cfg=data_cfg,
            video_keys=self.video_keys,
            state_keys=self.state_keys,
            action_keys=self.action_keys,
        )


class OxeDroidExterior1WristStrictManualVelocityData50Config(OxeDroidExterior1WristManualVelocityData50Config):
    def dataset_defaults(self) -> dict[str, object]:
        defaults = dict(super().dataset_defaults())
        defaults.update(
            _build_strict_franka_manual_velocity_defaults(
                position_action_key="action.eef_position",
                rotation_action_key="action.eef_rotation",
                gripper_action_key="action.gripper_position",
                filter_invalid_droid_task=True,
            )
        )
        return defaults


class OxeDroidExterior2WristStrictManualVelocityData50Config(OxeDroidExterior2WristManualVelocityData50Config):
    def dataset_defaults(self) -> dict[str, object]:
        defaults = dict(super().dataset_defaults())
        defaults.update(
            _build_strict_franka_manual_velocity_defaults(
                position_action_key="action.eef_position",
                rotation_action_key="action.eef_rotation",
                gripper_action_key="action.gripper_position",
                filter_invalid_droid_task=True,
            )
        )
        return defaults



###########################################################################################


###########################################################################################

class SO101Config:
    #input
    video_keys = [
        "video.primary_image",
        "video.wrist_image",
    ]
    
    state_keys = [
        "state.shoulder_pan.pos",
        "state.shoulder_lift.pos",
        "state.elbow_flex.pos",
        "state.wrist_flex.pos",
        "state.wrist_roll.pos",
        "state.gripper.pos",
    ]
    language_keys = ["annotation.human.action.task_description"]

    # output
    action_keys = [
        "action.shoulder_pan.pos",
        "action.shoulder_lift.pos",
        "action.elbow_flex.pos",
        "action.wrist_flex.pos",
        "action.wrist_roll.pos",
        "action.gripper.pos",
    ]
    

    observation_indices = [0]
    action_indices = list(range(16))


    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    key: "min_max" for key in self.state_keys
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    key: "min_max" for key in self.action_keys
                },
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)



class ArxX5DataConfig:
    video_keys = [
        "video.cam_high",
        "video.cam_left_wrist",
        "video.cam_right_wrist",
    ]
    state_keys = [
        "state.left_joints",
        "state.right_joints",
        "state.left_gripper",
        "state.right_gripper",
    ]
    action_keys = [
        "action.left_joints",
        "action.right_joints",
        "action.left_gripper",
        "action.right_gripper",
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.left_joints": "min_max",
                    "state.right_joints": "min_max",
                    "state.left_gripper": "binary",
                    "state.right_gripper": "binary",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.left_joints": "min_max",
                    "action.right_joints": "min_max",
                    "action.left_gripper": "binary",
                    "action.right_gripper": "binary",
                },
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)

###########################################################################################


class AgilexDataConfig:
    video_keys = [
        "video.cam_high",
        "video.cam_left_wrist",
        "video.cam_right_wrist",
    ]
    state_keys = [
        "state.left_joints",
        "state.right_joints",
        "state.left_gripper",
        "state.right_gripper",
    ]
    action_keys = [
        "action.left_joints",
        "action.right_joints",#@JinhuiYE this order is different from Dataset
        "action.left_gripper",
        "action.right_gripper",
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self, data_cfg=None):
        return ComposedModalityTransform(
            transforms=[
                *_build_configured_video_resize_transforms(data_cfg=data_cfg, video_keys=self.video_keys),
                *_build_dual_arm_joint_gripper_transform(
                    state_keys=self.state_keys,
                    action_keys=self.action_keys,
                    binary_threshold=0.49,
                    gripper_mode="binary",
                ).transforms,
            ]
        )

###########################################################################################


class AgilexData50Config:
    video_keys = [
        "video.cam_high",
        "video.cam_left_wrist",
        "video.cam_right_wrist",
    ]
    state_keys = [
        "state.left_joints",
        "state.right_joints",
        "state.left_gripper",
        "state.right_gripper",
    ]
    action_keys = [
        "action.left_joints",
        "action.right_joints",  # @JinhuiYE this order is different from Dataset
        "action.left_gripper",
        "action.right_gripper",
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(50))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self, data_cfg=None):
        return ComposedModalityTransform(
            transforms=[
                *_build_configured_video_resize_transforms(data_cfg=data_cfg, video_keys=self.video_keys),
                *_build_dual_arm_joint_gripper_transform(
                    state_keys=self.state_keys,
                    action_keys=self.action_keys,
                    binary_threshold=0.49,
                    gripper_mode="binary",
                ).transforms,
            ]
        )

###########################################################################################


class AgilexData32Config(AgilexData50Config):
    action_indices = list(range(32))


class AgilexWrapDataConfig(AgilexDataConfig):
    def transform(self, data_cfg=None):
        return ComposedModalityTransform(
            transforms=[
                *_build_configured_video_resize_transforms(data_cfg=data_cfg, video_keys=self.video_keys),
                *_build_dual_arm_joint_gripper_transform(
                    state_keys=self.state_keys,
                    action_keys=self.action_keys,
                    binary_threshold=0.49,
                    gripper_mode="binary",
                    action_joint_mode="wrap",
                ).transforms,
            ]
        )


class AgilexWrapData50Config(AgilexData50Config):
    def transform(self, data_cfg=None):
        return ComposedModalityTransform(
            transforms=[
                *_build_configured_video_resize_transforms(data_cfg=data_cfg, video_keys=self.video_keys),
                *_build_dual_arm_joint_gripper_transform(
                    state_keys=self.state_keys,
                    action_keys=self.action_keys,
                    binary_threshold=0.49,
                    gripper_mode="binary",
                    action_joint_mode="wrap",
                ).transforms,
            ]
        )


class AgilexWrapData32Config(AgilexWrapData50Config):
    action_indices = list(range(32))


class InternA1FrankaDataConfig:
    video_keys = [
        "video.primary_image",
        "video.wrist_image",
    ]
    state_keys = [
        "state.eef_position",
        "state.eef_rotation",
        "state.gripper_position",
    ]
    action_keys = [
        "action.eef_position",
        "action.eef_rotation",
        "action.gripper_position",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def dataset_defaults(self) -> dict[str, object]:
        return {
            "action_type": "delta_ee",
            "action_mode": "delta",
            "action_mode_reference": "action",
            "action_mode_apply_keys": [
                "action.eef_position",
                "action.eef_rotation",
            ],
            "gripper_norm_mode": "cluster_continuous",
            "gripper_cluster_split_value": 0.5,
            "gripper_cluster_family_metric": "max",
            "gripper_cluster_low_close": 0.0,
            "gripper_cluster_low_open": 0.07999999821186066,
            "gripper_cluster_high_close": 0.0,
            "gripper_cluster_high_open": 1.0,
        }

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        return {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

    def transform(self, data_cfg=None):
        gripper_mode = _resolve_gripper_norm_mode(data_cfg, default="min_max")
        gripper_cluster_kwargs = _resolve_gripper_cluster_kwargs(data_cfg)
        transforms = [
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.eef_position": "min_max",
                    "state.gripper_position": gripper_mode,
                },
                target_rotations={
                    "state.eef_rotation": "axis_angle",
                },
                **gripper_cluster_kwargs,
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.eef_position": "min_max",
                    "action.eef_rotation": "min_max",
                    "action.gripper_position": gripper_mode,
                },
                **gripper_cluster_kwargs,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)



class MolmoActFrankaDataConfig(InternA1FrankaDataConfig):
    video_keys = [
        "video.exterior_image_1",
        "video.exterior_image_2",
        "video.wrist_image",
    ]
    action_keys = [
        "action.delta_eef_position",
        "action.delta_eef_rotation",
        "action.gripper_command",
    ]
    language_keys = ["annotation.language.language_instruction"]

    def dataset_defaults(self) -> dict[str, object]:
        defaults = dict(super().dataset_defaults())
        defaults.update(
            {
                "action_type": "delta_eef",
                "action_mode": "delta",
                "action_mode_reference": "action",
                "action_mode_apply_keys": [
                    "action.delta_eef_position",
                    "action.delta_eef_rotation",
                ],
                "gripper_norm_mode": "min_max",
            }
        )
        return defaults

    def transform(self, data_cfg=None):
        gripper_mode = _resolve_gripper_norm_mode(data_cfg, default="min_max")
        gripper_cluster_kwargs = _resolve_gripper_cluster_kwargs(data_cfg)
        transforms = [
            *_build_configured_video_resize_transforms(data_cfg=data_cfg, video_keys=self.video_keys),
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.eef_position": "min_max",
                    "state.gripper_position": gripper_mode,
                },
                target_rotations={
                    "state.eef_rotation": "axis_angle",
                },
                **gripper_cluster_kwargs,
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.delta_eef_position": "min_max",
                    "action.delta_eef_rotation": "min_max",
                    "action.gripper_command": gripper_mode,
                },
                **gripper_cluster_kwargs,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


class MolmoActFrankaData50Config(MolmoActFrankaDataConfig):
    action_indices = list(range(50))



class MolmoActFrankaManualVelocityData50Config(MolmoActFrankaData50Config):
    def dataset_defaults(self) -> dict[str, object]:
        position_limits_1fps = [
            float(v * _DELTA_EEF_REFERENCE_FPS) for v in _DELTA_EEF_30FPS_ABS_LIMITS[:3]
        ]
        rotation_limits_1fps = [
            float(v * _DELTA_EEF_REFERENCE_FPS) for v in _DELTA_EEF_30FPS_ABS_LIMITS[3:]
        ]
        defaults = dict(super().dataset_defaults())
        defaults.update(
            {
                "action_target_mode": "delta_eef_velocity",
                "action_mode_apply_keys": [
                    "action.delta_eef_position",
                    "action.delta_eef_rotation",
                ],
                "manual_action_normalization_statistics": {
                    "action.delta_eef_position": _build_symmetric_min_max_stats(position_limits_1fps),
                    "action.delta_eef_rotation": _build_symmetric_min_max_stats(rotation_limits_1fps),
                    "action.gripper_command": {
                        "min": [0.0],
                        "max": [1.0],
                    },
                },
                "delta_eef_position_abs_limit": max(position_limits_1fps),
                "delta_eef_rotation_abs_limit": max(rotation_limits_1fps),
            }
        )
        return defaults

    def transform(self, data_cfg=None):
        gripper_mode = _resolve_gripper_norm_mode(data_cfg, default="min_max")
        gripper_cluster_kwargs = _resolve_gripper_cluster_kwargs(data_cfg)
        transforms = [
            *_build_configured_video_resize_transforms(data_cfg=data_cfg, video_keys=self.video_keys),
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.eef_position": "min_max",
                    "state.gripper_position": gripper_mode,
                },
                target_rotations={
                    "state.eef_rotation": "axis_angle",
                },
                **gripper_cluster_kwargs,
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.delta_eef_position": "min_max",
                    "action.delta_eef_rotation": "min_max",
                    "action.gripper_command": gripper_mode,
                },
                **gripper_cluster_kwargs,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


class MolmoActFrankaExterior1WristManualVelocityData50Config(MolmoActFrankaManualVelocityData50Config):
    video_keys = [
        "video.exterior_image_1",
        "video.wrist_image",
    ]


class MolmoActFrankaExterior2WristManualVelocityData50Config(MolmoActFrankaManualVelocityData50Config):
    video_keys = [
        "video.exterior_image_2",
        "video.wrist_image",
    ]


class MolmoActFrankaExterior1WristStrictManualVelocityData50Config(MolmoActFrankaExterior1WristManualVelocityData50Config):
    def dataset_defaults(self) -> dict[str, object]:
        defaults = dict(super().dataset_defaults())
        defaults.update(
            _build_strict_franka_manual_velocity_defaults(
                position_action_key="action.delta_eef_position",
                rotation_action_key="action.delta_eef_rotation",
                gripper_action_key="action.gripper_command",
            )
        )
        return defaults


class MolmoActFrankaExterior2WristStrictManualVelocityData50Config(MolmoActFrankaExterior2WristManualVelocityData50Config):
    def dataset_defaults(self) -> dict[str, object]:
        defaults = dict(super().dataset_defaults())
        defaults.update(
            _build_strict_franka_manual_velocity_defaults(
                position_action_key="action.delta_eef_position",
                rotation_action_key="action.delta_eef_rotation",
                gripper_action_key="action.gripper_command",
            )
        )
        return defaults


###########################################################################################


class SplitAlohaDataConfig:
    video_keys = [
        "video.rgb_head",
        "video.rgb_hand_left",
        "video.rgb_hand_right",
    ]
    state_keys = [
        "state.left_joints",
        "state.right_joints",
        "state.left_gripper",
        "state.right_gripper",
    ]
    action_keys = [
        "action.left_joints",
        "action.right_joints",
        "action.left_gripper",
        "action.right_gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def dataset_defaults(self) -> dict[str, str]:
        return {
            "gripper_norm_mode": "min_max",
        }

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        return {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

    def transform(self, data_cfg=None):
        gripper_mode = _resolve_gripper_norm_mode(data_cfg, default="binary")
        return ComposedModalityTransform(
            transforms=[
                *_build_configured_video_resize_transforms(data_cfg=data_cfg, video_keys=self.video_keys),
                *_build_dual_arm_joint_gripper_transform(
                    state_keys=self.state_keys,
                    action_keys=self.action_keys,
                    binary_threshold=0.05,
                    gripper_mode=gripper_mode,
                ).transforms,
            ]
        )



class SplitAlohaWrapDataConfig(SplitAlohaDataConfig):
    def transform(self, data_cfg=None):
        gripper_mode = _resolve_gripper_norm_mode(data_cfg, default="binary")
        return ComposedModalityTransform(
            transforms=[
                *_build_configured_video_resize_transforms(data_cfg=data_cfg, video_keys=self.video_keys),
                *_build_dual_arm_joint_gripper_transform(
                    state_keys=self.state_keys,
                    action_keys=self.action_keys,
                    binary_threshold=0.05,
                    gripper_mode=gripper_mode,
                    action_joint_mode="wrap",
                ).transforms,
            ]
        )



class SplitAlohaJointFlipWrapDataConfig(SplitAlohaWrapDataConfig):
    flip_joint_indices = [2]

    def transform(self, data_cfg=None):
        gripper_mode = _resolve_gripper_norm_mode(data_cfg, default="min_max")
        return ComposedModalityTransform(
            transforms=[
                *_build_configured_video_resize_transforms(data_cfg=data_cfg, video_keys=self.video_keys),
                *_build_dual_arm_joint_sign_flip_transform(
                    state_keys=self.state_keys,
                    action_keys=self.action_keys,
                    joint_indices=self.flip_joint_indices,
                ),
                *_build_dual_arm_joint_gripper_transform(
                    state_keys=self.state_keys,
                    action_keys=self.action_keys,
                    binary_threshold=0.05,
                    gripper_mode=gripper_mode,
                    action_joint_mode="wrap",
                ).transforms,
            ]
        )



class SplitAlohaJointFlipWrapData50Config(SplitAlohaJointFlipWrapDataConfig):
    action_indices = list(range(50))



def _resolve_gripper_norm_mode(data_cfg=None, default: str = "binary") -> str:
    if data_cfg is None:
        return default
    mode = str(data_cfg.get("gripper_norm_mode", default)).lower()
    if mode not in {"binary", "min_max", "cluster_binary", "cluster_continuous"}:
        raise ValueError(f"Unsupported gripper_norm_mode: {mode}")
    return mode


def _resolve_gripper_cluster_kwargs(data_cfg=None) -> dict:
    defaults = {
        "gripper_cluster_split_value": 1.0,
        "gripper_cluster_family_metric": "max",
        "gripper_cluster_low_close": -0.01,
        "gripper_cluster_low_open": 0.43,
        "gripper_cluster_high_close": 0.16,
        "gripper_cluster_high_open": 5.25,
        "gripper_cluster_low_binary_threshold": 0.10,
        "gripper_cluster_high_binary_threshold": 1.0,
    }
    if data_cfg is None:
        return defaults
    resolved = {}
    for key, default_value in defaults.items():
        value = data_cfg.get(key, default_value)
        if isinstance(default_value, str):
            resolved[key] = str(value)
        else:
            resolved[key] = float(value)
    return resolved


def _build_dual_arm_joint_sign_flip_transform(
    *,
    state_keys: list[str],
    action_keys: list[str],
    joint_indices: list[int],
) -> list:
    flip_dims = {}
    for key in [*state_keys, *action_keys]:
        if key.endswith("left_joints") or key.endswith("right_joints"):
            flip_dims[key] = list(joint_indices)
    if not flip_dims:
        return []
    return [
        StateActionSignFlipTransform(
            apply_to=list(flip_dims.keys()),
            flip_dims=flip_dims,
        )
    ]


def _build_agilex_video_resize_transforms(data_cfg=None, video_keys: list[str] | None = None) -> list:
    """Backward-compatible wrapper for configured video resizing."""
    return _build_configured_video_resize_transforms(data_cfg=data_cfg, video_keys=video_keys)


def _build_dual_arm_joint_gripper_transform(
    *,
    state_keys: list[str],
    action_keys: list[str],
    binary_threshold: float,
    gripper_mode: str,
    gripper_cluster_kwargs: dict | None = None,
    state_joint_mode: str = "min_max",
    action_joint_mode: str = "min_max",
) -> ComposedModalityTransform:
    state_modes = {
        "state.left_joints": state_joint_mode,
        "state.right_joints": state_joint_mode,
        "state.left_gripper": gripper_mode,
        "state.right_gripper": gripper_mode,
    }
    action_modes = {
        "action.left_joints": action_joint_mode,
        "action.right_joints": action_joint_mode,
        "action.left_gripper": gripper_mode,
        "action.right_gripper": gripper_mode,
    }
    transforms = [
        StateActionToTensor(apply_to=state_keys),
        StateActionTransform(
            apply_to=state_keys,
            binary_threshold=binary_threshold,
            normalization_modes=state_modes,
            **(gripper_cluster_kwargs or {}),
        ),
        StateActionToTensor(apply_to=action_keys),
        StateActionTransform(
            apply_to=action_keys,
            binary_threshold=binary_threshold,
            normalization_modes=action_modes,
            **(gripper_cluster_kwargs or {}),
        ),
    ]
    return ComposedModalityTransform(transforms=transforms)



class RoboCoinDataConfig(AgilexData50Config):
    video_keys = [
        "video.cam_front_rgb",
        "video.cam_left_wrist_rgb",
        "video.cam_right_wrist_rgb",
    ]
    language_keys = ["annotation.task_index"]

    def dataset_defaults(self) -> dict[str, float | bool | str]:
        return {
            "filter_outlier_trajectory": True,
            "filter_gripper_outlier_trajectory": True,
            "gripper_outlier_abs_limit": 1.0,
            "gripper_norm_mode": "min_max",
            "outlier_abs_limit": 6.2831852,
        }

    def transform(self, data_cfg=None):
        gripper_mode = _resolve_gripper_norm_mode(data_cfg, default="binary")
        return ComposedModalityTransform(
            transforms=[
                *_build_configured_video_resize_transforms(data_cfg=data_cfg, video_keys=self.video_keys),
                *_build_dual_arm_joint_gripper_transform(
                    state_keys=self.state_keys,
                    action_keys=self.action_keys,
                    binary_threshold=0.49,
                    gripper_mode=gripper_mode,
                    action_joint_mode="wrap",
                ).transforms,
            ]
        )



class RoboCoinJointFlipWrapDataConfig(RoboCoinDataConfig):
    flip_joint_indices = [2]

    def transform(self, data_cfg=None):
        gripper_mode = _resolve_gripper_norm_mode(data_cfg, default="min_max")
        return ComposedModalityTransform(
            transforms=[
                *_build_configured_video_resize_transforms(data_cfg=data_cfg, video_keys=self.video_keys),
                *_build_dual_arm_joint_sign_flip_transform(
                    state_keys=self.state_keys,
                    action_keys=self.action_keys,
                    joint_indices=self.flip_joint_indices,
                ),
                *_build_dual_arm_joint_gripper_transform(
                    state_keys=self.state_keys,
                    action_keys=self.action_keys,
                    binary_threshold=0.49,
                    gripper_mode=gripper_mode,
                    action_joint_mode="wrap",
                ).transforms,
            ]
        )


ROBOT_TYPE_CONFIG_MAP = {
    "libero_franka": Libero4in1DataConfig(),
    "vla_arena_franka": VLAArenaFrankaDataConfig(),
    "oxe_droid": OxeDroidDataConfig(),
    "oxe_droid_exterior1_wrist_manualvel_strict_50": OxeDroidExterior1WristStrictManualVelocityData50Config(),
    "oxe_droid_exterior2_wrist_manualvel_strict_50": OxeDroidExterior2WristStrictManualVelocityData50Config(),
    "oxe_bridge": OxeBridgeDataConfig(),
    "oxe_rt1": OxeRT1DataConfig(),
    "SO101": SO101Config(),
    "demo_sim_franka_delta_joints": SingleFrankaRobotiqDeltaJointsDataConfig(),
    "arx_x5": ArxX5DataConfig(),
    "robotwin": AgilexDataConfig(),
    "robotwin32": AgilexData32Config(),
    "robotwin50": AgilexData50Config(),
    "robotwin_wrap": AgilexWrapDataConfig(),
    "robotwin_wrap32": AgilexWrapData32Config(),
    "robotwin_wrap50": AgilexWrapData50Config(),
    "ROBOCOIN.AgileX_flip_wrap": RoboCoinJointFlipWrapDataConfig(),
    "fourier_gr1_arms_waist": FourierGr1ArmsWaistDataConfig(),
    "molmoact_franka_exterior1_wrist_manualvel_strict_50": MolmoActFrankaExterior1WristStrictManualVelocityData50Config(),
    "molmoact_franka_exterior2_wrist_manualvel_strict_50": MolmoActFrankaExterior2WristStrictManualVelocityData50Config(),
    "split_aloha_flip_wrap50": SplitAlohaJointFlipWrapData50Config(),
    
    "custom_robot_config": SingleFrankaRobotiqDeltaEefDataConfig(),
}

