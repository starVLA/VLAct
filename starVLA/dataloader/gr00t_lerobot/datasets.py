# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
In this file, we define 3 types of datasets:
1. LeRobotSingleDataset: a single dataset for a given embodiment tag
2. LeRobotMixtureDataset: a mixture of datasets for a given list of embodiment tags
3. CachedLeRobotSingleDataset: a single dataset for a given embodiment tag,
                                with caching for the video frames

See `scripts/load_dataset.py` for examples on how to use these datasets.
"""
import io
import os
import hashlib
import json, torch
import copy
from collections import defaultdict
from pathlib import Path
from typing import Sequence
import os, random
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
import torch.distributed as dist

from starVLA.dataloader.gr00t_lerobot.video import get_all_frames, get_frames_by_timestamps

from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.schema import (
    DatasetMetadata,
    DatasetStatisticalValues,
    LeRobotModalityMetadata,
    LeRobotActionMetadata,
    LeRobotModalityField,
    RotationType,
    LeRobotStateActionMetadata,
    LeRobotStateMetadata,
)
from starVLA.dataloader.gr00t_lerobot.transform import ComposedModalityTransform

from functools import partial
from typing import Tuple, List
import pickle
import gc

# LeRobot v2.0 dataset file names 
LE_ROBOT_MODALITY_FILENAME = "meta/modality.json"
LE_ROBOT_EPISODE_FILENAME = "meta/episodes.jsonl"
LE_ROBOT_TASKS_FILENAME = "meta/tasks.jsonl"
LE_ROBOT_INFO_FILENAME = "meta/info.json"
LE_ROBOT_STATS_FILENAME = "meta/stats_gr00t.json"
LE_ROBOT_DATA_FILENAME = "data/*/*.parquet"
LE_ROBOT_STEPS_FILENAME = "meta/steps.pkl"
EPSILON = 5e-4
DROID_INVALID_TASK_MARKERS = frozenset({"unknown_task", "n/a", "none", "no action", ""})
FRANKA_DELTA_EEF_ACTION_KEY = "actions.gripper.pose"
FRANKA_DELTA_EEF_STATE_KEY = "states.gripper.pose"
FRANKA_DELTA_EEF_POSITION_SLICE = slice(0, 3)
FRANKA_DELTA_EEF_ROTATION_SLICE = slice(3, 6)


def _normalize_droid_task_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def is_invalid_droid_task_text(value) -> bool:
    return _normalize_droid_task_text(value) in DROID_INVALID_TASK_MARKERS


def build_dataset_cache_key(
    *,
    dataset_name: str,
    filter_outlier_trajectory: bool,
    outlier_abs_limit: float,
    filter_gripper_outlier_trajectory: bool = False,
    gripper_outlier_abs_limit: float | None = None,
    filter_delta_eef_trajectory: bool = False,
    filter_delta_eef_steps: bool = False,
    delta_eef_position_abs_limit: float | None = None,
    delta_eef_rotation_abs_limit: float | None = None,
    delta_eef_valid_ratio: float | None = None,
    delete_pause_frame: bool = False,
    embodiment_tag: str | EmbodimentTag | None = None,
    robot_type: str | None = None,
    data_cfg: dict | None = None,
) -> str:
    config_dict = {
        "dataset_name": dataset_name,
        "delete_pause_frame": bool(delete_pause_frame),
        "filter_outlier_trajectory": bool(filter_outlier_trajectory),
        "outlier_abs_limit": float(outlier_abs_limit),
        "filter_gripper_outlier_trajectory": bool(filter_gripper_outlier_trajectory),
    }
    if gripper_outlier_abs_limit is not None:
        config_dict["gripper_outlier_abs_limit"] = float(gripper_outlier_abs_limit)
    if filter_delta_eef_steps:
        config_dict["filter_delta_eef_steps"] = True
        if delta_eef_position_abs_limit is not None:
            config_dict["delta_eef_position_abs_limit"] = float(delta_eef_position_abs_limit)
        if delta_eef_rotation_abs_limit is not None:
            config_dict["delta_eef_rotation_abs_limit"] = float(delta_eef_rotation_abs_limit)
        if delta_eef_valid_ratio is not None:
            config_dict["delta_eef_valid_ratio"] = float(delta_eef_valid_ratio)
    elif filter_delta_eef_trajectory:
        config_dict["filter_delta_eef_trajectory"] = True
        if delta_eef_position_abs_limit is not None:
            config_dict["delta_eef_position_abs_limit"] = float(delta_eef_position_abs_limit)
        if delta_eef_rotation_abs_limit is not None:
            config_dict["delta_eef_rotation_abs_limit"] = float(delta_eef_rotation_abs_limit)
    tag = embodiment_tag.value if isinstance(embodiment_tag, EmbodimentTag) else embodiment_tag
    robot_type_str = str(robot_type or tag or "")
    if robot_type_str.startswith("oxe_droid") and data_cfg:
        config_dict["filter_invalid_droid_task"] = bool(
            data_cfg.get("filter_invalid_droid_task", False)
        )
    config_str = str(sorted(config_dict.items()))
    return hashlib.md5(config_str.encode()).hexdigest()[:12]


#  LeRobot v3.0 dataset file names 
LE_ROBOT3_TASKS_FILENAME = "meta/tasks.parquet"
LE_ROBOT3_EPISODE_FILENAME = "meta/episodes/*/*.parquet"


def _get_allowed_step_indices_for_parquet(
    parquet_path: Path,
    allowed_step_indices_by_parquet: dict[str, np.ndarray] | None,
) -> np.ndarray | None:
    if allowed_step_indices_by_parquet is None:
        return None
    return allowed_step_indices_by_parquet.get(str(parquet_path))


def _filter_dataframe_by_allowed_indices(
    parquet_data: pd.DataFrame,
    allowed_indices: np.ndarray | None,
) -> pd.DataFrame:
    if allowed_indices is None:
        return parquet_data
    if parquet_data.empty:
        return parquet_data
    valid_indices = np.asarray(allowed_indices, dtype=np.int64)
    valid_indices = valid_indices[(valid_indices >= 0) & (valid_indices < len(parquet_data))]
    if valid_indices.size == 0:
        return parquet_data.iloc[0:0]
    return parquet_data.iloc[valid_indices]


def calculate_dataset_statistics(
    parquet_paths: list[Path],
    allowed_step_indices_by_parquet: dict[str, np.ndarray] | None = None,
) -> dict:
    """Calculate the dataset statistics of all columns for a list of parquet files."""
    # Dataset statistics
    all_low_dim_data_list = []
    # Collect all the data
    # parquet_paths = parquet_paths[:3]
    for parquet_path in tqdm(
        sorted(list(parquet_paths)),
        desc="Collecting all parquet files...",
    ):
        # Load the parquet file
        parquet_data = pd.read_parquet(parquet_path)
        parquet_data = _filter_dataframe_by_allowed_indices(
            parquet_data,
            _get_allowed_step_indices_for_parquet(parquet_path, allowed_step_indices_by_parquet),
        )
        if parquet_data.empty:
            continue
        all_low_dim_data_list.append(parquet_data)
    if not all_low_dim_data_list:
        raise ValueError("No rows available for statistics after step-level filtering.")
    all_low_dim_data = pd.concat(all_low_dim_data_list, axis=0)
    # Compute dataset statistics
    dataset_statistics = {}
    for le_modality in tqdm(all_low_dim_data.columns, desc="Processing modalities"):
        print(le_modality)
        if "task_info" in le_modality:
            continue
        print(f"Computing statistics for {le_modality}...")
        # 检查数据是否为空或无效
        try:
            np_data = np.vstack(
                [np.asarray(x, dtype=np.float32) for x in all_low_dim_data[le_modality]]
            )
        except Exception as e:
            print(f"Warning: Failed to process modality {le_modality} due to error: {e}")
            continue  

        dataset_statistics[le_modality] = {
            "mean": np.mean(np_data, axis=0).tolist(),
            "std": np.std(np_data, axis=0).tolist(),
            "min": np.min(np_data, axis=0).tolist(),
            "max": np.max(np_data, axis=0).tolist(),
            "q01": np.quantile(np_data, 0.01, axis=0).tolist(),
            "q99": np.quantile(np_data, 0.99, axis=0).tolist(),
        }
    return dataset_statistics


def _normalize_action_mode(mode: str) -> str:
    """Normalize action mode names to {abs, delta, rel}."""
    mode = str(mode).lower()
    if mode in {"absolute", "raw"}:
        mode = "abs"
    if mode not in {"abs", "delta", "rel"}:
        mode = "abs"
    return mode


def _is_rotation_key(action_key: str) -> bool:
    return "rotation" in action_key.lower()


def _wrap_rotation_delta(delta: np.ndarray) -> np.ndarray:
    """Wrap rotation delta values to [-pi, pi] to handle angle discontinuities."""
    return (delta + np.pi) % (2 * np.pi) - np.pi


def _get_action_col_slices(
    lerobot_modality_meta: "LeRobotModalityMetadata",
    action_keys_full: list[str],
    state_keys_full: list[str],
    action_mode_apply_keys: list[str] | None = None,
    action_mode_state_map: dict[str, str] | None = None,
) -> dict[str, list[tuple[str, tuple[int, int], str, tuple[int, int], str, str]]]:
    apply_keys = action_mode_apply_keys or action_keys_full
    action_mode_state_map = action_mode_state_map or {}

    action_meta = lerobot_modality_meta.action
    state_meta = lerobot_modality_meta.state

    # Build per-column mapping: action column -> list of (action_slice, state_column, state_slice)
    action_col_slices: dict[str, list[tuple[str, tuple[int, int], str, tuple[int, int], str, str]]] = {}
    for action_key in apply_keys:
        if not action_key.startswith("action."):
            raise ValueError(f"Invalid action key {action_key}. Expected prefix 'action.'.")
        state_key = action_mode_state_map.get(action_key, action_key.replace("action.", "state.", 1))
        if state_key not in state_keys_full:
            raise ValueError(
                f"State key {state_key} not found for action key {action_key}. "
                f"Add it to action_mode_state_map or remove {action_key} from action_mode_apply_keys."
            )

        action_subkey = action_key.replace("action.", "", 1)
        state_subkey = state_key.replace("state.", "", 1)
        if action_subkey not in action_meta or state_subkey not in state_meta:
            raise ValueError(f"Action/state key missing in metadata: {action_key} -> {state_key}")

        action_cfg = action_meta[action_subkey]
        state_cfg = state_meta[state_subkey]
        action_col = action_cfg.original_key or action_subkey
        state_col = state_cfg.original_key or state_subkey
        action_slice = (action_cfg.start, action_cfg.end)
        state_slice = (state_cfg.start, state_cfg.end)
        action_padding = "first_last" if action_cfg.absolute else "zero"
        state_padding = "first_last" if state_cfg.absolute else "zero"
        action_col_slices.setdefault(action_col, []).append(
            (action_key, action_slice, state_col, state_slice, action_padding, state_padding)
        )

    return action_col_slices


def _get_action_only_col_slices(
    lerobot_modality_meta: "LeRobotModalityMetadata",
    action_keys_full: list[str],
    action_mode_apply_keys: list[str] | None = None,
) -> dict[str, list[tuple[str, tuple[int, int], str]]]:
    apply_keys = action_mode_apply_keys or action_keys_full
    action_meta = lerobot_modality_meta.action

    action_col_slices: dict[str, list[tuple[str, tuple[int, int], str]]] = {}
    for action_key in apply_keys:
        if not action_key.startswith("action."):
            raise ValueError(f"Invalid action key {action_key}. Expected prefix 'action.'.")

        action_subkey = action_key.replace("action.", "", 1)
        if action_subkey not in action_meta:
            raise ValueError(f"Action key missing in metadata: {action_key}")

        action_cfg = action_meta[action_subkey]
        action_col = action_cfg.original_key or action_subkey
        action_slice = (action_cfg.start, action_cfg.end)
        action_padding = "first_last" if action_cfg.absolute else "zero"
        action_col_slices.setdefault(action_col, []).append((action_key, action_slice, action_padding))

    return action_col_slices


def _normalize_action_key_set(action_keys: list[str] | None) -> set[str]:
    normalized = set()
    for key in action_keys or []:
        key = str(key)
        if not key.startswith("action."):
            key = f"action.{key}"
        normalized.add(key)
    return normalized


class _StreamingArrayStatistics:
    """Exact mean/std/min/max with deterministic sampled quantiles."""

    def __init__(self, sample_limit: int = 1_000_000):
        self.sample_limit = int(sample_limit)
        self.count = 0
        self.sample_stride = 1
        self.sample_count = 0
        self.sum = None
        self.sumsq = None
        self.min = None
        self.max = None
        self.samples: list[np.ndarray] = []

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float32)
        if values.size == 0:
            return
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        values = values.reshape(-1, values.shape[-1])

        values64 = values.astype(np.float64, copy=False)
        if self.sum is None:
            width = values.shape[1]
            self.sum = np.zeros(width, dtype=np.float64)
            self.sumsq = np.zeros(width, dtype=np.float64)
            self.min = np.full(width, np.inf, dtype=np.float32)
            self.max = np.full(width, -np.inf, dtype=np.float32)

        self.sum += values64.sum(axis=0)
        self.sumsq += np.square(values64).sum(axis=0)
        self.min = np.minimum(self.min, values.min(axis=0))
        self.max = np.maximum(self.max, values.max(axis=0))

        start = self.count
        end = self.count + values.shape[0]
        sample_mask = (np.arange(start, end, dtype=np.int64) % self.sample_stride) == 0
        if sample_mask.any():
            sampled = values[sample_mask].copy()
            self.samples.append(sampled)
            self.sample_count += sampled.shape[0]
        self.count = end
        self._shrink_samples()

    def _shrink_samples(self) -> None:
        while self.sample_count > self.sample_limit:
            merged = np.concatenate(self.samples, axis=0)
            merged = merged[::2].copy()
            self.sample_stride *= 2
            self.samples = [merged]
            self.sample_count = merged.shape[0]

    def finalize(self) -> dict:
        if self.count == 0:
            raise ValueError("Cannot finalize empty statistics.")
        mean = self.sum / self.count
        var = np.maximum(self.sumsq / self.count - np.square(mean), 0.0)
        sample_values = np.concatenate(self.samples, axis=0)
        return {
            "mean": mean.tolist(),
            "std": np.sqrt(var).tolist(),
            "min": self.min.tolist(),
            "max": self.max.tolist(),
            "q01": np.quantile(sample_values, 0.01, axis=0).tolist(),
            "q99": np.quantile(sample_values, 0.99, axis=0).tolist(),
        }


def _calculate_action_only_statistics(
    *,
    parquet_paths: list[Path],
    lerobot_modality_meta: "LeRobotModalityMetadata",
    action_keys_full: list[str],
    action_indices: list[int],
    action_mode_apply_keys: list[str] | None,
    base_stats: dict | None,
    allowed_step_indices_by_parquet: dict[str, np.ndarray] | None,
    mode: str,
) -> dict:
    if base_stats is None:
        base_stats = calculate_dataset_statistics(
            parquet_paths,
            allowed_step_indices_by_parquet=allowed_step_indices_by_parquet,
        )

    action_col_slices = _get_action_only_col_slices(
        lerobot_modality_meta, action_keys_full, action_mode_apply_keys
    )
    if not action_col_slices:
        raise ValueError("No action columns found in the dataset.")
    if mode not in {"delta", "rel"}:
        raise ValueError(f"Unsupported action-only statistics mode: {mode}")

    def _get_chunk(array: np.ndarray, step_indices: np.ndarray, padding_strategy: str) -> np.ndarray:
        max_length = array.shape[0]
        front_padding = step_indices < 0
        end_padding = step_indices >= max_length
        padding_positions = np.logical_or(front_padding, end_padding)
        output = np.zeros((len(step_indices), array.shape[1]), dtype=array.dtype)
        if (~padding_positions).any():
            output[~padding_positions] = array[step_indices[~padding_positions]]
        if padding_positions.any():
            if padding_strategy == "first_last":
                output[front_padding] = array[0]
                output[end_padding] = array[-1]
            elif padding_strategy == "zero":
                output[padding_positions] = 0
            else:
                raise ValueError(f"Invalid padding strategy: {padding_strategy}")
        return output

    action_indices_array = np.asarray(action_indices, dtype=np.int64)
    accum = {col: _StreamingArrayStatistics() for col in action_col_slices.keys()}
    for parquet_path in tqdm(sorted(list(parquet_paths)), desc=f"Collecting action-only {mode} stats"):
        data = pd.read_parquet(parquet_path)
        trajectory_length = len(data)
        allowed_base_indices = _get_allowed_step_indices_for_parquet(
            parquet_path,
            allowed_step_indices_by_parquet,
        )
        if allowed_base_indices is None:
            base_indices_iter = range(trajectory_length)
        else:
            base_indices_iter = np.asarray(allowed_base_indices, dtype=np.int64)

        for action_col, slice_list in action_col_slices.items():
            if action_col not in data.columns:
                raise ValueError(f"{action_col} not found in parquet columns.")
            action_matrix = np.stack(data[action_col])
            if action_matrix.ndim == 1:
                action_matrix = action_matrix.reshape(-1, 1)
            action_padding_ref = slice_list[0][2]
            batch = []
            batch_rows = 0
            for base_index in base_indices_iter:
                action_steps = action_indices_array + int(base_index)
                action_chunk_full = _get_chunk(action_matrix, action_steps, action_padding_ref)

                for action_key, a_slice, _ in slice_list:
                    action_part_chunk = action_chunk_full[:, a_slice[0] : a_slice[1]]
                    if mode == "delta":
                        next_part_chunk = np.concatenate([action_part_chunk[1:], action_part_chunk[-1:]], axis=0)
                        out = next_part_chunk - action_part_chunk
                        if _is_rotation_key(action_key):
                            out = _wrap_rotation_delta(out)
                    else:
                        out = action_part_chunk - action_part_chunk[0]
                    action_chunk_full[:, a_slice[0] : a_slice[1]] = out

                batch.append(action_chunk_full)
                batch_rows += action_chunk_full.shape[0]
                if batch_rows >= 200_000:
                    accum[action_col].update(np.concatenate(batch, axis=0))
                    batch.clear()
                    batch_rows = 0
            if batch:
                accum[action_col].update(np.concatenate(batch, axis=0))

    output_stats = copy.deepcopy(base_stats)
    for action_col, stat in tqdm(accum.items(), desc=f"Finalizing action-only {mode} stats"):
        if stat.count == 0:
            continue
        output_stats[action_col] = stat.finalize()
    return output_stats


def calculate_action_only_delta_statistics(
    parquet_paths: list[Path],
    lerobot_modality_meta: "LeRobotModalityMetadata",
    action_keys_full: list[str],
    action_indices: list[int],
    action_mode_apply_keys: list[str] | None = None,
    base_stats: dict | None = None,
    allowed_step_indices_by_parquet: dict[str, np.ndarray] | None = None,
) -> dict:
    """Calculate forward-delta stats from absolute action trajectories without state references."""
    return _calculate_action_only_statistics(
        parquet_paths=parquet_paths,
        lerobot_modality_meta=lerobot_modality_meta,
        action_keys_full=action_keys_full,
        action_indices=action_indices,
        action_mode_apply_keys=action_mode_apply_keys,
        base_stats=base_stats,
        allowed_step_indices_by_parquet=allowed_step_indices_by_parquet,
        mode="delta",
    )


def calculate_action_only_rel_statistics(
    parquet_paths: list[Path],
    lerobot_modality_meta: "LeRobotModalityMetadata",
    action_keys_full: list[str],
    action_indices: list[int],
    action_mode_apply_keys: list[str] | None = None,
    base_stats: dict | None = None,
    allowed_step_indices_by_parquet: dict[str, np.ndarray] | None = None,
) -> dict:
    """Calculate rel stats from absolute action trajectories without state references."""
    return _calculate_action_only_statistics(
        parquet_paths=parquet_paths,
        lerobot_modality_meta=lerobot_modality_meta,
        action_keys_full=action_keys_full,
        action_indices=action_indices,
        action_mode_apply_keys=action_mode_apply_keys,
        base_stats=base_stats,
        allowed_step_indices_by_parquet=allowed_step_indices_by_parquet,
        mode="rel",
    )


def calculate_delta_action_statistics(
    parquet_paths: list[Path],
    lerobot_modality_meta: "LeRobotModalityMetadata",
    action_keys_full: list[str],
    state_keys_full: list[str],
    action_indices: list[int],
    state_indices: list[int],
    action_mode_apply_keys: list[str] | None = None,
    action_mode_state_map: dict[str, str] | None = None,
    base_stats: dict | None = None,
    allowed_step_indices_by_parquet: dict[str, np.ndarray] | None = None,
) -> dict:
    """
    Calculate action statistics using delta mode.

    Rule:
      - For t>0: a_t - a_{t-1}
      - For t=0: a_0 - s_0

    Mapping rule (only two cases):
      1) Use explicit action_mode_state_map if provided.
      2) Otherwise, replace 'action.' with 'state.' directly.
    """
    if base_stats is None:
        base_stats = calculate_dataset_statistics(
            parquet_paths,
            allowed_step_indices_by_parquet=allowed_step_indices_by_parquet,
        )

    action_col_slices = _get_action_col_slices(
        lerobot_modality_meta, action_keys_full, state_keys_full, action_mode_apply_keys, action_mode_state_map
    )
    if not action_col_slices:
        raise ValueError("No action columns found in the dataset.")

    def _get_chunk(array: np.ndarray, step_indices: np.ndarray, padding_strategy: str) -> np.ndarray:
        max_length = array.shape[0]
        front_padding = step_indices < 0
        end_padding = step_indices >= max_length
        padding_positions = np.logical_or(front_padding, end_padding)
        output = np.zeros((len(step_indices), array.shape[1]), dtype=array.dtype)
        if (~padding_positions).any():
            output[~padding_positions] = array[step_indices[~padding_positions]]
        if padding_positions.any():
            if padding_strategy == "first_last":
                output[front_padding] = array[0]
                output[end_padding] = array[-1]
            elif padding_strategy == "zero":
                output[padding_positions] = 0
            else:
                raise ValueError(f"Invalid padding strategy: {padding_strategy}")
        return output

    accum: dict[str, list[np.ndarray]] = {col: [] for col in action_col_slices.keys()}
    for parquet_path in tqdm(sorted(list(parquet_paths)), desc="Collecting delta action stats"):
        data = pd.read_parquet(parquet_path)
        trajectory_length = len(data)
        allowed_base_indices = _get_allowed_step_indices_for_parquet(
            parquet_path,
            allowed_step_indices_by_parquet,
        )
        if allowed_base_indices is None:
            base_indices_iter = range(trajectory_length)
        else:
            base_indices_iter = np.asarray(allowed_base_indices, dtype=np.int64)
        for action_col, slice_list in action_col_slices.items():
            if action_col not in data.columns:
                raise ValueError(f"{action_col} not found in parquet columns.")
            action_matrix = np.stack(data[action_col])
            if action_matrix.ndim == 1:
                action_matrix = action_matrix.reshape(-1, 1)
            action_padding_ref = slice_list[0][4]
            prepared_slices = []
            for action_key, a_slice, state_col, s_slice, action_padding, state_padding in slice_list:
                if state_col not in data.columns:
                    raise ValueError(f"{state_col} not found in parquet columns.")
                state_matrix = np.stack(data[state_col])
                if state_matrix.ndim == 1:
                    state_matrix = state_matrix.reshape(-1, 1)
                state_part_full = state_matrix[:, s_slice[0] : s_slice[1]]
                prepared_slices.append((action_key, a_slice, state_part_full, state_padding))
            for base_index in base_indices_iter:
                action_steps = np.array(action_indices) + base_index
                action_chunk_full = _get_chunk(action_matrix, action_steps, action_padding_ref)

                for action_key, a_slice, state_part_full, state_padding in prepared_slices:
                    action_part_chunk = action_chunk_full[:, a_slice[0] : a_slice[1]]
                    state_chunk = _get_chunk(state_part_full, np.array(state_indices) + base_index, state_padding)
                    if action_part_chunk.shape[1] != state_chunk.shape[1]:
                        raise ValueError(f"Action/state dim mismatch for {action_col}:{a_slice}")

                    out = action_part_chunk.copy()
                    if len(out) > 1:
                        out[1:] = action_part_chunk[1:] - action_part_chunk[:-1]
                    out[0] = action_part_chunk[0] - state_chunk[0]
                    if _is_rotation_key(action_key):
                        out = _wrap_rotation_delta(out)
                    action_chunk_full[:, a_slice[0] : a_slice[1]] = out

                accum[action_col].append(action_chunk_full)

    delta_stats = copy.deepcopy(base_stats)
    for action_col, series_list in accum.items():
        if not series_list:
            continue
        all_values = np.concatenate(series_list, axis=0).astype(np.float32)
        delta_stats[action_col] = {
            "mean": np.mean(all_values, axis=0).tolist(),
            "std": np.std(all_values, axis=0).tolist(),
            "min": np.min(all_values, axis=0).tolist(),
            "max": np.max(all_values, axis=0).tolist(),
            "q01": np.quantile(all_values, 0.01, axis=0).tolist(),
            "q99": np.quantile(all_values, 0.99, axis=0).tolist(),
        }
    return delta_stats


def calculate_rel_action_statistics(
    parquet_paths: list[Path],
    lerobot_modality_meta: "LeRobotModalityMetadata",
    action_keys_full: list[str],
    state_keys_full: list[str],
    action_indices: list[int],
    state_indices: list[int],
    action_mode_apply_keys: list[str] | None = None,
    action_mode_state_map: dict[str, str] | None = None,
    base_stats: dict | None = None,
    allowed_step_indices_by_parquet: dict[str, np.ndarray] | None = None,
) -> dict:
    """
    Calculate action statistics using rel mode.

    Rule:
      - For all t: a_t - s_0

    Mapping rule (only two cases):
      1) Use explicit action_mode_state_map if provided.
      2) Otherwise, replace 'action.' with 'state.' directly.
    """
    if base_stats is None:
        base_stats = calculate_dataset_statistics(
            parquet_paths,
            allowed_step_indices_by_parquet=allowed_step_indices_by_parquet,
        )

    action_col_slices = _get_action_col_slices(
        lerobot_modality_meta, action_keys_full, state_keys_full, action_mode_apply_keys, action_mode_state_map
    )
    if not action_col_slices:
        raise ValueError("No action columns found in the dataset.")

    def _get_chunk(array: np.ndarray, step_indices: np.ndarray, padding_strategy: str) -> np.ndarray:
        max_length = array.shape[0]
        front_padding = step_indices < 0
        end_padding = step_indices >= max_length
        padding_positions = np.logical_or(front_padding, end_padding)
        output = np.zeros((len(step_indices), array.shape[1]), dtype=array.dtype)
        if (~padding_positions).any():
            output[~padding_positions] = array[step_indices[~padding_positions]]
        if padding_positions.any():
            if padding_strategy == "first_last":
                output[front_padding] = array[0]
                output[end_padding] = array[-1]
            elif padding_strategy == "zero":
                output[padding_positions] = 0
            else:
                raise ValueError(f"Invalid padding strategy: {padding_strategy}")
        return output

    accum: dict[str, list[np.ndarray]] = {col: [] for col in action_col_slices.keys()}
    for parquet_path in tqdm(sorted(list(parquet_paths)), desc="Collecting rel action stats"):
        data = pd.read_parquet(parquet_path)
        trajectory_length = len(data)
        allowed_base_indices = _get_allowed_step_indices_for_parquet(
            parquet_path,
            allowed_step_indices_by_parquet,
        )
        if allowed_base_indices is None:
            base_indices_iter = range(trajectory_length)
        else:
            base_indices_iter = np.asarray(allowed_base_indices, dtype=np.int64)
        for action_col, slice_list in action_col_slices.items():
            if action_col not in data.columns:
                raise ValueError(f"{action_col} not found in parquet columns.")
            action_matrix = np.stack(data[action_col])
            if action_matrix.ndim == 1:
                action_matrix = action_matrix.reshape(-1, 1)
            action_padding_ref = slice_list[0][4]
            prepared_slices = []
            for action_key, a_slice, state_col, s_slice, action_padding, state_padding in slice_list:
                if state_col not in data.columns:
                    raise ValueError(f"{state_col} not found in parquet columns.")
                state_matrix = np.stack(data[state_col])
                if state_matrix.ndim == 1:
                    state_matrix = state_matrix.reshape(-1, 1)
                state_part_full = state_matrix[:, s_slice[0] : s_slice[1]]
                prepared_slices.append((action_key, a_slice, state_part_full, state_padding))
            for base_index in base_indices_iter:
                action_steps = np.array(action_indices) + base_index
                action_chunk_full = _get_chunk(action_matrix, action_steps, action_padding_ref)

                for action_key, a_slice, state_part_full, state_padding in prepared_slices:
                    action_part_chunk = action_chunk_full[:, a_slice[0] : a_slice[1]]
                    state_chunk = _get_chunk(state_part_full, np.array(state_indices) + base_index, state_padding)
                    if action_part_chunk.shape[1] != state_chunk.shape[1]:
                        raise ValueError(f"Action/state dim mismatch for {action_col}:{a_slice}")

                    out = action_part_chunk - state_chunk[0]
                    action_chunk_full[:, a_slice[0] : a_slice[1]] = out

                accum[action_col].append(action_chunk_full)

    rel_stats = copy.deepcopy(base_stats)
    for action_col, series_list in accum.items():
        if not series_list:
            continue
        all_values = np.concatenate(series_list, axis=0).astype(np.float32)
        rel_stats[action_col] = {
            "mean": np.mean(all_values, axis=0).tolist(),
            "std": np.std(all_values, axis=0).tolist(),
            "min": np.min(all_values, axis=0).tolist(),
            "max": np.max(all_values, axis=0).tolist(),
            "q01": np.quantile(all_values, 0.01, axis=0).tolist(),
            "q99": np.quantile(all_values, 0.99, axis=0).tolist(),
        }
    return rel_stats

class ModalityConfig(BaseModel):
    """Configuration for a modality."""

    delta_indices: list[int]
    """Delta indices to sample relative to the current index. The returned data will correspond to the original data at a sampled base index + delta indices."""
    modality_keys: list[str]
    """The keys to load for the modality in the dataset."""


class LeRobotSingleDataset(Dataset):
    """
    Base dataset class for LeRobot that supports sharding.
    """
    def __init__(
        self,
        dataset_path: Path | str,
        modality_configs: dict[str, ModalityConfig],
        embodiment_tag: str | EmbodimentTag,
        video_backend: str = "decord",
        video_backend_kwargs: dict | None = None,
        transforms: ComposedModalityTransform | None = None,
        delete_pause_frame: bool = False,
        data_cfg = None,
        **kwargs,
    ):
        """
        Initialize the dataset.

        Args:
            dataset_path (Path | str): The path to the dataset.
            modality_configs (dict[str, ModalityConfig]): The configuration for each modality. The keys are the modality names, and the values are the modality configurations.
                See `ModalityConfig` for more details.
            video_backend (str): Backend for video reading.
            video_backend_kwargs (dict): Keyword arguments for the video backend when initializing the video reader.
            transforms (ComposedModalityTransform): The transforms to apply to the dataset.
            embodiment_tag (EmbodimentTag): Overload the embodiment tag for the dataset. e.g. define it as "new_embodiment"
        """
        # first check if the path directory exists
        self.data_cfg = data_cfg
        if not Path(dataset_path).exists():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")
        # indict letobot version
        self._lerobot_version =  self.data_cfg.get("lerobot_version", "v2.0") #self._indict_lerobot_version(**kwargs)

        self._action_mode = None
        self._action_mode_state_map = {}
        self._action_mode_apply_keys = None
        self._action_mode_reference = "state"
        self._outlier_filter_required = None

        self.delete_pause_frame = delete_pause_frame

        self.modality_configs = modality_configs
        self.video_backend = video_backend
        self.video_backend_kwargs = video_backend_kwargs if video_backend_kwargs is not None else {}
        self.transforms = (
            transforms if transforms is not None else ComposedModalityTransform(transforms=[])
        )

        self._dataset_path = Path(dataset_path)
        self._dataset_name = self._dataset_path.name
        if isinstance(embodiment_tag, EmbodimentTag):
            self.tag = embodiment_tag.value
        else:
            self.tag = embodiment_tag
        self.robot_type = kwargs.get("robot_type", self.tag)
        self.normalization_group = kwargs.get("normalization_group", self.tag)

        self._init_action_mode()
        self._trajectory_ids, self._trajectory_lengths = self._get_trajectories()
        self.curr_traj_data = None
        self.curr_traj_id = None

        # Initialize raw dataset metadata needed by trajectory filtering before
        # building derived statistics metadata.
        self._lerobot_modality_meta = self._get_lerobot_modality_meta()
        self._lerobot_info_meta = self._get_lerobot_info_meta()
        self._data_path_pattern = self._get_data_path_pattern()
        self._video_path_pattern = self._get_video_path_pattern()
        self._chunk_size = self._get_chunk_size()
        # LeRobot-specific config
        self._tasks = self._get_tasks()
        # self._episodes = self._get_episode_info() # TODO why we need this func

        self._modality_keys = self._get_modality_keys()
        self._delta_indices = self._get_delta_indices()
        self._all_steps = self._get_all_steps()
        self._original_trajectory_count = int(len(self._trajectory_ids))
        self._original_step_count = int(np.sum(self._trajectory_lengths))
        self._filtered_trajectory_count = self._count_filtered_trajectories()
        self._filtered_step_count = int(len(self._all_steps))
        self._metadata = self._get_metadata(EmbodimentTag(self.tag))
        self._metadata = self._apply_manual_action_normalization_statistics(self._metadata)
        self.set_transforms_metadata(self.metadata)
        self.set_epoch(0)

        print(f"Initialized dataset {self.dataset_name} with {embodiment_tag}")
        if self._trajectory_filtering_enabled():
            print(
                f"Dataset {self.dataset_name} kept "
                f"{self._filtered_trajectory_count}/{self._original_trajectory_count} trajectories "
                f"and {self._filtered_step_count}/{self._original_step_count} steps after filtering"
            )


        # Check if the dataset is valid
        self._check_integrity()

    @property
    def dataset_path(self) -> Path:
        """The path to the dataset that contains the METADATA_FILENAME file."""
        return self._dataset_path

    @property
    def metadata(self) -> DatasetMetadata:
        """The metadata for the dataset, loaded from metadata.json in the dataset directory"""
        return self._metadata

    @property
    def trajectory_ids(self) -> np.ndarray:
        """The trajectory IDs in the dataset, stored as a 1D numpy array of strings."""
        return self._trajectory_ids

    @property
    def trajectory_lengths(self) -> np.ndarray:
        """The trajectory lengths in the dataset, stored as a 1D numpy array of integers.
        The order of the lengths is the same as the order of the trajectory IDs.
        """
        return self._trajectory_lengths

    @property
    def all_steps(self) -> list[tuple[int, int]]:
        """The trajectory IDs and base indices for all steps in the dataset.
        Example:
            self.trajectory_ids: [0, 1, 2]
            self.trajectory_lengths: [3, 2, 4]
            return: [
                ("traj_0", 0), ("traj_0", 1), ("traj_0", 2),
                ("traj_1", 0), ("traj_1", 1),
                ("traj_2", 0), ("traj_2", 1), ("traj_2", 2), ("traj_2", 3)
            ]
        """
        return self._all_steps

    @property
    def original_trajectory_count(self) -> int:
        return self._original_trajectory_count

    @property
    def filtered_trajectory_count(self) -> int:
        return self._filtered_trajectory_count

    @property
    def original_step_count(self) -> int:
        return self._original_step_count

    @property
    def filtered_step_count(self) -> int:
        return self._filtered_step_count

    @property
    def modality_keys(self) -> dict:
        """The modality keys for the dataset. The keys are the modality names, and the values are the keys for each modality.

        Example: {
            "video": ["video.image_side_0", "video.image_side_1"],
            "state": ["state.eef_position", "state.eef_rotation"],
            "action": ["action.eef_position", "action.eef_rotation"],
            "language": ["language.human.task"],
            "timestamp": ["timestamp"],
            "reward": ["reward"],
        }
        """
        return self._modality_keys

    @property
    def delta_indices(self) -> dict[str, np.ndarray]:
        """The delta indices for the dataset. The keys are the modality.key, and the values are the delta indices for each modality.key."""
        return self._delta_indices

    @property
    def dataset_name(self) -> str:
        """The name of the dataset."""
        return self._dataset_name

    @property
    def lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        """The metadata for the LeRobot dataset."""
        return self._lerobot_modality_meta

    @property
    def lerobot_info_meta(self) -> dict:
        """The metadata for the LeRobot dataset."""
        return self._lerobot_info_meta

    @property
    def data_path_pattern(self) -> str:
        """The path pattern for the LeRobot dataset."""
        return self._data_path_pattern

    @property
    def video_path_pattern(self) -> str:
        """The path pattern for the LeRobot dataset."""
        return self._video_path_pattern

    @property
    def chunk_size(self) -> int:
        """The chunk size for the LeRobot dataset."""
        return self._chunk_size

    @property
    def tasks(self) -> pd.DataFrame:
        """The tasks for the dataset."""
        return self._tasks

    def _get_metadata(self, embodiment_tag: EmbodimentTag) -> DatasetMetadata:
        """Get the metadata for the dataset.

        Returns:
            dict: The metadata for the dataset.
        """

        # 1. Modality metadata
        modality_meta_path = self.dataset_path / LE_ROBOT_MODALITY_FILENAME
        assert (
            modality_meta_path.exists()
        ), f"Please provide a {LE_ROBOT_MODALITY_FILENAME} file in {self.dataset_path}"
        # 1.1. State and action modalities
        simplified_modality_meta: dict[str, dict] = {}
        with open(modality_meta_path, "r") as f:
            le_modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))
        le_modality_meta = self._augment_lerobot_modality_meta(le_modality_meta)
        for modality in ["state", "action"]:
            simplified_modality_meta[modality] = {}
            le_state_action_meta: dict[str, LeRobotStateActionMetadata] = getattr(
                le_modality_meta, modality
            )
            for subkey in le_state_action_meta:
                state_action_dtype = np.dtype(le_state_action_meta[subkey].dtype)
                if np.issubdtype(state_action_dtype, np.floating):
                    continuous = True
                else:
                    continuous = False
                simplified_modality_meta[modality][subkey] = {
                    "absolute": le_state_action_meta[subkey].absolute,
                    "rotation_type": le_state_action_meta[subkey].rotation_type,
                    "shape": [
                        le_state_action_meta[subkey].end - le_state_action_meta[subkey].start
                    ],
                    "continuous": continuous,
                }

        # 1.2. Video modalities
        le_info_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        assert (
            le_info_path.exists()
        ), f"Please provide a {LE_ROBOT_INFO_FILENAME} file in {self.dataset_path}"
        with open(le_info_path, "r") as f:
            le_info = json.load(f)
        simplified_modality_meta["video"] = {}
        for new_key in le_modality_meta.video:
            original_key = le_modality_meta.video[new_key].original_key
            if original_key is None:
                original_key = new_key
            le_video_meta = le_info["features"][original_key]
            height = le_video_meta["shape"][le_video_meta["names"].index("height")]
            width = le_video_meta["shape"][le_video_meta["names"].index("width")]
            channels, fps = self._resolve_video_feature_channels_and_fps(le_video_meta)
            simplified_modality_meta["video"][new_key] = {
                "resolution": [width, height],
                "channels": channels,
                "fps": fps,
            }


        # 2. Dataset statistics
        def is_main():
            return (not dist.is_initialized()) or dist.get_rank() == 0
        
        action_mode = _normalize_action_mode(self.data_cfg.get("action_mode", "abs") if self.data_cfg else "abs")
        le_statistics_by_mode = None

        def _normalize_cached_statistics(loaded_stats: object) -> dict | None:
            """Normalize cached statistics into a mode->stats mapping."""
            if not isinstance(loaded_stats, dict):
                return None

            mode_keys = {"abs", "delta", "rel"}
            payload_stats = loaded_stats.get("statistics")
            if isinstance(payload_stats, dict):
                if any(key in payload_stats for key in mode_keys):
                    normalized = {key: value for key, value in payload_stats.items() if key in mode_keys}
                    return normalized if normalized else None

                payload_mode = action_mode
                cache_config = loaded_stats.get("__cache_config")
                if isinstance(cache_config, dict) and "mode" in cache_config:
                    payload_mode = _normalize_action_mode(cache_config["mode"])

                cleaned_payload = {
                    key: value for key, value in payload_stats.items() if not str(key).startswith("__")
                }
                return {payload_mode: cleaned_payload} if cleaned_payload else None

            if any(key in loaded_stats for key in mode_keys):
                normalized = {key: value for key, value in loaded_stats.items() if key in mode_keys}
                return normalized if normalized else None

            cleaned = {
                key: value for key, value in loaded_stats.items() if not str(key).startswith("__")
            }
            return {"abs": cleaned} if cleaned else None

        stats_path = self._get_stats_cache_path()
        tmp_path = stats_path.with_suffix(".tmp")
        
        # ---------- all rank try to read  ----------
        if stats_path.exists():
            try:
                with open(stats_path, "r") as f:
                    le_statistics = json.load(f)
                le_statistics_by_mode = _normalize_cached_statistics(le_statistics)
                if le_statistics_by_mode is None:
                    raise ValueError(
                        f"Invalid statistics cache format at {stats_path}: "
                        f"expected dict or mode mapping, got {type(le_statistics).__name__}"
                    )
            except Exception as e:
                print(
                    f"[RANK {os.environ.get('RANK', 'NA')}] "
                    f"Failed to load dataset statistics ({e}), rebuilding..."
                )
                le_statistics_by_mode = None

        # ---------- rank0 build ----------
        if le_statistics_by_mode is None:
            le_statistics_by_mode = {}

        computed_any = False
        cache_metadata = {}
        if self._trajectory_filtering_enabled():
            cache_metadata.update(
                {
                    "__original_trajectory_count": self.original_trajectory_count,
                    "__filtered_trajectory_count": self.filtered_trajectory_count,
                    "__original_step_count": self.original_step_count,
                    "__filtered_step_count": self.filtered_step_count,
                }
            )

        def _write_statistics_cache(reason: str) -> None:
            try:
                stats_path.parent.mkdir(parents=True, exist_ok=True)
                with open(tmp_path, "w") as f:
                    cache_payload = dict(le_statistics_by_mode)
                    cache_payload.update(cache_metadata)
                    json.dump(cache_payload, f, indent=4)
                os.replace(tmp_path, stats_path)
                print(f"[RANK 0] Saved dataset statistics cache {reason} to {stats_path}")
            except Exception as e:
                print(
                    f"[RANK 0] Failed to write dataset statistics cache ({e}), "
                    "continuing with in-memory statistics."
                )
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass

        if is_main():
            action_keys_full = []
            state_keys_full = []
            if "action" in self.modality_configs:
                action_keys_full = list(self.modality_configs["action"].modality_keys)
            if "state" in self.modality_configs:
                state_keys_full = list(self.modality_configs["state"].modality_keys)
            if "action" in self.modality_configs:
                action_indices = list(self.modality_configs["action"].delta_indices)
            else:
                action_indices = None
            if "state" in self.modality_configs:
                state_indices = list(self.modality_configs["state"].delta_indices)
            else:
                state_indices = None
            if action_indices is None:
                raise ValueError("Action modality is required to compute action mode statistics.")

            apply_keys = None
            if self.data_cfg:
                apply_keys = self.data_cfg.get("action_mode_apply_keys", None)
            if apply_keys:
                normalized = []
                for key in apply_keys:
                    key = str(key)
                    if not key.startswith("action."):
                        key = f"action.{key}"
                    normalized.append(key)
                apply_keys = normalized
            else:
                apply_keys = action_keys_full

            state_map_cfg = self.data_cfg.get("action_mode_state_map", {}) if self.data_cfg else {}
            normalized_state_map = {}
            for action_key, state_key in (state_map_cfg or {}).items():
                action_key = str(action_key)
                state_key = str(state_key)
                if not action_key.startswith("action."):
                    action_key = f"action.{action_key}"
                if not state_key.startswith("state."):
                    state_key = f"state.{state_key}"
                normalized_state_map[action_key] = state_key
            parquet_files_filtered = self._get_statistics_parquet_files()
            allowed_step_indices_by_parquet = self._get_allowed_step_indices_by_parquet()
            if allowed_step_indices_by_parquet is not None:
                cache_metadata["__filter_delta_eef_steps"] = True
                cache_metadata["__stats_scope"] = "filtered_steps"
            if not parquet_files_filtered:
                if self._trajectory_filtering_enabled() and self.filtered_trajectory_count == 0:
                    print(
                        f"[RANK {os.environ.get('RANK', 'NA')}] "
                        f"Dataset {self.dataset_name} has 0/{self.original_trajectory_count} "
                        "trajectories after filtering; falling back to raw stats for metadata "
                        "and the dataset will be skipped by the mixture loader."
                    )
                    parquet_files_filtered = self._list_all_statistics_parquet_files()
                    cache_metadata = {
                        "__empty_after_filtering": True,
                        "__original_trajectory_count": self.original_trajectory_count,
                        "__filtered_trajectory_count": self.filtered_trajectory_count,
                        "__original_step_count": self.original_step_count,
                        "__filtered_step_count": self.filtered_step_count,
                        "__stats_fallback": "unfiltered_dataset_statistics",
                    }
                if not parquet_files_filtered:
                    raise ValueError(f"No parquet files available for statistics in {self.dataset_name}")
            action_mode_reference = "state"
            if self.data_cfg:
                action_mode_reference = str(self.data_cfg.get("action_mode_reference", "state")).lower()
            if action_mode_reference in {"self", "action_only"}:
                action_mode_reference = "action"
            if action_mode_reference == "state" and state_indices is None:
                raise ValueError("State modality is required to compute state-referenced action mode statistics.")
        
            if "abs" not in le_statistics_by_mode:
                print(f"[RANK 0] Calculating dataset statistics for {self.dataset_name}")

                le_statistics_by_mode["abs"] = calculate_dataset_statistics(
                    parquet_files_filtered,
                    allowed_step_indices_by_parquet=allowed_step_indices_by_parquet,
                )
                computed_any = True

            modes_to_compute = [] if action_mode == "abs" else [action_mode]
            for mode in modes_to_compute:
                mode_computed = False
                if mode not in le_statistics_by_mode:
                    if mode == "delta":
                        if action_mode_reference == "action":
                            le_statistics_by_mode[mode] = calculate_action_only_delta_statistics(
                                parquet_paths=parquet_files_filtered,
                                lerobot_modality_meta=le_modality_meta,
                                action_keys_full=action_keys_full,
                                action_indices=action_indices,
                                action_mode_apply_keys=apply_keys,
                                base_stats=le_statistics_by_mode["abs"],
                                allowed_step_indices_by_parquet=allowed_step_indices_by_parquet,
                            )
                        else:
                            le_statistics_by_mode[mode] = calculate_delta_action_statistics(
                                parquet_paths=parquet_files_filtered,
                                lerobot_modality_meta=le_modality_meta,
                                action_keys_full=action_keys_full,
                                state_keys_full=state_keys_full,
                                action_indices=action_indices,
                                state_indices=state_indices,
                                action_mode_apply_keys=apply_keys,
                                action_mode_state_map=normalized_state_map,
                                base_stats=le_statistics_by_mode["abs"],
                                allowed_step_indices_by_parquet=allowed_step_indices_by_parquet,
                            )
                    else:
                        if action_mode_reference == "action":
                            le_statistics_by_mode[mode] = calculate_action_only_rel_statistics(
                                parquet_paths=parquet_files_filtered,
                                lerobot_modality_meta=le_modality_meta,
                                action_keys_full=action_keys_full,
                                action_indices=action_indices,
                                action_mode_apply_keys=apply_keys,
                                base_stats=le_statistics_by_mode["abs"],
                                allowed_step_indices_by_parquet=allowed_step_indices_by_parquet,
                            )
                        else:
                            le_statistics_by_mode[mode] = calculate_rel_action_statistics(
                                parquet_paths=parquet_files_filtered,
                                lerobot_modality_meta=le_modality_meta,
                                action_keys_full=action_keys_full,
                                state_keys_full=state_keys_full,
                                action_indices=action_indices,
                                state_indices=state_indices,
                                action_mode_apply_keys=apply_keys,
                                action_mode_state_map=normalized_state_map,
                                base_stats=le_statistics_by_mode["abs"],
                                allowed_step_indices_by_parquet=allowed_step_indices_by_parquet,
                            )
                    mode_computed = True
                if mode_computed:
                    computed_any = True

            if is_main() and computed_any:
                _write_statistics_cache("after statistics computation")

        # ---------- sync statistics from rank0 ----------
        if dist.is_initialized():
            obj_list = [le_statistics_by_mode if is_main() else None]
            dist.broadcast_object_list(obj_list, src=0)
            le_statistics_by_mode = obj_list[0]
        elif le_statistics_by_mode is None:
            with open(stats_path, "r") as f:
                le_statistics_by_mode = _normalize_cached_statistics(json.load(f))

        if not isinstance(le_statistics_by_mode, dict):
            raise ValueError(
                f"Invalid statistics cache for {self.dataset_path}: "
                f"expected mode mapping dict, got {type(le_statistics_by_mode).__name__}"
            )

        # Validate selected mode stats
        selected_mode = action_mode if action_mode in le_statistics_by_mode else "abs"
        le_statistics = le_statistics_by_mode[selected_mode]
        for stat in le_statistics.values():
            DatasetStatisticalValues.model_validate(stat)


        dataset_statistics = {}
        for our_modality in ["state", "action"]:
            dataset_statistics[our_modality] = {}
            for subkey in simplified_modality_meta[our_modality]:
                dataset_statistics[our_modality][subkey] = {}
                state_action_meta = le_modality_meta.get_key_meta(f"{our_modality}.{subkey}")
                assert isinstance(state_action_meta, LeRobotStateActionMetadata)
                le_modality = state_action_meta.original_key
                for stat_name in le_statistics[le_modality]:
                    indices = np.arange(
                        state_action_meta.start,
                        state_action_meta.end,
                    )
                    stat = np.array(le_statistics[le_modality][stat_name])
                    dataset_statistics[our_modality][subkey][stat_name] = stat[indices].tolist()

        # 3. Full dataset metadata
        metadata = DatasetMetadata(
            statistics=dataset_statistics,  # type: ignore
            modalities=simplified_modality_meta,  # type: ignore
            embodiment_tag=embodiment_tag,
        )

        return metadata

    def _get_trajectories(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the trajectories in the dataset."""
        # Get trajectory lengths, IDs, and whitelist from dataset metadata
        # v2.0
        if self._lerobot_version == "v2.0":
            file_path = self.dataset_path / LE_ROBOT_EPISODE_FILENAME
            with open(file_path, "r") as f:
                episode_metadata = [json.loads(line) for line in f]
            trajectory_ids = []
            trajectory_lengths = []
            for episode in episode_metadata:
                trajectory_ids.append(episode["episode_index"])
                trajectory_lengths.append(episode["length"])
            return np.array(trajectory_ids), np.array(trajectory_lengths)
        # v3.0
        elif self._lerobot_version == "v3.0":
            file_paths = sorted(list((self.dataset_path).glob(LE_ROBOT3_EPISODE_FILENAME)))
            trajectory_ids = []
            trajectory_lengths = []
            # data_chunck_index = []
            # data_file_index = []
            # vido_from_index = []
            self.trajectory_ids_to_metadata = {}
            for file_path in file_paths:
                episodes_data = pd.read_parquet(file_path)
                timestamp_cols = [
                    c
                    for c in episodes_data.columns
                    if str(c).startswith("videos/") and str(c).endswith("/from_timestamp")
                ]
                for index, episode in episodes_data.iterrows():
                    trajectory_ids.append(episode["episode_index"])
                    trajectory_lengths.append(episode["length"])

                    from_timestamps = {}
                    for col in timestamp_cols:
                        value = episode[col]
                        if pd.isna(value):
                            continue
                        # videos/{video_key}/from_timestamp -> {video_key}
                        video_key = str(col)[len("videos/") : -len("/from_timestamp")]
                        from_timestamps[video_key] = float(value)

                    # TODO auto map key? just map to file_path and file_from_index
                    episode_meta = {
                        "data/chunk_index": episode["data/chunk_index"],
                        "data/file_index": episode["data/file_index"],
                        "data/file_from_index": index,
                        "videos/from_timestamps": from_timestamps,
                    }
                    self.trajectory_ids_to_metadata[trajectory_ids[-1]] = episode_meta

            # 这里应该可以直接读取到 save index 信息
            return np.array(trajectory_ids), np.array(trajectory_lengths)

    def _filter_outlier_trajectory_configured(self) -> bool:
        return self._filter_joint_outlier_trajectory_configured() or self._filter_gripper_outlier_trajectory_configured()

    def _filter_joint_outlier_trajectory_configured(self) -> bool:
        return bool(self.data_cfg.get("filter_outlier_trajectory", False)) if self.data_cfg else False

    def _filter_gripper_outlier_trajectory_configured(self) -> bool:
        return bool(self.data_cfg.get("filter_gripper_outlier_trajectory", False)) if self.data_cfg else False

    def _filter_delta_eef_trajectory_configured(self) -> bool:
        return bool(self.data_cfg.get("filter_delta_eef_trajectory", False)) if self.data_cfg else False

    def _filter_delta_eef_steps_configured(self) -> bool:
        return bool(self.data_cfg.get("filter_delta_eef_steps", False)) if self.data_cfg else False

    def _filter_outlier_trajectory_enabled(self) -> bool:
        return self._filter_outlier_trajectory_configured() and self._dataset_needs_outlier_filtering()

    def _filter_delta_eef_trajectory_enabled(self) -> bool:
        return self._filter_delta_eef_trajectory_configured() and not self._filter_delta_eef_steps_enabled()

    def _filter_delta_eef_steps_enabled(self) -> bool:
        return self._filter_delta_eef_steps_configured()

    def _filter_invalid_droid_task_enabled(self) -> bool:
        if not self.data_cfg:
            return False
        return bool(self.data_cfg.get("filter_invalid_droid_task", False))

    def _trajectory_filtering_enabled(self) -> bool:
        return (
            self._filter_outlier_trajectory_enabled()
            or self._filter_delta_eef_trajectory_enabled()
            or self._filter_delta_eef_steps_enabled()
            or self._filter_invalid_droid_task_enabled()
        )

    def _get_outlier_abs_limit(self) -> float:
        return float(self.data_cfg.get("outlier_abs_limit", np.pi)) if self.data_cfg else float(np.pi)

    def _get_gripper_outlier_abs_limit(self) -> float:
        if not self.data_cfg:
            return self._get_outlier_abs_limit()
        return float(self.data_cfg.get("gripper_outlier_abs_limit", self._get_outlier_abs_limit()))

    def _get_delta_eef_position_abs_limit(self) -> float:
        if not self.data_cfg:
            return 0.02
        return float(self.data_cfg.get("delta_eef_position_abs_limit", 0.02))

    def _get_delta_eef_rotation_abs_limit(self) -> float:
        if not self.data_cfg:
            return 0.05
        return float(self.data_cfg.get("delta_eef_rotation_abs_limit", 0.05))

    def _get_delta_eef_valid_ratio(self) -> float:
        if not self.data_cfg:
            return 0.5
        return float(self.data_cfg.get("delta_eef_valid_ratio", 0.5))

    def _get_action_target_mode(self) -> str:
        if not self.data_cfg:
            return "legacy"
        mode = str(self.data_cfg.get("action_target_mode", "legacy")).lower()
        if mode not in {"legacy", "delta_eef_velocity"}:
            raise ValueError(
                f"Invalid action_target_mode: {mode}. Expected one of: legacy, delta_eef_velocity."
            )
        return mode

    def _uses_direct_delta_eef_filter_target(self) -> bool:
        if self._get_action_target_mode() == "delta_eef_velocity":
            return True
        action_key_set = set(self.modality_keys.get("action", []))
        return (
            "action.robot_0_delta_eef_position" in action_key_set
            and "action.robot_0_delta_eef_rotation" in action_key_set
        )

    def _get_manual_action_normalization_statistics(self) -> dict[str, dict]:
        if not self.data_cfg:
            return {}
        raw_stats = self.data_cfg.get("manual_action_normalization_statistics", {}) or {}
        if not isinstance(raw_stats, dict):
            raise ValueError(
                "manual_action_normalization_statistics must be a dict mapping action keys to stats."
            )

        normalized_stats = {}
        for action_key, stats in raw_stats.items():
            action_key = str(action_key)
            if not action_key.startswith("action."):
                action_key = f"action.{action_key}"
            if not isinstance(stats, dict):
                raise ValueError(
                    f"manual stats for {action_key} must be a dict, got {type(stats).__name__}."
                )
            normalized_stats[action_key] = stats
        return self._apply_delta_eef_limit_overrides_to_manual_stats(normalized_stats)

    def _build_symmetric_manual_stats_from_limit(self, limit: float, dim: int) -> dict[str, list[float]]:
        dim = int(dim)
        if dim <= 0:
            raise ValueError(f"Manual action normalization override requires positive dim, got {dim}.")
        abs_limit = float(limit)
        return {
            "min": [-abs_limit] * dim,
            "max": [abs_limit] * dim,
        }

    def _override_manual_stat_limit(
        self,
        manual_stats: dict[str, dict],
        action_key: str,
        limit: float,
    ) -> None:
        if action_key not in manual_stats:
            return
        existing = manual_stats[action_key]
        min_values = existing.get("min", [])
        max_values = existing.get("max", [])
        dim = len(max_values) or len(min_values)
        if dim == 0:
            raise ValueError(
                f"Manual action normalization override for `{action_key}` requires existing min/max values."
            )
        manual_stats[action_key] = self._build_symmetric_manual_stats_from_limit(limit, dim)

    def _apply_delta_eef_limit_overrides_to_manual_stats(
        self,
        manual_stats: dict[str, dict],
    ) -> dict[str, dict]:
        if not manual_stats or not self.data_cfg:
            return manual_stats

        position_limit = self.data_cfg.get("delta_eef_position_abs_limit", None)
        rotation_limit = self.data_cfg.get("delta_eef_rotation_abs_limit", None)

        if position_limit is not None:
            self._override_manual_stat_limit(
                manual_stats,
                "action.eef_position",
                float(position_limit),
            )
            self._override_manual_stat_limit(
                manual_stats,
                "action.delta_eef_position",
                float(position_limit),
            )
            self._override_manual_stat_limit(
                manual_stats,
                "action.robot_0_delta_eef_position",
                float(position_limit),
            )
            self._override_manual_stat_limit(
                manual_stats,
                "action.robot_1_delta_eef_position",
                float(position_limit),
            )
        if rotation_limit is not None:
            self._override_manual_stat_limit(
                manual_stats,
                "action.eef_rotation",
                float(rotation_limit),
            )
            self._override_manual_stat_limit(
                manual_stats,
                "action.delta_eef_rotation",
                float(rotation_limit),
            )
            self._override_manual_stat_limit(
                manual_stats,
                "action.robot_0_delta_eef_rotation",
                float(rotation_limit),
            )
            self._override_manual_stat_limit(
                manual_stats,
                "action.robot_1_delta_eef_rotation",
                float(rotation_limit),
            )
        return manual_stats

    def _build_full_manual_statistical_values(
        self,
        stats: dict,
    ) -> DatasetStatisticalValues:
        min_values = np.asarray(stats["min"], dtype=np.float32)
        max_values = np.asarray(stats["max"], dtype=np.float32)
        if min_values.shape != max_values.shape:
            raise ValueError(
                f"Manual action normalization requires matching min/max shapes, got {min_values.shape} vs {max_values.shape}."
            )
        mean_values = (min_values + max_values) / 2.0
        std_values = np.maximum((max_values - min_values) / 2.0, 1e-6)
        return DatasetStatisticalValues.model_validate(
            {
                "min": min_values.tolist(),
                "max": max_values.tolist(),
                "mean": mean_values.tolist(),
                "std": std_values.tolist(),
                "q01": min_values.tolist(),
                "q99": max_values.tolist(),
            }
        )

    def _apply_manual_action_normalization_statistics(
        self,
        metadata: DatasetMetadata,
    ) -> DatasetMetadata:
        manual_stats = self._get_manual_action_normalization_statistics()
        if not manual_stats:
            return metadata

        for action_key, stats in manual_stats.items():
            subkey = action_key.replace("action.", "", 1)
            if subkey not in metadata.statistics.action:
                raise KeyError(
                    f"Manual action normalization key `{action_key}` not found in dataset action statistics."
                )
            metadata.statistics.action[subkey] = self._build_full_manual_statistical_values(stats)
        return metadata

    def _get_trajectory_filter_config_key(self) -> str:
        return build_dataset_cache_key(
            dataset_name=self.dataset_name,
            filter_outlier_trajectory=self._filter_outlier_trajectory_enabled(),
            outlier_abs_limit=self._get_outlier_abs_limit(),
            filter_gripper_outlier_trajectory=self._filter_gripper_outlier_trajectory_configured(),
            gripper_outlier_abs_limit=self._get_gripper_outlier_abs_limit(),
            filter_delta_eef_trajectory=self._filter_delta_eef_trajectory_enabled(),
            filter_delta_eef_steps=self._filter_delta_eef_steps_enabled(),
            delta_eef_position_abs_limit=self._get_delta_eef_position_abs_limit(),
            delta_eef_rotation_abs_limit=self._get_delta_eef_rotation_abs_limit(),
            delta_eef_valid_ratio=self._get_delta_eef_valid_ratio(),
            delete_pause_frame=self.delete_pause_frame,
            embodiment_tag=self.tag,
            robot_type=self.robot_type,
            data_cfg=self.data_cfg,
        )

    def _get_absolute_joint_action_slices(self) -> dict[str, list[tuple[int, int]]]:
        action_slices: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for subkey, metadata in self.lerobot_modality_meta.action.items():
            subkey_lower = subkey.lower()
            if "joint" not in subkey_lower or "gripper" in subkey_lower or not metadata.absolute:
                continue
            original_key = metadata.original_key or "action"
            action_slices[original_key].append((metadata.start, metadata.end))
        return action_slices

    def _get_absolute_gripper_action_slices(self) -> dict[str, list[tuple[int, int]]]:
        action_slices: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for subkey, metadata in self.lerobot_modality_meta.action.items():
            subkey_lower = subkey.lower()
            if "gripper" not in subkey_lower or not metadata.absolute:
                continue
            original_key = metadata.original_key or "action"
            action_slices[original_key].append((metadata.start, metadata.end))
        return action_slices

    def _stats_slices_exceed_abs_limit(
        self,
        le_statistics: dict,
        action_slices: dict[str, list[tuple[int, int]]],
        limit: float,
    ) -> bool:
        for original_key, slices in action_slices.items():
            if original_key not in le_statistics:
                continue
            stats = le_statistics[original_key]
            if not isinstance(stats, dict):
                continue
            min_values = np.asarray(stats.get("min", []), dtype=np.float32)
            max_values = np.asarray(stats.get("max", []), dtype=np.float32)
            if min_values.ndim == 0:
                min_values = min_values[None]
            if max_values.ndim == 0:
                max_values = max_values[None]
            for start, end in slices:
                if np.any(max_values[start:end] > limit) or np.any(min_values[start:end] < -limit):
                    return True
        return False

    def _extract_abs_le_statistics(self, loaded_stats: object) -> dict | None:
        if not isinstance(loaded_stats, dict):
            return None

        payload_stats = loaded_stats.get("statistics")
        if isinstance(payload_stats, dict):
            if "abs" in payload_stats and isinstance(payload_stats["abs"], dict):
                return payload_stats["abs"]
            if "action" in payload_stats or "state" in payload_stats:
                return None
            return payload_stats

        if "abs" in loaded_stats and isinstance(loaded_stats["abs"], dict):
            return loaded_stats["abs"]
        if "action" in loaded_stats or "state" in loaded_stats:
            return None
        return loaded_stats

    def _load_unfiltered_abs_le_statistics(self) -> dict | None:
        stats_path = self.dataset_path / LE_ROBOT_STATS_FILENAME
        if stats_path.exists():
            try:
                with open(stats_path, "r") as f:
                    return self._extract_abs_le_statistics(json.load(f))
            except Exception as e:
                print(
                    f"[RANK {os.environ.get('RANK', 'NA')}] "
                    f"Failed to load raw stats for outlier pre-check ({e}), rebuilding..."
                )

        parquet_files = self._list_all_statistics_parquet_files()
        if not parquet_files:
            return None
        return calculate_dataset_statistics(parquet_files)

    def _dataset_stats_exceed_outlier_abs_limit(self, le_statistics: dict) -> bool:
        if self._filter_joint_outlier_trajectory_configured() and self._stats_slices_exceed_abs_limit(
            le_statistics,
            self._get_absolute_joint_action_slices(),
            self._get_outlier_abs_limit(),
        ):
            return True
        if self._filter_gripper_outlier_trajectory_configured() and self._stats_slices_exceed_abs_limit(
            le_statistics,
            self._get_absolute_gripper_action_slices(),
            self._get_gripper_outlier_abs_limit(),
        ):
            return True
        return False

    def _dataset_needs_outlier_filtering(self) -> bool:
        if self._outlier_filter_required is not None:
            return self._outlier_filter_required
        if not self._filter_outlier_trajectory_configured():
            self._outlier_filter_required = False
            return False

        has_joint_slices = (
            self._filter_joint_outlier_trajectory_configured() and bool(self._get_absolute_joint_action_slices())
        )
        has_gripper_slices = (
            self._filter_gripper_outlier_trajectory_configured()
            and bool(self._get_absolute_gripper_action_slices())
        )
        if not has_joint_slices and not has_gripper_slices:
            self._outlier_filter_required = False
            return False

        try:
            le_statistics = self._load_unfiltered_abs_le_statistics()
            self._outlier_filter_required = (
                self._dataset_stats_exceed_outlier_abs_limit(le_statistics)
                if le_statistics is not None
                else False
            )
        except Exception as e:
            print(
                f"[RANK {os.environ.get('RANK', 'NA')}] "
                f"Outlier pre-check failed for {self.dataset_name} ({e}), keeping filtering enabled."
            )
            self._outlier_filter_required = True
        return self._outlier_filter_required

    def _list_all_statistics_parquet_files(self) -> list[Path]:
        return [
            parquet_path
            for parquet_path in self.dataset_path.glob(LE_ROBOT_DATA_FILENAME)
            if "episode_033675.parquet" not in parquet_path.name
        ]

    def _trajectory_has_outlier_action(self, trajectory_df: pd.DataFrame) -> bool:
        checks: list[tuple[dict[str, list[tuple[int, int]]], float]] = []
        if self._filter_joint_outlier_trajectory_configured():
            checks.append((self._get_absolute_joint_action_slices(), self._get_outlier_abs_limit()))
        if self._filter_gripper_outlier_trajectory_configured():
            checks.append((self._get_absolute_gripper_action_slices(), self._get_gripper_outlier_abs_limit()))

        for action_slices, limit in checks:
            for original_key, slices in action_slices.items():
                if original_key not in trajectory_df.columns:
                    continue
                values = np.stack(trajectory_df[original_key].to_numpy()).astype(np.float32)
                if values.ndim == 1:
                    values = values[:, None]
                for start, end in slices:
                    chunk = values[:, start:end]
                    if np.any(chunk > limit) or np.any(chunk < -limit):
                        return True
        return False

    def _trajectory_has_large_delta_eef(self, trajectory_df: pd.DataFrame) -> bool:
        if not self._filter_delta_eef_trajectory_enabled():
            return False
        if self._uses_direct_delta_eef_filter_target():
            position_action_keys, rotation_action_keys = self._get_delta_eef_filter_action_keys()
            position_target = self._extract_delta_eef_filter_target(
                trajectory_df,
                position_action_keys,
            )
            rotation_target = self._extract_delta_eef_filter_target(
                trajectory_df,
                rotation_action_keys,
            )
            position_limit = self._get_delta_eef_position_abs_limit()
            rotation_limit = self._get_delta_eef_rotation_abs_limit()
            position_bad = np.any(np.abs(position_target) > position_limit)
            rotation_bad = np.any(np.abs(rotation_target) > rotation_limit)
            return bool(position_bad or rotation_bad)
        if FRANKA_DELTA_EEF_ACTION_KEY not in trajectory_df.columns:
            raise ValueError(f"Missing required column `{FRANKA_DELTA_EEF_ACTION_KEY}` for delta-eef filtering.")
        if FRANKA_DELTA_EEF_STATE_KEY not in trajectory_df.columns:
            raise ValueError(f"Missing required column `{FRANKA_DELTA_EEF_STATE_KEY}` for delta-eef filtering.")

        action_pose = np.stack(trajectory_df[FRANKA_DELTA_EEF_ACTION_KEY].to_numpy()).astype(np.float32)
        state_pose = np.stack(trajectory_df[FRANKA_DELTA_EEF_STATE_KEY].to_numpy()).astype(np.float32)
        if action_pose.ndim == 1:
            action_pose = action_pose[:, None]
        if state_pose.ndim == 1:
            state_pose = state_pose[:, None]
        if action_pose.shape != state_pose.shape:
            raise ValueError(
                "Delta-eef filtering requires matching action/state pose shapes, "
                f"got {action_pose.shape} vs {state_pose.shape}."
            )
        if action_pose.shape[1] < FRANKA_DELTA_EEF_ROTATION_SLICE.stop:
            raise ValueError(
                "Delta-eef filtering expects at least 6 pose dimensions, "
                f"got {action_pose.shape[1]}."
            )

        delta_pose = action_pose - state_pose
        delta_pose[:, FRANKA_DELTA_EEF_ROTATION_SLICE] = _wrap_rotation_delta(
            delta_pose[:, FRANKA_DELTA_EEF_ROTATION_SLICE]
        )

        position_limit = self._get_delta_eef_position_abs_limit()
        rotation_limit = self._get_delta_eef_rotation_abs_limit()
        position_bad = np.any(
            np.abs(delta_pose[:, FRANKA_DELTA_EEF_POSITION_SLICE]) > position_limit
        )
        rotation_bad = np.any(
            np.abs(delta_pose[:, FRANKA_DELTA_EEF_ROTATION_SLICE]) > rotation_limit
        )
        return bool(position_bad or rotation_bad)

    def _get_delta_eef_filter_action_keys(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        action_keys = tuple(self.modality_keys.get("action", []))
        action_key_set = set(action_keys)
        candidate_key_groups = (
            (("action.eef_position",), ("action.eef_rotation",)),
            (("action.delta_eef_position",), ("action.delta_eef_rotation",)),
            (
                ("action.x", "action.y", "action.z"),
                ("action.roll", "action.pitch", "action.yaw"),
            ),
            (
                (
                    "action.robot_0_delta_eef_position",
                    "action.robot_1_delta_eef_position",
                ),
                (
                    "action.robot_0_delta_eef_rotation",
                    "action.robot_1_delta_eef_rotation",
                ),
            ),
        )

        for position_action_keys, rotation_action_keys in candidate_key_groups:
            required_keys = (*position_action_keys, *rotation_action_keys)
            if all(key in action_key_set for key in required_keys):
                return position_action_keys, rotation_action_keys

        raise ValueError(
            "Delta-eef filtering requires action position/rotation keys. "
            f"Available action keys: {list(action_keys)}"
        )

    def _get_action_chunk_padding_strategy(self, action_key: str) -> str:
        key_meta = self.lerobot_modality_meta.get_key_meta(action_key)
        return "first_last" if key_meta.absolute else "zero"

    def _get_delta_eef_filter_indices(self, action_keys: tuple[str, ...]) -> np.ndarray:
        indices = np.asarray(self.delta_indices[action_keys[0]], dtype=np.int64)
        for action_key in action_keys[1:]:
            candidate = np.asarray(self.delta_indices[action_key], dtype=np.int64)
            if not np.array_equal(candidate, indices):
                raise ValueError(
                    "Delta-eef filtering requires matching delta indices across action keys, "
                    f"got {action_keys[0]}={indices.tolist()} vs {action_key}={candidate.tolist()}."
                )
        return indices

    def _get_delta_eef_filter_padding_strategy(self, action_keys: tuple[str, ...]) -> str:
        padding_strategy = self._get_action_chunk_padding_strategy(action_keys[0])
        for action_key in action_keys[1:]:
            candidate = self._get_action_chunk_padding_strategy(action_key)
            if candidate != padding_strategy:
                raise ValueError(
                    "Delta-eef filtering requires matching padding strategies across action keys, "
                    f"got {action_keys[0]}={padding_strategy} vs {action_key}={candidate}."
                )
        return padding_strategy

    def _extract_delta_eef_filter_target(
        self,
        trajectory_df: pd.DataFrame,
        action_keys: tuple[str, ...],
    ) -> np.ndarray:
        if len(action_keys) == 1:
            target = np.asarray(
                self._extract_transformed_action_sequence_from_trajectory(
                    trajectory_df,
                    action_keys[0],
                ),
                dtype=np.float32,
            )
            return target[:, None] if target.ndim == 1 else target

        component_targets = []
        for action_key in action_keys:
            target = np.asarray(
                self._extract_transformed_action_sequence_from_trajectory(
                    trajectory_df,
                    action_key,
                ),
                dtype=np.float32,
            )
            if target.ndim == 1:
                target = target[:, None]
            if target.ndim != 2:
                raise ValueError(
                    "Split delta-eef filtering expects 2D action components, "
                    f"got {action_key} with shape {target.shape}."
                )
            component_targets.append(target)
        return np.concatenate(component_targets, axis=1)

    def _extract_delta_eef_filter_target_from_chunk_data(
        self,
        data: dict,
        action_keys: tuple[str, ...],
    ) -> np.ndarray:
        if len(action_keys) == 1:
            target = np.asarray(data[action_keys[0]], dtype=np.float32)
            return target[:, None] if target.ndim == 1 else target

        component_targets = []
        for action_key in action_keys:
            target = np.asarray(data[action_key], dtype=np.float32)
            if target.ndim == 1:
                target = target[:, None]
            if target.ndim != 2:
                raise ValueError(
                    "Split delta-eef filtering expects 2D action components, "
                    f"got {action_key} with shape {target.shape}."
                )
            component_targets.append(target)
        return np.concatenate(component_targets, axis=1)

    def _chunk_contains_flag(
        self,
        flags: np.ndarray,
        step_indices: np.ndarray,
        padding_strategy: str,
    ) -> bool:
        if flags.ndim != 1:
            raise ValueError(f"Expected 1D flag array, got shape {flags.shape}")
        if flags.size == 0:
            return False

        max_length = flags.shape[0]
        front_padding = step_indices < 0
        end_padding = step_indices >= max_length
        padding_positions = np.logical_or(front_padding, end_padding)
        chunk_flags = np.zeros(len(step_indices), dtype=bool)
        if (~padding_positions).any():
            chunk_flags[~padding_positions] = flags[step_indices[~padding_positions]]
        if padding_positions.any():
            if padding_strategy == "first_last":
                chunk_flags[front_padding] = flags[0]
                chunk_flags[end_padding] = flags[-1]
            elif padding_strategy == "zero":
                chunk_flags[padding_positions] = False
            else:
                raise ValueError(f"Invalid padding strategy: {padding_strategy}")
        return bool(np.any(chunk_flags))

    def _get_chunk_flags(
        self,
        flags: np.ndarray,
        step_indices: np.ndarray,
        padding_strategy: str,
    ) -> np.ndarray:
        if flags.ndim != 1:
            raise ValueError(f"Expected 1D flag array, got shape {flags.shape}")
        if flags.size == 0:
            return np.zeros(len(step_indices), dtype=bool)

        max_length = flags.shape[0]
        front_padding = step_indices < 0
        end_padding = step_indices >= max_length
        padding_positions = np.logical_or(front_padding, end_padding)
        chunk_flags = np.zeros(len(step_indices), dtype=bool)
        if (~padding_positions).any():
            chunk_flags[~padding_positions] = flags[step_indices[~padding_positions]]
        if padding_positions.any():
            if padding_strategy == "first_last":
                chunk_flags[front_padding] = flags[0]
                chunk_flags[end_padding] = flags[-1]
            elif padding_strategy == "zero":
                chunk_flags[padding_positions] = False
            else:
                raise ValueError(f"Invalid padding strategy: {padding_strategy}")
        return chunk_flags

    def _get_valid_delta_eef_step_mask(
        self,
        trajectory_df: pd.DataFrame,
        trajectory_length: int,
    ) -> np.ndarray:
        position_limit = self._get_delta_eef_position_abs_limit()
        rotation_limit = self._get_delta_eef_rotation_abs_limit()
        position_action_keys, rotation_action_keys = self._get_delta_eef_filter_action_keys()
        position_indices = self._get_delta_eef_filter_indices(position_action_keys)
        rotation_indices = self._get_delta_eef_filter_indices(rotation_action_keys)
        position_padding = self._get_delta_eef_filter_padding_strategy(position_action_keys)
        rotation_padding = self._get_delta_eef_filter_padding_strategy(rotation_action_keys)

        if self._uses_direct_delta_eef_filter_target():
            position_target = self._extract_delta_eef_filter_target(
                trajectory_df,
                position_action_keys,
            )
            rotation_target = self._extract_delta_eef_filter_target(
                trajectory_df,
                rotation_action_keys,
            )
            position_bad = np.any(np.abs(position_target) > position_limit, axis=1)
            rotation_bad = np.any(np.abs(rotation_target) > rotation_limit, axis=1)
        else:
            if FRANKA_DELTA_EEF_ACTION_KEY not in trajectory_df.columns:
                raise ValueError(
                    f"Missing required column `{FRANKA_DELTA_EEF_ACTION_KEY}` for delta-eef step filtering."
                )
            if FRANKA_DELTA_EEF_STATE_KEY not in trajectory_df.columns:
                raise ValueError(
                    f"Missing required column `{FRANKA_DELTA_EEF_STATE_KEY}` for delta-eef step filtering."
                )

            action_pose = np.stack(trajectory_df[FRANKA_DELTA_EEF_ACTION_KEY].to_numpy()).astype(np.float32)
            state_pose = np.stack(trajectory_df[FRANKA_DELTA_EEF_STATE_KEY].to_numpy()).astype(np.float32)
            if action_pose.ndim == 1:
                action_pose = action_pose[:, None]
            if state_pose.ndim == 1:
                state_pose = state_pose[:, None]
            if action_pose.shape != state_pose.shape:
                raise ValueError(
                    "Delta-eef step filtering requires matching action/state pose shapes, "
                    f"got {action_pose.shape} vs {state_pose.shape}."
                )
            if action_pose.shape[1] < FRANKA_DELTA_EEF_ROTATION_SLICE.stop:
                raise ValueError(
                    "Delta-eef step filtering expects at least 6 pose dimensions, "
                    f"got {action_pose.shape[1]}."
                )

            delta_pose = action_pose - state_pose
            delta_pose[:, FRANKA_DELTA_EEF_ROTATION_SLICE] = _wrap_rotation_delta(
                delta_pose[:, FRANKA_DELTA_EEF_ROTATION_SLICE]
            )
            position_bad = np.any(
                np.abs(delta_pose[:, FRANKA_DELTA_EEF_POSITION_SLICE]) > position_limit,
                axis=1,
            )
            rotation_bad = np.any(
                np.abs(delta_pose[:, FRANKA_DELTA_EEF_ROTATION_SLICE]) > rotation_limit,
                axis=1,
            )

        valid_mask = np.ones(int(trajectory_length), dtype=bool)
        valid_ratio_threshold = self._get_delta_eef_valid_ratio()
        for base_index in range(int(trajectory_length)):
            position_chunk_bad = self._get_chunk_flags(
                position_bad,
                position_indices + base_index,
                position_padding,
            )
            rotation_chunk_bad = self._get_chunk_flags(
                rotation_bad,
                rotation_indices + base_index,
                rotation_padding,
            )
            combined_chunk_valid = np.logical_not(np.logical_or(position_chunk_bad, rotation_chunk_bad))
            if combined_chunk_valid.size == 0:
                valid_mask[base_index] = False
                continue
            if float(np.count_nonzero(combined_chunk_valid)) / float(combined_chunk_valid.size) < valid_ratio_threshold:
                valid_mask[base_index] = False
        return valid_mask

    def _get_action_valid_mask_from_data(self, data: dict) -> np.ndarray | None:
        if not self._filter_delta_eef_steps_enabled():
            return None

        position_limit = self._get_delta_eef_position_abs_limit()
        rotation_limit = self._get_delta_eef_rotation_abs_limit()
        position_action_keys, rotation_action_keys = self._get_delta_eef_filter_action_keys()

        position_target = self._extract_delta_eef_filter_target_from_chunk_data(data, position_action_keys)
        rotation_target = self._extract_delta_eef_filter_target_from_chunk_data(data, rotation_action_keys)
        position_bad = np.any(np.abs(position_target) > position_limit, axis=1)
        rotation_bad = np.any(np.abs(rotation_target) > rotation_limit, axis=1)
        return np.logical_not(np.logical_or(position_bad, rotation_bad))

    def _trajectory_has_invalid_droid_task(self, trajectory_df: pd.DataFrame) -> bool:
        if not self._filter_invalid_droid_task_enabled():
            return False
        if "task_index" not in trajectory_df.columns or trajectory_df.empty:
            return False

        task_index_value = trajectory_df["task_index"].iloc[0]
        task_index = int(task_index_value if isinstance(task_index_value, (int, float)) else task_index_value.item())
        if task_index not in self.tasks.index:
            print(f"Skipping trajectory due to missing task metadata: task_index={task_index}")
            return True

        task_row = self.tasks.loc[task_index]
        if isinstance(task_row, pd.DataFrame):
            task_row = task_row.iloc[0]
        for column in ("task_name", "task"):
            if column in task_row and is_invalid_droid_task_text(task_row[column]):
                return True
        return False

    def _get_stats_cache_path(self) -> Path:
        if not self._trajectory_filtering_enabled():
            return self.dataset_path / LE_ROBOT_STATS_FILENAME
        cache_key = self._get_trajectory_filter_config_key()
        return self.dataset_path / "meta" / f"stats_gr00t_filtered_{cache_key}.json"

    def _get_steps_cache_path(self) -> Path:
        if not self._trajectory_filtering_enabled():
            return self.dataset_path / "meta" / "steps_data_index.pkl"
        cache_key = self._get_trajectory_filter_config_key()
        return self.dataset_path / "meta" / f"steps_data_index_filtered_{cache_key}.pkl"

    def _get_allowed_step_indices_by_parquet(self) -> dict[str, np.ndarray] | None:
        if not self._filter_delta_eef_steps_enabled() or self._lerobot_version != "v2.0":
            return None

        step_indices_by_path: dict[str, np.ndarray] = {}
        current_trajectory_id = None
        current_indices: list[int] = []

        def flush_current() -> None:
            if current_trajectory_id is None:
                return
            trajectory_id = int(current_trajectory_id)
            chunk_index = trajectory_id // self.chunk_size
            parquet_path = self.dataset_path / self.data_path_pattern.format(
                episode_chunk=chunk_index,
                episode_index=trajectory_id,
            )
            step_indices_by_path[str(parquet_path)] = np.asarray(current_indices, dtype=np.int64)

        for trajectory_id, base_index in tqdm(
            self.all_steps,
            desc="Indexing filtered steps for stats",
            disable=len(self.all_steps) < 1_000_000,
        ):
            trajectory_id = int(trajectory_id)
            if current_trajectory_id is None:
                current_trajectory_id = trajectory_id
            elif trajectory_id != current_trajectory_id:
                flush_current()
                current_trajectory_id = trajectory_id
                current_indices = []
            current_indices.append(int(base_index))
        flush_current()

        return step_indices_by_path

    def _get_statistics_parquet_files(self) -> list[Path]:
        if not self._trajectory_filtering_enabled() or self._lerobot_version != "v2.0":
            return self._list_all_statistics_parquet_files()

        filtered_trajectory_ids = sorted({int(trajectory_id) for trajectory_id, _ in self.all_steps})
        parquet_files = []
        for trajectory_id in filtered_trajectory_ids:
            chunk_index = self.get_episode_chunk(trajectory_id)
            parquet_path = self.dataset_path / self.data_path_pattern.format(
                episode_chunk=chunk_index, episode_index=trajectory_id
            )
            if parquet_path.exists() and "episode_033675.parquet" not in parquet_path.name:
                parquet_files.append(parquet_path)
        return parquet_files

    def _load_cached_steps(self, steps_path: Path) -> list[tuple[int, int]] | None:
        if not steps_path.exists():
            return None
        expected_config_key = self._get_steps_config_key()
        try:
            with open(steps_path, "rb") as f:
                cached_data = pickle.load(f)
            if cached_data.get("config_key") != expected_config_key:
                return None
            return cached_data["steps"]
        except Exception as e:
            print(
                f"[RANK {os.environ.get('RANK', 'NA')}] "
                f"Failed to load cached steps ({e}), will rebuild."
            )
            return None

    def _get_all_steps(self) -> list[tuple[int, int]]:
        """Get the trajectory IDs and base indices for all steps in the dataset.

        Returns:
            list[tuple[str, int]]: A list of (trajectory_id, base_index) tuples.
        """
        def is_main():
            return (not dist.is_initialized()) or dist.get_rank() == 0
    
        config_key = self._get_steps_config_key()
        steps_path = self._get_steps_cache_path()
    
        # ---------- try to read from cache  ----------
        cached_steps = self._load_cached_steps(steps_path)
        if cached_steps is not None and not dist.is_initialized():
            return cached_steps

        # In distributed runs every rank must execute the same collectives in
        # the same order. Let rank0 be the source even when other ranks can
        # read the cache, otherwise later broadcasts can receive this steps list.
        if is_main():
            if cached_steps is not None:
                all_steps = cached_steps
            else:
                all_steps = self._get_all_steps_single_process()
                summary = getattr(self, "_last_step_filter_summary", {}) or {}

                cache_data = {
                    "config_key": config_key,
                    "steps": all_steps,
                    "num_trajectories": len(self.trajectory_ids),
                    "filtered_trajectory_count": self._count_filtered_trajectories(all_steps),
                    "total_steps": len(all_steps),
                    "original_step_count": int(np.sum(self.trajectory_lengths)),
                    "computed_timestamp": pd.Timestamp.now().isoformat(),
                    "delete_pause_frame": self.delete_pause_frame,
                }
                cache_data.update(summary)

                try:
                    steps_path.parent.mkdir(parents=True, exist_ok=True)
                    tmp_path = steps_path.with_suffix(".tmp")

                    with open(tmp_path, "wb") as f:
                        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                    os.replace(tmp_path, steps_path)

                    print(f"[RANK 0] Cached steps saved to {steps_path}")
                except Exception as e:
                    print(
                        f"[RANK 0] Failed to write cached steps ({e}), "
                        "continuing with in-memory steps."
                    )
                    tmp_path = steps_path.with_suffix(".tmp")
                    if tmp_path.exists():
                        try:
                            tmp_path.unlink()
                        except OSError:
                            pass
        else:
            all_steps = None

        # ---------- sync steps from rank0 ----------
        if dist.is_initialized():
            obj_list = [all_steps]
            dist.broadcast_object_list(obj_list, src=0)
            all_steps = obj_list[0]
            assert all_steps is not None
            return all_steps

        assert all_steps is not None
        return all_steps

    def _get_steps_config_key(self) -> str:
        """Generate a configuration key for steps caching."""
        return self._get_trajectory_filter_config_key()

    def _count_filtered_trajectories(self, steps: list[tuple[int, int]] | None = None) -> int:
        if not self._trajectory_filtering_enabled():
            return int(len(self._trajectory_ids))
        step_source = self._all_steps if steps is None else steps
        return len({int(trajectory_id) for trajectory_id, _ in step_source})


    def _get_all_steps_single_process(self) -> list[tuple[int, int]]:
        """Original single-process implementation as fallback."""
        all_steps: list[tuple[int, int]] = []
        skipped_trajectories = 0
        processed_trajectories = 0
        filtered_steps = 0
        semantic_filtered_trajectories = 0
        semantic_filtered_steps = 0
        original_step_count = int(np.sum(self.trajectory_lengths))

        # Check if language modality is configured
        has_language_modality = 'language' in self.modality_keys and len(self.modality_keys['language']) > 0
        desc = "Filtering Trajectory" if self._trajectory_filtering_enabled() else "Getting All Step"
        # TODO why trajectory_length here, why not use data length?
        for trajectory_id, trajectory_length in tqdm(
            zip(self.trajectory_ids, self.trajectory_lengths),
            total=len(self.trajectory_ids),
            desc=desc,
        ):
            try:
                if self._lerobot_version == "v2.0":
                    data = self.get_trajectory_data(trajectory_id)
                elif self._lerobot_version == "v3.0":
                    data = self.get_trajectory_data_lerobot_v3(trajectory_id)
                
                trajectory_skipped = False
                valid_step_mask = None

                if self._filter_outlier_trajectory_enabled() and self._trajectory_has_outlier_action(data):
                    skipped_trajectories += 1
                    trajectory_skipped = True
                    continue

                if self._filter_delta_eef_trajectory_enabled() and self._trajectory_has_large_delta_eef(data):
                    skipped_trajectories += 1
                    trajectory_skipped = True
                    continue

                if self._trajectory_has_invalid_droid_task(data):
                    skipped_trajectories += 1
                    semantic_filtered_trajectories += 1
                    semantic_filtered_steps += int(trajectory_length)
                    trajectory_skipped = True
                    continue

                if self._filter_delta_eef_steps_enabled():
                    valid_step_mask = self._get_valid_delta_eef_step_mask(data, int(trajectory_length))
                    filtered_steps += int(len(valid_step_mask) - np.count_nonzero(valid_step_mask))
                    if not np.any(valid_step_mask):
                        skipped_trajectories += 1
                        trajectory_skipped = True
                        continue
            
                # Check if trajectory has valid language instruction (if language modality is configured)
                if has_language_modality:
                    self.curr_traj_data = data  # Set current trajectory data for get_language to work

                    language_instruction = self.get_language(trajectory_id, self.modality_keys['language'][0], 0)
                    if not language_instruction or language_instruction[0] == "":
                        print(f"Skipping trajectory {trajectory_id} due to empty language instruction")
                        skipped_trajectories += 1
                        semantic_filtered_trajectories += 1
                        semantic_filtered_steps += int(trajectory_length)
                        trajectory_skipped = True
                        continue

            except Exception as e:
                print(f"Skipping trajectory {trajectory_id} due to read error: {e}")
                skipped_trajectories += 1
                trajectory_skipped = True
                continue
        
            if not trajectory_skipped:
                processed_trajectories += 1
        
            if valid_step_mask is None:
                for base_index in range(trajectory_length):
                    all_steps.append((trajectory_id, base_index))
            else:
                for base_index in np.flatnonzero(valid_step_mask):
                    all_steps.append((trajectory_id, int(base_index)))
                
        # Print summary statistics
        print(
            f"Single-process summary: Processed {processed_trajectories} trajectories, "
            f"skipped {skipped_trajectories} trajectories after filtering/validation"
        )
        if self._filter_delta_eef_steps_enabled():
            print(f"Step-level delta-eef filtering removed {filtered_steps} steps")
        print(f"Total steps: {len(all_steps)} from {len(self.trajectory_ids)} trajectories")
        if semantic_filtered_trajectories:
            print(
                "Semantic/language filtering removed "
                f"{semantic_filtered_trajectories} instruction trajectories "
                f"and {semantic_filtered_steps} candidate action chunks"
            )

        self._last_step_filter_summary = {
            "semantic_filtered_trajectory_count": int(semantic_filtered_trajectories),
            "semantic_filtered_step_count": int(semantic_filtered_steps),
            "post_semantic_step_count": int(original_step_count - semantic_filtered_steps),
            "delta_eef_filtered_step_count": int(filtered_steps),
        }
                   
        return all_steps

    def _get_position_and_gripper_values(self, data: pd.DataFrame) -> tuple[list, list]:
        """Get position and gripper values based on available columns in the dataset."""
        # Get action keys from modality_keys
        action_keys = self.modality_keys.get('action', [])
        
        # Extract position data
        delta_position_values = None
        position_candidates = ['delta_eef_position']
        coordinate_candidates = ['x', 'y', 'z']
        
        # First try combined position fields
        for pos_key in position_candidates:
            full_key = f"action.{pos_key}"
            if full_key in action_keys:
                try:
                    # Get the lerobot key for this modality
                    le_action_cfg = self.lerobot_modality_meta.action
                    subkey = pos_key
                    if subkey in le_action_cfg:
                        le_key = le_action_cfg[subkey].original_key or subkey
                        if le_key in data.columns:
                            data_array = np.stack(data[le_key])
                            le_indices = np.arange(le_action_cfg[subkey].start, le_action_cfg[subkey].end)
                            filtered_data = data_array[:, le_indices]
                            delta_position_values = filtered_data.tolist()
                            break
                except Exception:
                    continue
        
        # If combined fields not found, try individual x,y,z coordinates
        if delta_position_values is None:
            x_data, y_data, z_data = None, None, None
            for coord in coordinate_candidates:
                full_key = f"action.{coord}"
                if full_key in action_keys:
                    try:
                        le_action_cfg = self.lerobot_modality_meta.action
                        if coord in le_action_cfg:
                            le_key = le_action_cfg[coord].original_key or coord
                            if le_key in data.columns:
                                data_array = np.stack(data[le_key])
                                le_indices = np.arange(le_action_cfg[coord].start, le_action_cfg[coord].end)
                                coord_data = data_array[:, le_indices].flatten()
                                if coord == 'x':
                                    x_data = coord_data
                                elif coord == 'y':
                                    y_data = coord_data
                                elif coord == 'z':
                                    z_data = coord_data
                    except Exception:
                        continue
            
            if x_data is not None and y_data is not None and z_data is not None:
                delta_position_values = np.column_stack((x_data, y_data, z_data)).tolist()
        
        if delta_position_values is None:
            # Fallback to the old hardcoded approach if metadata approach fails
            if 'action.delta_eef_position' in data.columns:
                delta_position_values = data['action.delta_eef_position'].to_numpy().tolist()
            elif all(col in data.columns for col in ['action.x', 'action.y', 'action.z']):
                x_vals = data['action.x'].to_numpy()
                y_vals = data['action.y'].to_numpy() 
                z_vals = data['action.z'].to_numpy()
                delta_position_values = np.column_stack((x_vals, y_vals, z_vals)).tolist()
            else:
                raise ValueError(f"No suitable position columns found. Available columns: {data.columns.tolist()}")
        
        # Extract gripper data
        gripper_values = None
        gripper_candidates = ['gripper_close', 'gripper']
        
        for grip_key in gripper_candidates:
            full_key = f"action.{grip_key}"
            if full_key in action_keys:
                try:
                    le_action_cfg = self.lerobot_modality_meta.action
                    if grip_key in le_action_cfg:
                        le_key = le_action_cfg[grip_key].original_key or grip_key
                        if le_key in data.columns:
                            data_array = np.stack(data[le_key])
                            le_indices = np.arange(le_action_cfg[grip_key].start, le_action_cfg[grip_key].end)
                            gripper_data = data_array[:, le_indices].flatten()
                            gripper_values = gripper_data.tolist()
                            break
                except Exception:
                    continue
        
        if gripper_values is None:
            # Fallback to the old hardcoded approach if metadata approach fails
            if 'action.gripper_close' in data.columns:
                gripper_values = data['action.gripper_close'].to_numpy().tolist()
            elif 'action.gripper' in data.columns:
                gripper_values = data['action.gripper'].to_numpy().tolist()
            else:
                raise ValueError(f"No suitable gripper columns found. Available columns: {data.columns.tolist()}")
        
        return delta_position_values, gripper_values

    def _get_modality_keys(self) -> dict:
        """Get the modality keys for the dataset.
        The keys are the modality names, and the values are the keys for each modality.
        See property `modality_keys` for the expected format.
        """
        modality_keys = defaultdict(list)
        for modality, config in self.modality_configs.items():
            modality_keys[modality] = config.modality_keys
        return modality_keys

    def _get_delta_indices(self) -> dict[str, np.ndarray]:
        """Restructure the delta indices to use modality.key as keys instead of just the modalities."""
        delta_indices: dict[str, np.ndarray] = {}
        for config in self.modality_configs.values():
            for key in config.modality_keys:
                delta_indices[key] = np.array(config.delta_indices)
        return delta_indices

    def _init_action_mode(self) -> None:
        if self.data_cfg is None:
            self._action_mode = "abs"
            return

        action_mode = self.data_cfg.get("action_mode", "abs")
        if action_mode is None:
            action_mode = "abs"
        action_mode = str(action_mode).lower()
        if action_mode in {"absolute", "raw"}:
            action_mode = "abs"
        if action_mode not in {"abs", "delta", "rel"}:
            raise ValueError(f"Invalid action_mode: {action_mode}. Expected one of: abs, delta, rel.")
        self._action_mode = action_mode

        apply_keys = self.data_cfg.get("action_mode_apply_keys", None)
        if apply_keys:
            normalized = []
            for key in apply_keys:
                key = str(key)
                if not key.startswith("action."):
                    key = f"action.{key}"
                normalized.append(key)
            self._action_mode_apply_keys = normalized

        state_map = self.data_cfg.get("action_mode_state_map", {}) or {}
        normalized_map = {}
        for action_key, state_key in state_map.items():
            action_key = str(action_key)
            state_key = str(state_key)
            if not action_key.startswith("action."):
                action_key = f"action.{action_key}"
            if not state_key.startswith("state."):
                state_key = f"state.{state_key}"
            normalized_map[action_key] = state_key
        self._action_mode_state_map = normalized_map

        action_mode_reference = self.data_cfg.get("action_mode_reference", "state")
        if action_mode_reference is None:
            action_mode_reference = "state"
        action_mode_reference = str(action_mode_reference).lower()
        if action_mode_reference in {"self", "action_only"}:
            action_mode_reference = "action"
        if action_mode_reference not in {"state", "action"}:
            raise ValueError(
                f"Invalid action_mode_reference: {action_mode_reference}. Expected one of: state, action."
            )
        self._action_mode_reference = action_mode_reference

    def _infer_state_key_for_action(self, action_key: str) -> str | None:
        if action_key in self._action_mode_state_map:
            return self._action_mode_state_map[action_key]

        if not action_key.startswith("action."):
            return None
        base = action_key.replace("action.", "", 1)
        candidate = f"state.{base}"
        if candidate in self.modality_keys.get("state", []):
            return candidate
        state_modality = getattr(getattr(self, "lerobot_modality_meta", None), "state", None) or {}
        if base in state_modality:
            return candidate
        if hasattr(self, "_metadata"):
            if base in getattr(self.metadata.modalities, "state", {}):
                return candidate
        try:
            self.lerobot_modality_meta.get_key_meta(candidate)
            return candidate
        except Exception:
            pass
        return None

    def _should_scale_action_target_by_fps(self, action_key: str) -> bool:
        if self._get_action_target_mode() != "delta_eef_velocity":
            return False
        action_key_lower = action_key.lower()
        if "gripper" in action_key_lower:
            return False
        is_delta_eef_key = action_key_lower in {
            "action.delta_eef_position",
            "action.delta_eef_rotation",
            "action.robot_0_delta_eef_position",
            "action.robot_0_delta_eef_rotation",
            "action.robot_1_delta_eef_position",
            "action.robot_1_delta_eef_rotation",
        }
        if self._action_mode in (None, "abs"):
            return is_delta_eef_key
        if self._action_mode != "delta":
            return False
        return action_key_lower in {
            "action.eef_position",
            "action.eef_rotation",
        } or is_delta_eef_key

    def _postprocess_action_target(self, action_key: str, action_values: np.ndarray) -> np.ndarray:
        if not self._should_scale_action_target_by_fps(action_key):
            return action_values
        fps = float(self.lerobot_info_meta.get("fps", 1))
        return action_values * fps

    def _transform_action_values(
        self,
        action_key: str,
        action_values: np.ndarray,
        state_values: np.ndarray | None = None,
    ) -> np.ndarray:
        if self._action_mode in (None, "abs"):
            return self._postprocess_action_target(action_key, action_values)

        if self._action_mode_reference == "action":
            if self._action_mode == "delta":
                next_values = np.concatenate([action_values[1:], action_values[-1:]], axis=0)
                out = next_values - action_values
                if _is_rotation_key(action_key):
                    out = _wrap_rotation_delta(out)
            elif self._action_mode == "rel":
                out = action_values - action_values[0]
            else:
                out = action_values
            return self._postprocess_action_target(action_key, out)

        if state_values is None:
            raise ValueError(f"State values are required for state-referenced action target `{action_key}`.")
        if action_values.shape[1] != state_values.shape[1]:
            raise ValueError(
                f"Action/state dim mismatch for {action_key}: {action_values.shape} vs {state_values.shape}"
            )

        state0 = state_values[0]
        if self._action_mode == "delta":
            out = action_values.copy()
            if len(out) > 1:
                out[1:] = action_values[1:] - action_values[:-1]
            out[0] = action_values[0] - state0
            if _is_rotation_key(action_key):
                out = _wrap_rotation_delta(out)
        elif self._action_mode == "rel":
            out = action_values - state0
        else:
            out = action_values
        return self._postprocess_action_target(action_key, out)

    def _extract_raw_trajectory_values(
        self,
        trajectory_df: pd.DataFrame,
        key: str,
    ) -> np.ndarray:
        key_meta = self.lerobot_modality_meta.get_key_meta(key)
        original_key = key_meta.original_key or key.split(".", 1)[1]
        if original_key not in trajectory_df.columns:
            raise ValueError(
                f"Missing required column `{original_key}` when extracting trajectory values for `{key}`."
            )
        values = np.stack(trajectory_df[original_key].to_numpy()).astype(np.float32)
        if values.ndim == 1:
            values = values[:, None]
        return values[:, key_meta.start:key_meta.end]

    def _extract_transformed_action_sequence_from_trajectory(
        self,
        trajectory_df: pd.DataFrame,
        action_key: str,
    ) -> np.ndarray:
        action_values = self._extract_raw_trajectory_values(trajectory_df, action_key)
        state_values = None
        if self._action_mode_reference == "state" and self._action_mode not in (None, "abs"):
            state_key = self._infer_state_key_for_action(action_key)
            if state_key is None:
                raise ValueError(f"Unable to infer state key for action key `{action_key}`.")
            state_values = self._extract_raw_trajectory_values(trajectory_df, state_key)
        return self._transform_action_values(action_key, action_values, state_values=state_values)

    def _resolve_available_action_key(self, action_key: str, data: dict) -> str | None:
        if action_key in data:
            return action_key

        action_aliases = {
            "action.eef_position": ("action.delta_eef_position",),
            "action.eef_rotation": ("action.delta_eef_rotation",),
            "action.delta_eef_position": ("action.eef_position",),
            "action.delta_eef_rotation": ("action.eef_rotation",),
        }
        for candidate in action_aliases.get(action_key, ()):
            if candidate in data:
                return candidate
        return None

    def _apply_action_mode(self, data: dict) -> dict:
        action_keys = self._action_mode_apply_keys or self.modality_keys.get("action", [])
        for action_key in action_keys:
            resolved_action_key = self._resolve_available_action_key(action_key, data)
            if resolved_action_key is None:
                print(f"[WARNING] Action key {action_key} not found in data")
                continue
            action_values = np.asarray(data[resolved_action_key])
            if action_values.ndim != 2:
                raise ValueError(
                    f"Expected 2D array for action, got {resolved_action_key}: {action_values.shape}"
                )
            state_values = None
            if self._action_mode_reference == "state":
                state_key = self._infer_state_key_for_action(resolved_action_key)
                if state_key is None or state_key not in data:
                    continue
                state_values = np.asarray(data[state_key])
                if state_values.ndim != 2:
                    raise ValueError(
                        f"Expected 2D array for state, got {state_key}: {state_values.shape}"
                    )
            data[resolved_action_key] = self._transform_action_values(
                resolved_action_key,
                action_values,
                state_values=state_values,
            )

        return data

    def _get_lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        """Get the metadata for the LeRobot dataset."""
        modality_meta_path = self.dataset_path / LE_ROBOT_MODALITY_FILENAME
        if modality_meta_path.exists():
            with open(modality_meta_path, "r") as f:
                modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))
            return self._augment_lerobot_modality_meta(modality_meta)

        synthesized = self._build_missing_lerobot_modality_meta()
        if synthesized is not None:
            return synthesized

        raise AssertionError(
            f"Please provide a {LE_ROBOT_MODALITY_FILENAME} file in {self.dataset_path}"
        )

    def _build_missing_lerobot_modality_meta(self) -> LeRobotModalityMetadata | None:
        info_meta_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        if not info_meta_path.exists():
            return None

        with open(info_meta_path, "r") as f:
            info_meta = json.load(f)
        features = info_meta.get("features", {})

        required_franka_keys = {
            "images.rgb.head",
            "images.rgb.hand",
            "states.gripper.pose",
            "states.gripper.position",
            "actions.gripper.pose",
            "actions.gripper.position",
        }
        if not required_franka_keys.issubset(features):
            return None

        modality_meta = LeRobotModalityMetadata(
            state={
                "eef_position": LeRobotStateMetadata(
                    start=0,
                    end=3,
                    dtype="float32",
                    original_key="states.gripper.pose",
                ),
                "eef_rotation": LeRobotStateMetadata(
                    start=3,
                    end=6,
                    dtype="float32",
                    rotation_type=RotationType.EULER_ANGLES_RPY,
                    original_key="states.gripper.pose",
                ),
                "gripper_position": LeRobotStateMetadata(
                    start=0,
                    end=1,
                    dtype="float32",
                    original_key="states.gripper.position",
                ),
            },
            action={
                "eef_position": LeRobotActionMetadata(
                    start=0,
                    end=3,
                    dtype="float32",
                    original_key="actions.gripper.pose",
                ),
                "eef_rotation": LeRobotActionMetadata(
                    start=3,
                    end=6,
                    dtype="float32",
                    rotation_type=RotationType.EULER_ANGLES_RPY,
                    original_key="actions.gripper.pose",
                ),
                "gripper_position": LeRobotActionMetadata(
                    start=0,
                    end=1,
                    dtype="float32",
                    original_key="actions.gripper.position",
                ),
            },
            video={
                "primary_image": LeRobotModalityField(original_key="images.rgb.head"),
                "wrist_image": LeRobotModalityField(original_key="images.rgb.hand"),
            },
            annotation={
                "human.action.task_description": LeRobotModalityField(
                    original_key="task_index"
                ),
            },
        )
        return self._augment_lerobot_modality_meta(modality_meta)

    def _augment_lerobot_modality_meta(
        self, modality_meta: LeRobotModalityMetadata
    ) -> LeRobotModalityMetadata:
        modality_meta = self._maybe_augment_split_aloha_pose_metadata(modality_meta)
        modality_meta = self._maybe_augment_annotation_aliases(modality_meta)
        modality_meta = self._maybe_augment_video_key_aliases(modality_meta)
        return modality_meta

    def _maybe_augment_split_aloha_pose_metadata(
        self, modality_meta: LeRobotModalityMetadata
    ) -> LeRobotModalityMetadata:
        required_keys = {
            key
            for config in self.modality_configs.values()
            for key in config.modality_keys
        }
        if not any(key.endswith("eef_pose") for key in required_keys):
            return modality_meta

        synthetic_state_fields = {
            "left_armbase_eef_pose": LeRobotStateMetadata(
                start=0,
                end=7,
                dtype="float32",
                original_key="states.left_ee_to_left_armbase_pose",
            ),
            "right_armbase_eef_pose": LeRobotStateMetadata(
                start=0,
                end=7,
                dtype="float32",
                original_key="states.right_ee_to_right_armbase_pose",
            ),
            "left_robot_eef_pose": LeRobotStateMetadata(
                start=0,
                end=7,
                dtype="float32",
                original_key="states.left_ee_to_robot_pose",
            ),
            "right_robot_eef_pose": LeRobotStateMetadata(
                start=0,
                end=7,
                dtype="float32",
                original_key="states.right_ee_to_robot_pose",
            ),
        }
        synthetic_action_fields = {
            "left_armbase_eef_pose": LeRobotActionMetadata(
                start=0,
                end=7,
                dtype="float32",
                original_key="actions.left_ee_to_left_armbase_pose",
            ),
            "right_armbase_eef_pose": LeRobotActionMetadata(
                start=0,
                end=7,
                dtype="float32",
                original_key="actions.right_ee_to_right_armbase_pose",
            ),
            "left_robot_eef_pose": LeRobotActionMetadata(
                start=0,
                end=7,
                dtype="float32",
                original_key="actions.left_ee_to_robot_pose",
            ),
            "right_robot_eef_pose": LeRobotActionMetadata(
                start=0,
                end=7,
                dtype="float32",
                original_key="actions.right_ee_to_robot_pose",
            ),
        }

        for subkey, metadata in synthetic_state_fields.items():
            if subkey not in modality_meta.state:
                modality_meta.state[subkey] = metadata
        for subkey, metadata in synthetic_action_fields.items():
            if subkey not in modality_meta.action:
                modality_meta.action[subkey] = metadata
        return modality_meta

    def _maybe_augment_annotation_aliases(
        self, modality_meta: LeRobotModalityMetadata
    ) -> LeRobotModalityMetadata:
        required_keys = {
            key
            for config in self.modality_configs.values()
            for key in config.modality_keys
        }
        annotation_aliases = {
            "annotation.task_index": "task_index",
            "annotation.substask": "annotation.substask",
        }
        requested_aliases = {
            key.replace("annotation.", ""): original_key
            for key, original_key in annotation_aliases.items()
            if key in required_keys
        }
        if not requested_aliases:
            return modality_meta

        if modality_meta.annotation is None:
            modality_meta.annotation = {}
        for subkey, original_key in requested_aliases.items():
            if subkey not in modality_meta.annotation:
                modality_meta.annotation[subkey] = LeRobotModalityField(original_key=original_key)
        return modality_meta

    def _maybe_augment_video_key_aliases(
        self, modality_meta: LeRobotModalityMetadata
    ) -> LeRobotModalityMetadata:
        video_alias_pairs = [
            ("cam_high_rgb", "cam_front_rgb"),
            ("cam_left_wrist_rgb", "cam_left_wrist_rgb_rgb"),
            ("cam_right_wrist_rgb", "cam_right_wrist_rgb_rgb"),
        ]
        for canonical_key, alias_key in video_alias_pairs:
            if canonical_key in modality_meta.video and alias_key not in modality_meta.video:
                modality_meta.video[alias_key] = LeRobotModalityField(
                    original_key=modality_meta.video[canonical_key].original_key
                )
            elif alias_key in modality_meta.video and canonical_key not in modality_meta.video:
                modality_meta.video[canonical_key] = LeRobotModalityField(
                    original_key=modality_meta.video[alias_key].original_key
                )
        return modality_meta

    def _get_lerobot_info_meta(self) -> dict:
        """Get the metadata for the LeRobot dataset."""
        info_meta_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        with open(info_meta_path, "r") as f:
            info_meta = json.load(f)
        return info_meta

    def _get_data_path_pattern(self) -> str:
        """Get the data path pattern for the LeRobot dataset."""
        return self.lerobot_info_meta["data_path"]

    def _get_video_path_pattern(self) -> str:
        """Get the video path pattern for the LeRobot dataset."""
        return self.lerobot_info_meta["video_path"]

    def _normalize_video_key(self, key: str) -> str:
        return key.replace("video.", "", 1) if key.startswith("video.") else key

    def _get_video_original_key(self, key: str) -> str:
        normalized_key = self._normalize_video_key(key)
        original_key = self.lerobot_modality_meta.video[normalized_key].original_key
        return original_key or normalized_key

    def _get_video_feature_meta(self, key: str) -> dict:
        return self.lerobot_info_meta["features"][self._get_video_original_key(key)]

    def _video_uses_image_backend(self, key: str) -> bool:
        return str(self._get_video_feature_meta(key).get("dtype", "")).lower() == "image"

    def _resolve_video_feature_channels_and_fps(self, le_video_meta: dict) -> tuple[int, float]:
        names = le_video_meta.get("names") or []
        shape = le_video_meta.get("shape") or []

        channel_dim_name = "channel" if "channel" in names else "channels" if "channels" in names else None
        if channel_dim_name is None:
            channels = 3
        else:
            channels = int(shape[names.index(channel_dim_name)])

        if str(le_video_meta.get("dtype", "")).lower() == "image":
            fps = float(self.lerobot_info_meta.get("fps", 1))
            return channels, fps

        # NOTE(FH): different lerobot dataset versions have different keys for fps metadata.
        try:
            fps = float(le_video_meta["video_info"]["video.fps"])
        except KeyError:
            fps = float(le_video_meta["info"]["video.fps"])

        return channels, fps

    def _decode_image_payload(self, image_value) -> np.ndarray:
        if isinstance(image_value, dict):
            image_bytes = image_value.get("bytes")
            image_path = image_value.get("path")
            if image_bytes:
                with Image.open(io.BytesIO(image_bytes)) as img:
                    return np.asarray(img.convert("RGB"))
            if image_path:
                image_path = Path(image_path)
                if not image_path.is_absolute():
                    image_path = self.dataset_path / image_path
                with Image.open(image_path) as img:
                    return np.asarray(img.convert("RGB"))
            raise ValueError(f"Unsupported image payload keys: {list(image_value.keys())}")

        if isinstance(image_value, (bytes, bytearray)):
            with Image.open(io.BytesIO(image_value)) as img:
                return np.asarray(img.convert("RGB"))

        if isinstance(image_value, Image.Image):
            return np.asarray(image_value.convert("RGB"))

        image_array = np.asarray(image_value)
        if image_array.ndim == 2:
            image_array = np.repeat(image_array[..., None], 3, axis=-1)
        if image_array.ndim != 3:
            raise ValueError(f"Expected image array with 3 dimensions, got shape {image_array.shape}")
        if image_array.dtype != np.uint8:
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)
        return image_array

    def _load_image_frames_from_trajectory(
        self,
        trajectory_id: int,
        key: str,
        step_indices: np.ndarray,
    ) -> np.ndarray:
        normalized_key = self._normalize_video_key(key)
        le_key = self._get_video_original_key(normalized_key)
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert le_key in self.curr_traj_data.columns, f"No {le_key} found in {trajectory_id=}"
        frames = [
            self._decode_image_payload(self.curr_traj_data[le_key].iloc[int(step_index)])
            for step_index in step_indices
        ]
        return np.stack(frames, axis=0)

    def _get_chunk_size(self) -> int:
        """Get the chunk size for the LeRobot dataset."""
        return self.lerobot_info_meta["chunks_size"]

    def _get_tasks(self) -> pd.DataFrame:
        """Get the tasks for the dataset."""
        if self._lerobot_version == "v2.0":
            tasks_path = self.dataset_path / LE_ROBOT_TASKS_FILENAME
            with open(tasks_path, "r") as f:
                tasks = [json.loads(line) for line in f]
            df = pd.DataFrame(tasks)
            return df.set_index("task_index")
        
        elif self._lerobot_version == "v3.0":
            tasks_path = self.dataset_path / LE_ROBOT3_TASKS_FILENAME
            df = pd.read_parquet(tasks_path)
            df = df.reset_index()  # 把索引变成一列，列名通常为 'index'
            df = df.rename(columns={'index': 'task'})  # 把 'index' 列重命名为 'task'
            df = df[['task_index', 'task']]  # 调整列顺序
            return df
    def _check_integrity(self):
        """Use the config to check if the keys are valid and detect silent data corruption."""
        ERROR_MSG_HEADER = f"Error occurred in initializing dataset {self.dataset_name}:\n"

        for modality_config in self.modality_configs.values():
            for key in modality_config.modality_keys:
                if key == "lapa_action" or key == "dream_actions":
                    continue  # no need for any metadata for lapa actions because it comes normalized
                # Check if the key is valid
                try:
                    self.lerobot_modality_meta.get_key_meta(key)
                except Exception as e:
                    raise ValueError(
                        ERROR_MSG_HEADER + f"Unable to find key {key} in modality metadata:\n{e}"
                    )

    def set_transforms_metadata(
        self,
        metadata: DatasetMetadata,
        original_metadata: DatasetMetadata | None = None,
    ):
        """Set the metadata for the transforms. This is useful for transforms that need to know the metadata, such as the normalization values."""
        self.transforms.set_metadata(
            metadata,
            original_metadata=original_metadata or self.metadata,
        )

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset.

        Args:
            epoch (int): The epoch to set.
        """
        self.epoch = epoch

    def __len__(self) -> int:
        """Get the total number of data points in the dataset.

        Returns:
            int: the total number of data points in the dataset.
        """
        return len(self.all_steps)

    def __str__(self) -> str:
        """Get the description of the dataset."""
        return f"{self.dataset_name} ({len(self)} steps)"


    def __getitem__(self, index: int) -> dict:
        """Get the data for a single step in a trajectory.

        Args:
            index (int): The index of the step to get.

        Returns:
            dict: The data for the step.
        """
        trajectory_id, base_index = self.all_steps[index]
        raw_data = self.get_step_data(trajectory_id, base_index)
        action_valid_mask = self._get_action_valid_mask_from_data(raw_data)
        data = self.transforms(raw_data)
        sample = self._pack_sample(data)
        if action_valid_mask is not None:
            if "action_valid_mask" in sample:
                action_valid_mask = np.logical_and(sample["action_valid_mask"].astype(bool), action_valid_mask)
            sample["action_valid_mask"] = action_valid_mask.astype(np.float32)
        return sample

    def _pack_sample(self, data: dict) -> dict:
        """Pack transformed modality data into training sample format."""
        prim_images = []
        wrist_views = []
        for video_key in self.modality_keys["video"]:
            image = data[video_key][0]
            image = Image.fromarray(image)
            if "wrist" not in video_key:
                prim_images.append(image)
            else:
                wrist_views.append(image)
        all_images = prim_images + wrist_views

        language = data[self.modality_keys["language"][0]][0]
        action_valid_mask = None
        if "action" in data:
            action = data["action"]
            if isinstance(action, torch.Tensor):
                action = action.detach().cpu().numpy()
            action = action.astype(np.float16)
        else:
            action = []
            for action_key in self.modality_keys["action"]:
                action.append(data[action_key])
            action = np.concatenate(action, axis=1).astype(np.float16)

        sample = {
            "action": action,
            "image": all_images,
            "lang": language,
            "language": language,
            "robot_type": self.robot_type,
        }
        if "action_dim_valid_mask" in data:
            action_dim_valid_mask = data["action_dim_valid_mask"]
            if isinstance(action_dim_valid_mask, torch.Tensor):
                action_dim_valid_mask = action_dim_valid_mask.detach().cpu().numpy()
            sample["action_dim_valid_mask"] = action_dim_valid_mask.astype(np.float32)
        if action_valid_mask is not None:
            sample["action_valid_mask"] = action_valid_mask.astype(np.float32)

        if self.data_cfg is not None and self.data_cfg.get("include_state", False) not in ["False", False]:
            if "state" in data:
                state = data["state"]
                if isinstance(state, torch.Tensor):
                    state = state.detach().cpu().numpy()
                state = state.astype(np.float16)
            else:
                state = []
                for state_key in self.modality_keys["state"]:
                    state.append(data[state_key])
                state = np.concatenate(state, axis=1).astype(np.float16)
            sample["state"] = state

        return sample

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        """Get the RAW data for a single step in a trajectory. No transforms are applied.

        Args:
            trajectory_id (int): The name of the trajectory.
            base_index (int): The base step index in the trajectory.

        Returns:
            dict: The RAW data for the step.

        Example return:
            {
                "video": {
                    "video.image_side_0": [B, T, H, W, C],
                    "video.image_side_1": [B, T, H, W, C],
                },
                "state": {
                    "state.eef_position": [B, T, state_dim],
                    "state.eef_rotation": [B, T, state_dim],
                },
                "action": {
                    "action.eef_position": [B, T, action_dim],
                    "action.eef_rotation": [B, T, action_dim],
                },
            }
        """
        data = {}
        # Get the data for all modalities # just for action base data
        self.curr_traj_data = self.get_trajectory_data(trajectory_id)
        # TODO @JinhuiYE The logic below is poorly implemented. Data reading should be directly based on curr_traj_data.
        for modality in self.modality_keys:
            # Get the data corresponding to each key in the modality
            for key in self.modality_keys[modality]:
                data[key] = self.get_data_by_modality(trajectory_id, modality, key, base_index)
        data = self._apply_action_mode(data)
        return data

    def get_trajectory_data(self, trajectory_id: int) -> pd.DataFrame:
        """Get the data for a trajectory."""
        if self._lerobot_version == "v2.0":
        
            if self.curr_traj_id == trajectory_id and self.curr_traj_data is not None:
                return self.curr_traj_data
            else:
                chunk_index = self.get_episode_chunk(trajectory_id)
                parquet_path = self.dataset_path / self.data_path_pattern.format(
                    episode_chunk=chunk_index, episode_index=trajectory_id
                )
                assert parquet_path.exists(), f"Parquet file not found at {parquet_path}"
                return pd.read_parquet(parquet_path)
        elif self._lerobot_version == "v3.0":
            return self.get_trajectory_data_lerobot_v3(trajectory_id)
    
    def get_trajectory_data_lerobot_v3(self, trajectory_id: int) -> pd.DataFrame:
        """Get the data for a trajectory from lerobot v3."""
        if self.curr_traj_id == trajectory_id and self.curr_traj_data is not None:
            return self.curr_traj_data
        else: #TODO check detail later
            episode_meta = self.trajectory_ids_to_metadata[trajectory_id]
            chunk_index = episode_meta["data/chunk_index"]
            file_index = self.get_episode_file_index(trajectory_id)
            # file_from_index = self.get_episode_file_from_index(trajectory_id)
            
            
            parquet_path = self.dataset_path / self.data_path_pattern.format(
                chunk_index=chunk_index, file_index=file_index
            )
            assert parquet_path.exists(), f"Parquet file not found at {parquet_path}"
            file_data = pd.read_parquet(parquet_path)
            
            # filter by trajectory_id
            episode_data = file_data.loc[file_data["episode_index"] == trajectory_id].copy()
            return episode_data


    def get_trajectory_index(self, trajectory_id: int) -> int:
        """Get the index of the trajectory in the dataset by the trajectory ID.
        This is useful when you need to get the trajectory length or sampling weight corresponding to the trajectory ID.

        Args:
            trajectory_id (str): The ID of the trajectory.

        Returns:
            int: The index of the trajectory in the dataset.
        """
        trajectory_indices = np.where(self.trajectory_ids == trajectory_id)[0]
        if len(trajectory_indices) != 1:
            raise ValueError(
                f"Error finding trajectory index for {trajectory_id}, found {trajectory_indices=}"
            )
        return trajectory_indices[0]

    def get_episode_chunk(self, ep_index: int) -> int:
        """Get the chunk index for an episode index."""
        return ep_index // self.chunk_size
    def get_episode_file_index(self, ep_index: int) -> int:
        """Get the file index for an episode index."""
        episode_meta = self.trajectory_ids_to_metadata[ep_index]
        return episode_meta["data/file_index"]
    
    def get_episode_file_from_index(self, ep_index: int) -> int:
        """Get the file from index for an episode index."""
        episode_meta = self.trajectory_ids_to_metadata[ep_index]
        return episode_meta["data/file_from_index"]


    def retrieve_data_and_pad(
        self,
        array: np.ndarray,
        step_indices: np.ndarray,
        max_length: int,
        padding_strategy: str = "first_last",
    ) -> np.ndarray:
        """Retrieve the data from the dataset and pad it if necessary.
        Args:
            array (np.ndarray): The array to retrieve the data from.
            step_indices (np.ndarray): The step indices to retrieve the data for.
            max_length (int): The maximum length of the data.
            padding_strategy (str): The padding strategy, either "first" or "last".
        """
        # Get the padding indices
        front_padding_indices = step_indices < 0
        end_padding_indices = step_indices >= max_length
        padding_positions = np.logical_or(front_padding_indices, end_padding_indices)
        # Retrieve the data with the non-padding indices
        # If there exists some padding, Given T step_indices, the shape of the retrieved data will be (T', ...) where T' < T
        raw_data = array[step_indices[~padding_positions]]
        assert isinstance(raw_data, np.ndarray), f"{type(raw_data)=}"
        # This is the shape of the output, (T, ...)
        if raw_data.ndim == 1:
            expected_shape = (len(step_indices),)
        else:
            expected_shape = (len(step_indices), *array.shape[1:])

        # Pad the data
        output = np.zeros(expected_shape)
        # Assign the non-padded data
        output[~padding_positions] = raw_data
        # If there exists some padding, pad the data
        if padding_positions.any():
            if padding_strategy == "first_last":
                # Use first / last step data to pad
                front_padding_data = array[0]
                end_padding_data = array[-1]
                output[front_padding_indices] = front_padding_data
                output[end_padding_indices] = end_padding_data
            elif padding_strategy == "zero":
                # Use zero padding
                output[padding_positions] = 0
            else:
                raise ValueError(f"Invalid padding strategy: {padding_strategy}")
        return output

    def get_video_path(self, trajectory_id: int, key: str) -> Path:
        chunk_index = self.get_episode_chunk(trajectory_id)
        original_key = self.lerobot_modality_meta.video[key].original_key
        if original_key is None:
            original_key = key
        if self._lerobot_version == "v2.0":
            video_filename = self.video_path_pattern.format(
                episode_chunk=chunk_index, episode_index=trajectory_id, video_key=original_key
            )
        elif self._lerobot_version == "v3.0":
            episode_meta = self.trajectory_ids_to_metadata[trajectory_id]
            video_filename = self.video_path_pattern.format(
                video_key=original_key,
                chunk_index=episode_meta["data/chunk_index"],
                file_index=episode_meta["data/file_index"],
            )
        return self.dataset_path / video_filename

    def get_video(
        self,
        trajectory_id: int,
        key: str,
        base_index: int,
    ) -> np.ndarray:
        """Get the video frames for a trajectory by a base index.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (str): The ID of the trajectory.
            key (str): The key of the video.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The video frames for the trajectory and frame indices. Shape: (T, H, W, C)
        """
        # Get the step indices
        step_indices = self.delta_indices[key] + base_index
        # print(f"{step_indices=}")
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Ensure the indices are within the valid range
        # This is equivalent to padding the video with extra frames at the beginning and end
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, self.trajectory_lengths[trajectory_index] - 1)
        assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
        if self._video_uses_image_backend(key):
            return self._load_image_frames_from_trajectory(trajectory_id, key, step_indices)
        # Get the sub-key
        key = key.replace("video.", "")
        video_path = self.get_video_path(trajectory_id, key)
        # Get the action/state timestamps for each frame in the video
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert "timestamp" in self.curr_traj_data.columns, f"No timestamp found in {trajectory_id=}"
        timestamp: np.ndarray = self.curr_traj_data["timestamp"].to_numpy()
        # Get the corresponding video timestamps from the step indices
        video_timestamp = timestamp[step_indices]
        if self._lerobot_version == "v3.0":
            episode_meta = self.trajectory_ids_to_metadata.get(trajectory_id, {})
            from_timestamps = episode_meta.get("videos/from_timestamps", {})
            original_video_key = self.lerobot_modality_meta.video[key].original_key
            if original_video_key is None:
                original_video_key = key
            from_timestamp = float(from_timestamps.get(original_video_key, 0.0))
            video_timestamp = video_timestamp + from_timestamp

        return get_frames_by_timestamps(
            video_path.as_posix(),
            video_timestamp,
            video_backend=self.video_backend, # TODO
            video_backend_kwargs=self.video_backend_kwargs,
        )

    def get_state_or_action(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        base_index: int,
    ) -> np.ndarray:
        """Get the state or action data for a trajectory by a base index.
        If the step indices are out of range, pad with the data:
            if the data is stored in absolute format, pad with the first or last step data;
            otherwise, pad with zero.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            modality (str): The modality of the data.
            key (str): The key of the data.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The data for the trajectory and step indices.
        """
        # Get the step indices
        step_indices = self.delta_indices[key] + base_index
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Get the maximum length of the trajectory
        max_length = self.trajectory_lengths[trajectory_index]
        assert key.startswith(modality + "."), f"{key} must start with {modality + '.'}, got {key}"
        # Get the sub-key, e.g. state.joint_angles -> joint_angles
        key = key.replace(modality + ".", "")
        # Get the lerobot key
        le_state_or_action_cfg = getattr(self.lerobot_modality_meta, modality)
        le_key = le_state_or_action_cfg[key].original_key
        if le_key is None:
            le_key = key
        # Get the data array, shape: (T, D)
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert le_key in self.curr_traj_data.columns, f"No {le_key} found in {trajectory_id=}"
        data_array: np.ndarray = np.stack(self.curr_traj_data[le_key])  # type: ignore
        if data_array.ndim == 1:
            # Scalar parquet columns (e.g. gripper position) should behave like 1D features.
            data_array = data_array[:, None]
        assert data_array.ndim == 2, f"Expected 2D array, got key {le_key} is{data_array.shape} array"
        le_indices = np.arange(
            le_state_or_action_cfg[key].start,
            le_state_or_action_cfg[key].end,
        )
        data_array = data_array[:, le_indices]
        # Get the state or action configuration
        state_or_action_cfg = getattr(self.metadata.modalities, modality)[key]

        # Pad the data
        return self.retrieve_data_and_pad(
            array=data_array,
            step_indices=step_indices,
            max_length=max_length,
            padding_strategy="first_last" if state_or_action_cfg.absolute else "zero",
            # padding_strategy="zero",           # HACK for realdata
        )

    def get_language(
        self,
        trajectory_id: int,
        key: str,
        base_index: int,
    ) -> list[str]:
        """Get the language annotation data for a trajectory by step indices.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            key (str): The key of the annotation.
            base_index (int): The base index of the trajectory.

        Returns:
            list[str]: The annotation data for the trajectory and step indices. If no matching data is found, return empty strings.
        """
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        # Get the step indices
        step_indices = self.delta_indices[key] + base_index
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Get the maximum length of the trajectory
        max_length = self.trajectory_lengths[trajectory_index]
        # Get the end times corresponding to the closest indices
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, max_length - 1)
        # Get the annotations
        task_indices: list[int] = []
        assert key.startswith(
            "annotation."
        ), f"Language key must start with 'annotation.', got {key}"
        subkey = key.replace("annotation.", "")
        annotation_meta = self.lerobot_modality_meta.annotation
        assert annotation_meta is not None, f"Annotation metadata is None for {subkey}"
        assert (
            subkey in annotation_meta
        ), f"Annotation key {subkey} not found in metadata, available annotation keys: {annotation_meta.keys()}"
        subkey_meta = annotation_meta[subkey]
        original_key = subkey_meta.original_key
        if original_key is None:
            original_key = key
        language_annotations: list[str] = []
        for i in range(len(step_indices)):
            value = self.curr_traj_data[original_key].iloc[step_indices[i]]
            if hasattr(value, "item") and not isinstance(value, (str, bytes)):
                try:
                    value = value.item()
                except ValueError:
                    pass

            if pd.isna(value):
                language_annotations.append("")
                continue

            if isinstance(value, (int, float, np.integer, np.floating)):
                task_value = self.tasks.loc[int(value)]["task"]
                if isinstance(task_value, pd.Series):
                    task_value = task_value.iloc[0]
                language_annotations.append(str(task_value))
                continue

            language_annotations.append(str(value))

        return language_annotations

    def get_data_by_modality(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        base_index: int,
    ):
        """Get the data corresponding to the modality for a trajectory by a base index.
        This method will call the corresponding helper method based on the modality.
        See the helper methods for more details.
        NOTE: For the language modality, the data is padded with empty strings if no matching data is found.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            modality (str): The modality of the data.
            key (str): The key of the data.
            base_index (int): The base index of the trajectory.
        """
        if modality == "video":
            return self.get_video(trajectory_id, key, base_index)
        elif modality == "state" or modality == "action":
            return self.get_state_or_action(trajectory_id, modality, key, base_index)
        elif modality == "language":
            return self.get_language(trajectory_id, key, base_index)
        else:
            raise ValueError(f"Invalid modality: {modality}")

    def _save_dataset_statistics_(self, save_path: Path | str, format: str = "json") -> None:
        """
        Save dataset statistics to specified path in the required format.
        Only includes statistics for keys that are actually used in the dataset.
        Gripper-related keys will be placed at the end.
        
        Args:
            save_path (Path | str): Path to save the statistics file
            format (str): Save format, currently only supports "json"
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build the data structure to save
        statistics_data = {}
        
        # Get used modality keys
        used_action_keys, used_state_keys = get_used_modality_keys(self.modality_keys)
        
        # Organize statistics by tag
        tag = self.tag
        tag_stats = {}
        
        # Process action statistics (only for used keys)
        if hasattr(self.metadata.statistics, 'action') and self.metadata.statistics.action:
            action_stats = self.metadata.statistics.action
            
            # Filter to only include used action keys and reorder: non-gripper first, gripper last
            non_gripper_keys = []
            gripper_keys = []
            
            for key in action_stats.keys():
                if key in used_action_keys:
                    if "gripper" in key.lower():
                        gripper_keys.append(key)
                    else:
                        non_gripper_keys.append(key)
            
            # Reorder: non-gripper first, gripper last
            reordered_keys = non_gripper_keys + gripper_keys
            
            filtered_action_stats = {}
            for key in reordered_keys:
                filtered_action_stats[key] = action_stats[key]
            
            if filtered_action_stats:
                # Combine statistics from filtered action sub-keys
                combined_action_stats = combine_modality_stats(filtered_action_stats)
                
                # Add mask field based on whether it's gripper or not
                mask = generate_action_mask_for_used_keys(
                    self.metadata.modalities.action, filtered_action_stats.keys()
                )
                combined_action_stats["mask"] = mask
                
                tag_stats["action"] = combined_action_stats
        
        # Process state statistics (only for used keys)
        if hasattr(self.metadata.statistics, 'state') and self.metadata.statistics.state:
            state_stats = self.metadata.statistics.state
            
            # Filter to only include used state keys, optionally reorder gripper to end
            non_gripper_keys = []
            gripper_keys = []
            
            for key in state_stats.keys():
                if key in used_state_keys:
                    if "gripper" in key.lower():
                        gripper_keys.append(key)
                    else:
                        non_gripper_keys.append(key)
            
            # Reorder: non-gripper first, gripper last
            reordered_keys = non_gripper_keys + gripper_keys
            
            filtered_state_stats = {}
            for key in reordered_keys:
                filtered_state_stats[key] = state_stats[key]
            
            if filtered_state_stats:
                combined_state_stats = combine_modality_stats(filtered_state_stats)
                tag_stats["state"] = combined_state_stats
        
        # Add dataset counts
        tag_stats["num_transitions"] = len(self)
        tag_stats["num_trajectories"] = len(self.trajectory_ids)
        
        statistics_data[tag] = tag_stats
        
        # Save as JSON file
        if format.lower() == "json":
            if not str(save_path).endswith('.json'):
                save_path = save_path.with_suffix('.json')
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(statistics_data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}. Currently only 'json' is supported.")
        
        print(f"Single dataset statistics saved to: {save_path}")
        print(f"Used action keys (reordered): {list(used_action_keys)}")
        print(f"Used state keys (reordered): {list(used_state_keys)}")


class CachedLeRobotSingleDataset(LeRobotSingleDataset):
    def __init__(self, img_resize: tuple[int, int] | None = None, *args, **kwargs):
        """
        This class caches the video frames for each trajectory and key.
        It is recommended to use this class if the video frames need to be accessed multiple times.

        Args:
            resize_img (tuple[int, int], optional): The size to resize the video frames to reduce memory usage.
        """
        # Convert img_resize to tuple if it is not already
        if img_resize is not None and not isinstance(img_resize, tuple):
            img_resize = tuple(img_resize)
            assert len(img_resize) == 2, f"Expected tuple of length 2, got {img_resize}"
        self.img_resize = img_resize

        # Initialize img_resize attribute first to ensure it exists
        super().__init__(*args, **kwargs)
        cached_frames: dict[str, np.ndarray] = {}

        for key in self.modality_keys["video"]:
            all_frames = []
            original_key = key
            key = key.replace("video.", "")
            for trajectory_id, trajectory_length in tqdm(
                zip(self.trajectory_ids, self.trajectory_lengths),
                total=len(self.trajectory_ids),
                desc=f"Caching {key} frames",
            ):
                if self._video_uses_image_backend(key):
                    self.curr_traj_id = trajectory_id
                    self.curr_traj_data = self.get_trajectory_data(trajectory_id)
                    frames = self._load_image_frames_from_trajectory(
                        trajectory_id,
                        key,
                        np.arange(trajectory_length, dtype=int),
                    )
                    if img_resize is not None:
                        frames = np.stack(
                            [
                                np.asarray(
                                    Image.fromarray(frame).resize(tuple(img_resize), resample=Image.BILINEAR)
                                )
                                for frame in frames
                            ],
                            axis=0,
                        )
                else:
                    video_path = self.get_video_path(trajectory_id, key)
                    frames = get_all_frames(
                        video_path.as_posix(),
                        video_backend=self.video_backend,
                        video_backend_kwargs=self.video_backend_kwargs,
                        resize_size=img_resize,
                    )
                assert frames.ndim == 4, f"Expected 4D array, got {frames.shape} array"
                assert frames.shape[3] == 3, f"Expected 3 channels, got {frames.shape[3]} channels"
                
                # Apply image cropping if enabled and the video key is base_view
                # Note: crop_obs_camera functionality has been removed
                
                # assert (
                #     frames.shape[0] == trajectory_length
                # ), f"Expected {trajectory_length} frames, got {frames.shape[0]} frames"
                all_frames.append(frames)
            cached_frames[key] = np.concatenate(all_frames, axis=0)
            print(f"{key}: {cached_frames[key].shape}")
        self.cached_frames = cached_frames
        self.start_indices = np.cumsum(self.trajectory_lengths) - self.trajectory_lengths

    def get_video(self, trajectory_id: int, key: str, base_index: int) -> np.ndarray:
        step_indices = self.delta_indices[key] + base_index
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Ensure the indices are within the valid range
        # This is equivalent to padding the video with extra frames at the beginning and end
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, self.trajectory_lengths[trajectory_index] - 1)
        assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
        # Get the sub-key
        key = key.replace("video.", "")
        # Calculate the absolute indices
        absolute_indices = self.start_indices[trajectory_index] + step_indices
        return self.cached_frames[key][absolute_indices]

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        """Get the RAW data for a single step. No transforms are applied.

        Args:
            trajectory_id (str): The ID of the trajectory.
            base_index (int): The base index of the step.

        Returns:
            dict: The data for the step.
        """
        data = {}
        self.curr_traj_data = self.get_trajectory_data(trajectory_id)
        # Get the data for all modalities
        for modality in self.modality_keys:
            # Get the data corresponding to each key in the modality
            for key in self.modality_keys[modality]:
                data[key] = self.get_data_by_modality(trajectory_id, modality, key, base_index)
        return data

    def set_transforms_metadata(
        self,
        metadata: DatasetMetadata,
        original_metadata: DatasetMetadata | None = None,
    ):
        """Set the metadata for the transforms. This is useful for transforms that need to know the metadata, such as the normalization values."""
        if self.img_resize is not None:
            all_video_keys = [key for key in self.modality_keys["video"]]
            for key in metadata.modalities.video:
                if key in all_video_keys:
                    metadata.modalities.video[key].resolution = self.img_resize
        super().set_transforms_metadata(metadata, original_metadata=original_metadata)


def safe_hash(input_tuple):
    # keep 128 bits of the hash
    tuple_string = repr(input_tuple).encode("utf-8")
    sha256 = hashlib.sha256()
    sha256.update(tuple_string)

    seed = int(sha256.hexdigest(), 16)

    return seed & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF


class MixtureSpecElement(BaseModel):
    dataset_path: list[Path] | Path = Field(..., description="The path to the dataset.")
    dataset_weight: float = Field(..., description="The weight of the dataset in the mixture.")
    distribute_weights: bool = Field(
        default=False,
        description="Whether to distribute the weights of the dataset across all the paths. If True, the weights will be evenly distributed across all the paths.",
    )


# Helper functions for dataset statistics

def combine_modality_stats(modality_stats: dict) -> dict:
    """
    Combine statistics from all sub-keys under a modality.
    
    Args:
        modality_stats (dict): Statistics for a modality, containing multiple sub-keys.
                               Each sub-key contains DatasetStatisticalValues object.
        
    Returns:
        dict: Combined statistics
    """
    combined_stats = {
        "mean": [],
        "std": [],
        "max": [],
        "min": [],
        "q01": [],
        "q99": []
    }
    
    # Combine statistics in sub-key order
    for subkey in modality_stats.keys():
        subkey_stats = modality_stats[subkey]  # This is a DatasetStatisticalValues object
        
        # Convert DatasetStatisticalValues to dict-like access
        for stat_name in ["mean", "std", "max", "min", "q01", "q99"]:
            stat_value = getattr(subkey_stats, stat_name)
            if isinstance(stat_value, (list, tuple)):
                combined_stats[stat_name].extend(stat_value)
            else:
                # Handle NDArray case - convert to list
                if hasattr(stat_value, 'tolist'):
                    combined_stats[stat_name].extend(stat_value.tolist())
                else:
                    combined_stats[stat_name].append(float(stat_value))
    
    return combined_stats

def generate_action_mask_for_used_keys(action_modalities: dict, used_action_keys_ordered) -> list[bool]:
    """
    Generate mask based on action modalities, but only for used keys.
    Gripper-related are False, others are True.
    
    Args:
        action_modalities (dict): Configuration information for action modalities.
        used_action_keys_ordered: Iterable of actually used action keys in the correct order.
        
    Returns:
        list[bool]: List of mask values
    """
    mask = []
    
    # Generate mask in the same order as the statistics were combined
    for subkey in used_action_keys_ordered:
        if subkey in action_modalities:
            subkey_config = action_modalities[subkey]
            
            # Get dimension count from shape
            if hasattr(subkey_config, 'shape') and len(subkey_config.shape) > 0:
                dim_count = subkey_config.shape[0]
            else:
                dim_count = 1
            
            # Check if it's gripper-related
            is_gripper = "gripper" in subkey.lower()
            
            # Generate mask value for each dimension
            for _ in range(dim_count):
                mask.append(not is_gripper)  # gripper is False, others are True
    
    return mask

def get_used_modality_keys(modality_keys: dict) -> tuple[list, list]:
    """Extract used action and state keys from modality configuration."""
    used_action_keys = []
    used_state_keys = []
    
    # Extract action keys (remove "action." prefix)
    for action_key in modality_keys.get("action", []):
        if action_key.startswith("action."):
            clean_key = action_key.replace("action.", "")
            used_action_keys.append(clean_key)
    
    # Extract state keys (remove "state." prefix)  
    for state_key in modality_keys.get("state", []):
        if state_key.startswith("state."):
            clean_key = state_key.replace("state.", "")
            used_state_keys.append(clean_key)
    
    return used_action_keys, used_state_keys

class LeRobotMixtureDataset(Dataset):
    """
    A mixture of multiple datasets. This class samples a single dataset based on the dataset weights and then calls the `__getitem__` method of the sampled dataset.
    It is recommended to modify the single dataset class instead of this class.
    """

    def __init__(
        self,
        data_mixture: Sequence[tuple[LeRobotSingleDataset, float]],
        mode: str,
        balance_dataset_weights: bool = False,
        balance_trajectory_weights: bool = True,
        seed: int = 42,
        metadata_config: dict = {
            "percentile_mixing_method": "min_max",
        },
        **kwargs,
    ):
        """
        Initialize the mixture dataset.

        Args:
            data_mixture (list[tuple[LeRobotSingleDataset, float]]): Datasets and their corresponding weights.
            mode (str): If "train", __getitem__ will return different samples every epoch; if "val" or "test", __getitem__ will return the same sample every epoch.
            balance_dataset_weights (bool): If True, the weight of dataset will be multiplied by the total trajectory length of each dataset.
            balance_trajectory_weights (bool): If True, sample trajectories within a dataset weighted by their length; otherwise, use equal weighting.
            seed (int): Random seed for sampling.
        """
        datasets: list[LeRobotSingleDataset] = []
        dataset_sampling_weights: list[float] = []
        for dataset, weight in data_mixture:
            # Check if dataset is valid and has data
            if len(dataset) == 0:
                print(f"Warning: Skipping empty dataset {dataset.dataset_name}")
                continue
            datasets.append(dataset)
            dataset_sampling_weights.append(weight)
        
        if len(datasets) == 0:
            raise ValueError("No valid datasets found in the mixture. All datasets are empty.")
        
        self.datasets = datasets
        self.balance_dataset_weights = balance_dataset_weights
        self.balance_trajectory_weights = balance_trajectory_weights
        self.seed = seed
        self.mode = mode
        self.data_cfg = kwargs["data_cfg"] if "data_cfg" in kwargs else None

        # Set properties for sampling

        # 1. Dataset lengths
        self._dataset_lengths = np.array([len(dataset) for dataset in self.datasets])
        print(f"Dataset lengths: {self._dataset_lengths}")

        # 2. Dataset sampling weights
        self._dataset_sampling_weights = np.array(dataset_sampling_weights)
        
        if self.balance_dataset_weights:
            self._dataset_sampling_weights *= self._dataset_lengths
        
        # Check for zero or negative weights before normalization
        if np.any(self._dataset_sampling_weights <= 0):
            print(f"Warning: Found zero or negative sampling weights: {self._dataset_sampling_weights}")
            # Set minimum weight to prevent division issues
            self._dataset_sampling_weights = np.maximum(self._dataset_sampling_weights, 1e-8)
        
        # Normalize weights
        weights_sum = self._dataset_sampling_weights.sum()
        if weights_sum == 0 or np.isnan(weights_sum):
            print(f"Error: Invalid weights sum: {weights_sum}")
            # Fallback to equal weights
            self._dataset_sampling_weights = np.ones(len(self.datasets)) / len(self.datasets)
            print(f"Fallback to equal weights")
        else:
            self._dataset_sampling_weights /= weights_sum

        # 3. Trajectory sampling weights
        self._trajectory_sampling_weights: list[np.ndarray] = []
        for i, dataset in enumerate(self.datasets):
            trajectory_sampling_weights = np.ones(len(dataset.trajectory_lengths))
            if self.balance_trajectory_weights:
                trajectory_sampling_weights *= dataset.trajectory_lengths
            
            # Check for zero or negative weights before normalization
            if np.any(trajectory_sampling_weights <= 0):
                print(f"Warning: Dataset {i} has zero or negative trajectory weights")
                trajectory_sampling_weights = np.maximum(trajectory_sampling_weights, 1e-8)
            
            # Normalize weights
            weights_sum = trajectory_sampling_weights.sum()
            if weights_sum == 0 or np.isnan(weights_sum):
                print(f"Error: Dataset {i} has invalid trajectory weights sum: {weights_sum}")
                # Fallback to equal weights
                trajectory_sampling_weights = np.ones(len(dataset.trajectory_lengths)) / len(dataset.trajectory_lengths)
            else:
                trajectory_sampling_weights /= weights_sum
            
            self._trajectory_sampling_weights.append(trajectory_sampling_weights)

        # 4. Primary dataset indices
        self._primary_dataset_indices = np.array(dataset_sampling_weights) == 1.0
        if not np.any(self._primary_dataset_indices):
            print(f"Warning: No dataset with weight 1.0 found. Original weights: {dataset_sampling_weights}")
            # Fallback: use the dataset(s) with maximum weight as primary
            max_weight = max(dataset_sampling_weights)
            self._primary_dataset_indices = np.array(dataset_sampling_weights) == max_weight
            print(f"Using datasets with maximum weight {max_weight} as primary: {self._primary_dataset_indices}")
            
        if not np.any(self._primary_dataset_indices):
            # This should never happen, but just in case
            print("Error: Still no primary dataset found. Using first dataset as primary.")
            self._primary_dataset_indices = np.zeros(len(self.datasets), dtype=bool)
            self._primary_dataset_indices[0] = True

        # Set the epoch and sample the first epoch
        self.set_epoch(0)

        self._sequential_step_sampling = True
        if self.data_cfg is not None:
            seq_cfg = self.data_cfg.get("sequential_step_sampling", True)
            self._sequential_step_sampling = seq_cfg not in ["False", False]

        self._step_order: list[np.ndarray] = []
        self._step_pos: list[int] = []
        if self._sequential_step_sampling:
            for dataset in self.datasets:
                self._step_order.append(np.arange(len(dataset.all_steps)))
                if self.mode == "train":
                    rng = np.random.default_rng(self.seed)
                    rng.shuffle(self._step_order[-1])
                self._step_pos.append(0)

        self.update_metadata(metadata_config, cached_statistics_path=kwargs.get("cached_statistics_path"))

    @property
    def dataset_lengths(self) -> np.ndarray:
        """The lengths of each dataset."""
        return self._dataset_lengths

    @property
    def dataset_sampling_weights(self) -> np.ndarray:
        """The sampling weights for each dataset."""
        return self._dataset_sampling_weights

    @property
    def trajectory_sampling_weights(self) -> list[np.ndarray]:
        """The sampling weights for each trajectory in each dataset."""
        return self._trajectory_sampling_weights

    @property
    def primary_dataset_indices(self) -> np.ndarray:
        """The indices of the primary datasets."""
        return self._primary_dataset_indices

    def __str__(self) -> str:
        dataset_descriptions = []
        for dataset, weight in zip(self.datasets, self.dataset_sampling_weights):
            dataset_description = {
                "Dataset": str(dataset),
                "Sampling weight": float(weight),
            }
            dataset_descriptions.append(dataset_description)
        return json.dumps({"Mixture dataset": dataset_descriptions}, indent=2)

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset.

        Args:
            epoch (int): The epoch to set.
        """
        self.epoch = epoch
        # self.sampled_steps = self.sample_epoch()

    def sample_step(self, index: int) -> tuple[LeRobotSingleDataset, int, int]:
        """Sample a single step from the dataset."""
        # return self.sampled_steps[index]

        # Set seed
        seed = index if self.mode != "train" else safe_hash((self.epoch, index, self.seed))
        rng = np.random.default_rng(seed)

        # Sample dataset
        dataset_index = rng.choice(len(self.datasets), p=self.dataset_sampling_weights)
        dataset = self.datasets[dataset_index]

        # Sample trajectory
        # trajectory_index = rng.choice(
        #     len(dataset.trajectory_ids), p=self.trajectory_sampling_weights[dataset_index]
        # )
        # trajectory_id = dataset.trajectory_ids[trajectory_index]

        # # Sample step
        # base_index = rng.choice(dataset.trajectory_lengths[trajectory_index])
        # return dataset, trajectory_id, base_index
        if len(dataset.all_steps) == 0:
            raise ValueError(f"Dataset {dataset.dataset_name} has no steps.")

        if not self._sequential_step_sampling:
            single_step_index = rng.choice(len(dataset.all_steps))
        else:
            step_pos = self._step_pos[dataset_index]
            if step_pos >= len(dataset.all_steps):
                order = np.arange(len(dataset.all_steps))
                if self.mode == "train":
                    seed = safe_hash((self.epoch, dataset_index, self.seed, step_pos))
                    rng = np.random.default_rng(seed)
                    rng.shuffle(order)
                self._step_order[dataset_index] = order
                step_pos = 0

            single_step_index = self._step_order[dataset_index][step_pos]
            self._step_pos[dataset_index] = step_pos + 1
        trajectory_id, base_index = dataset.all_steps[single_step_index]
        return dataset, trajectory_id, base_index

    _getitem_count = 0

    def __getitem__(self, index: int) -> dict:
        """Get the data for a single trajectory and start index.

        Args:
            index (int): The index of the trajectory to get.

        Returns:
            dict: The data for the trajectory and start index.
        """
        LeRobotMixtureDataset._getitem_count += 1
        if LeRobotMixtureDataset._getitem_count % 1000 == 0:
            gc.collect()

        max_retries = 10
        last_exception = None

        for attempt in range(max_retries):
            try:
                while True: # @DUG
                    dataset, trajectory_id, step = self.sample_step(index)
                    key = dataset.modality_keys["video"][0].replace("video.", "")
                    if dataset._video_uses_image_backend(key):
                        break
                    video_path = dataset.get_video_path(trajectory_id, key)
                    if os.path.exists(video_path):
                        break
                    index = random.randint(0, len(self) - 1)
                    
                raw_data = dataset.get_step_data(trajectory_id, step)
                action_valid_mask = dataset._get_action_valid_mask_from_data(raw_data)
                data = dataset.transforms(raw_data)
                sample = dataset._pack_sample(data)
                if action_valid_mask is not None:
                    if "action_valid_mask" in sample:
                        action_valid_mask = np.logical_and(
                            sample["action_valid_mask"].astype(bool),
                            action_valid_mask,
                        )
                    sample["action_valid_mask"] = action_valid_mask.astype(np.float32)

                if self._should_prefix_control_frequency() or self._should_prefix_embodiment():
                    instruction = self._format_instruction_with_prompt_prefixes(sample["lang"], dataset)
                    sample["lang"] = instruction
                    sample["language"] = instruction
                sample["subset_path"] = str(dataset.dataset_path)
                sample["dataset_name"] = dataset.dataset_name
                sample["trajectory_id"] = int(trajectory_id)
                sample["step_index"] = int(step)
                sample["robot_tag"] = dataset.tag
                sample["robot_type"] = dataset.robot_type
                return sample
                
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    # Log the error but continue trying
                    print(f"Attempt {attempt + 1}/{max_retries} failed for index {index}: {e}")
                    print(f"Retrying with new sample...")
                    # For retry, we can use a slightly different index to get a new sample
                    # This helps avoid getting stuck on the same problematic sample
                    index = random.randint(0, len(self) - 1)
                else:
                    # All retries exhausted
                    print(f"All {max_retries} attempts failed for index {index}")
                    print(f"Last error: {last_exception}")
                    # Return a dummy sample or re-raise the exception
                    raise last_exception

    def __len__(self) -> int:
        """Get the length of a single epoch in the mixture.

        Returns:
            int: The length of a single epoch in the mixture.
        """
        # Check for potential issues
        if len(self.datasets) == 0:
            return 0
            
        # Check if any dataset lengths are 0 or NaN
        if np.any(self.dataset_lengths == 0) or np.any(np.isnan(self.dataset_lengths)):
            print(f"Warning: Found zero or NaN dataset lengths: {self.dataset_lengths}")
            # Filter out zero/NaN length datasets
            valid_indices = (self.dataset_lengths > 0) & (~np.isnan(self.dataset_lengths))
            if not np.any(valid_indices):
                print("Error: All datasets have zero or NaN length")
                return 0
        else:
            valid_indices = np.ones(len(self.datasets), dtype=bool)
        
        # Check if any sampling weights are 0 or NaN
        if np.any(self.dataset_sampling_weights == 0) or np.any(np.isnan(self.dataset_sampling_weights)):
            print(f"Warning: Found zero or NaN sampling weights: {self.dataset_sampling_weights}")
            # Use only valid weights
            valid_weights = (self.dataset_sampling_weights > 0) & (~np.isnan(self.dataset_sampling_weights))
            valid_indices = valid_indices & valid_weights
            if not np.any(valid_indices):
                print("Error: All sampling weights are zero or NaN")
                return 0
        
        # Check primary dataset indices
        primary_and_valid = self.primary_dataset_indices & valid_indices
        if not np.any(primary_and_valid):
            print(f"Warning: No valid primary datasets found. Primary indices: {self.primary_dataset_indices}, Valid indices: {valid_indices}")
            # Fallback: use the largest valid dataset
            if np.any(valid_indices):
                max_length = self.dataset_lengths[valid_indices].max()
                print(f"Fallback: Using maximum dataset length: {max_length}")
                return int(max_length)
            else:
                return 0
        
        # Calculate the ratio and get max
        ratios = (self.dataset_lengths / self.dataset_sampling_weights)[primary_and_valid]
        
        # Check for NaN or inf in ratios
        if np.any(np.isnan(ratios)) or np.any(np.isinf(ratios)):
            print(f"Warning: Found NaN or inf in ratios: {ratios}")
            print(f"Dataset lengths: {self.dataset_lengths[primary_and_valid]}")
            print(f"Sampling weights: {self.dataset_sampling_weights[primary_and_valid]}")
            # Filter out invalid ratios
            valid_ratios = ratios[~np.isnan(ratios) & ~np.isinf(ratios)]
            if len(valid_ratios) == 0:
                print("Error: All ratios are NaN or inf")
                return 0
            max_ratio = valid_ratios.max()
        else:
            max_ratio = ratios.max()
        
        result = int(max_ratio)
        if result == 0:
            print(f"Warning: Dataset mixture length is 0")
        return result

    @staticmethod
    def compute_overall_statistics(
        per_task_stats: list[dict[str, dict[str, list[float] | np.ndarray]]],
        dataset_sampling_weights: list[float] | np.ndarray,
        percentile_mixing_method: str = "weighted_average",
    ) -> dict[str, dict[str, list[float]]]:
        """
        Computes overall statistics from per-task statistics using dataset sample weights.

        Args:
            per_task_stats: List of per-task statistics.
            Example format of one element in the per-task statistics list:
                {
                    "state.gripper": {
                        "min": [...],
                        "max": [...],
                        "mean": [...],
                        "std": [...],
                        "q01": [...],
                        "q99": [...],
                    },
                    ...
                }
            dataset_sampling_weights: List of sample weights for each task.
            percentile_mixing_method: The method to mix the percentiles, either "weighted_average" or "weighted_std".

        Returns:
            A dict of overall statistics per modality.
        """
        # Normalize the sample weights to sum to 1
        dataset_sampling_weights = np.array(dataset_sampling_weights)
        normalized_weights = dataset_sampling_weights / dataset_sampling_weights.sum()

        # Initialize overall statistics dict
        overall_stats: dict[str, dict[str, list[float]]] = {}

        # Get the list of modality keys
        modality_keys = per_task_stats[0].keys()

        for modality in modality_keys:
            # Number of dimensions (assuming consistent across tasks)
            num_dims = len(per_task_stats[0][modality]["mean"])

            # Initialize accumulators for means and variances
            weighted_means = np.zeros(num_dims)
            weighted_squares = np.zeros(num_dims)

            # Collect min, max, q01, q99 from all tasks
            min_list = []
            max_list = []
            q01_list = []
            q99_list = []

            for task_idx, task_stats in enumerate(per_task_stats):
                w_i = normalized_weights[task_idx]
                stats = task_stats[modality]
                means = np.array(stats["mean"])
                stds = np.array(stats["std"])

                # Update weighted sums for mean and variance
                weighted_means += w_i * means
                weighted_squares += w_i * (stds**2 + means**2)

                # Collect min, max, q01, q99
                min_list.append(stats["min"])
                max_list.append(stats["max"])
                q01_list.append(stats["q01"])
                q99_list.append(stats["q99"])

            # Compute overall mean
            overall_mean = weighted_means.tolist()

            # Compute overall variance and std deviation
            overall_variance = weighted_squares - weighted_means**2
            overall_std = np.sqrt(overall_variance).tolist()

            # Compute overall min and max per dimension
            overall_min = np.min(np.array(min_list), axis=0).tolist()
            overall_max = np.max(np.array(max_list), axis=0).tolist()

            # Compute overall q01 and q99 per dimension
            # Use weighted average of per-task quantiles
            q01_array = np.array(q01_list)
            q99_array = np.array(q99_list)
            if percentile_mixing_method == "weighted_average":
                weighted_q01 = np.average(q01_array, axis=0, weights=normalized_weights).tolist()
                weighted_q99 = np.average(q99_array, axis=0, weights=normalized_weights).tolist()
                # std_q01 = np.std(q01_array, axis=0).tolist()
                # std_q99 = np.std(q99_array, axis=0).tolist()
                # print(modality)
                # print(f"{std_q01=}, {std_q99=}")
                # print(f"{weighted_q01=}, {weighted_q99=}")
            elif percentile_mixing_method == "min_max":
                weighted_q01 = np.min(q01_array, axis=0).tolist()
                weighted_q99 = np.max(q99_array, axis=0).tolist()
            else:
                raise ValueError(f"Invalid percentile mixing method: {percentile_mixing_method}")

            # Store the overall statistics for the modality
            overall_stats[modality] = {
                "min": overall_min,
                "max": overall_max,
                "mean": overall_mean,
                "std": overall_std,
                "q01": weighted_q01,
                "q99": weighted_q99,
            }

        return overall_stats

    @staticmethod
    def merge_metadata(
        metadatas: list[DatasetMetadata],
        dataset_sampling_weights: list[float],
        percentile_mixing_method: str,
    ) -> DatasetMetadata:
        """Merge multiple metadata into one."""
        # Convert to dicts
        metadata_dicts = [metadata.model_dump(mode="json") for metadata in metadatas]
        # Create a new metadata dict
        merged_metadata = {}

        # Check all metadata have the same embodiment tag
        assert all(
            metadata.embodiment_tag == metadatas[0].embodiment_tag for metadata in metadatas
        ), "All metadata must have the same embodiment tag"
        merged_metadata["embodiment_tag"] = metadatas[0].embodiment_tag

        # Merge the dataset statistics
        dataset_statistics = {}
        dataset_statistics["state"] = LeRobotMixtureDataset.compute_overall_statistics(
            per_task_stats=[m["statistics"]["state"] for m in metadata_dicts],
            dataset_sampling_weights=dataset_sampling_weights,
            percentile_mixing_method=percentile_mixing_method,
        )
        dataset_statistics["action"] = LeRobotMixtureDataset.compute_overall_statistics(
            per_task_stats=[m["statistics"]["action"] for m in metadata_dicts],
            dataset_sampling_weights=dataset_sampling_weights,
            percentile_mixing_method=percentile_mixing_method,
        )
        merged_metadata["statistics"] = dataset_statistics

        # Merge the modality configs
        modality_configs = defaultdict(set)
        first_modality_configs = {}
        for metadata in metadata_dicts:
            for modality, configs in metadata["modalities"].items():
                modality_configs[modality].add(json.dumps(configs, sort_keys=True))
                if modality not in first_modality_configs:
                    first_modality_configs[modality] = copy.deepcopy(configs)
        merged_metadata["modalities"] = {}
        for modality, configs in modality_configs.items():
            if modality == "video":
                # Video source metadata can differ across datasets while still sharing
                # normalization statistics. Keep a representative copy here and
                # restore per-dataset video metadata before applying transforms.
                merged_metadata["modalities"][modality] = first_modality_configs[modality]
                continue
            # Non-video modalities must still agree inside the same normalization group.
            assert (
                len(configs) == 1
            ), f"Multiple modality configs for modality {modality}: {list(configs)}"
            merged_metadata["modalities"][modality] = json.loads(configs.pop())

        return DatasetMetadata.model_validate(merged_metadata)

    @staticmethod
    def _metadata_for_dataset_transforms(dataset, merged_metadata: DatasetMetadata) -> DatasetMetadata:
        """Preserve dataset-specific video metadata when applying shared stats."""
        metadata_dict = merged_metadata.model_dump()
        metadata_dict["modalities"]["video"] = dataset.metadata.modalities.model_dump()["video"]
        return DatasetMetadata.model_validate(metadata_dict)

    def update_metadata(self, metadata_config: dict, cached_statistics_path: Path | str | None = None) -> None:
        """
        Merge multiple metadatas into one and set the transforms with the merged metadata.

        Args:
            metadata_config (dict): Configuration for the metadata.
                "percentile_mixing_method": The method to mix the percentiles, either "weighted_average" or "min_max".
                    weighted_average: Use the weighted average of the percentiles using the weight used in sampling the datasets.
                    min_max: Use the min of the 1st percentile and max of the 99th percentile.
        """
        self.tag = EmbodimentTag.NEW_EMBODIMENT.value
        self.merged_metadata: dict[str, DatasetMetadata] = {}

        # If cached path is provided, try to load and apply
        if cached_statistics_path is not None:
            try:
                cached_stats = self.load_merged_statistics(cached_statistics_path)
                self.apply_cached_statistics(cached_stats)
                return
            except (FileNotFoundError, KeyError, ValidationError) as e:
                print(f"Failed to load cached statistics: {e}")
                print("Falling back to computing statistics from scratch...")

        # Group metadata by normalization group
        all_metadatas: dict[str, list[DatasetMetadata]] = {}
        all_group_weights: dict[str, list[float]] = {}
        for dataset, dataset_weight in zip(self.datasets, self.dataset_sampling_weights.tolist()):
            if dataset.normalization_group not in all_metadatas:
                all_metadatas[dataset.normalization_group] = []
                all_group_weights[dataset.normalization_group] = []
            all_metadatas[dataset.normalization_group].append(dataset.metadata)
            all_group_weights[dataset.normalization_group].append(dataset_weight)
        for group_key, metadatas in all_metadatas.items():
            self.merged_metadata[group_key] = self.merge_metadata(
                metadatas=metadatas,
                dataset_sampling_weights=all_group_weights[group_key],
                percentile_mixing_method=metadata_config["percentile_mixing_method"],
            )
        for dataset in self.datasets:
            dataset.set_transforms_metadata(
                self._metadata_for_dataset_transforms(
                    dataset, self.merged_metadata[dataset.normalization_group]
                ),
                original_metadata=dataset.metadata,
            )

    def _should_prefix_control_frequency(self) -> bool:
        behavior_cfg = self.data_cfg or {}
        return str(behavior_cfg.get("action_target_mode", "legacy")).lower() == "delta_eef_velocity"

    def _should_prefix_embodiment(self) -> bool:
        behavior_cfg = self.data_cfg or {}
        return bool(behavior_cfg.get("prompt_prefix_embodiment", False))

    def _format_instruction_with_control_frequency(self, instruction: str, dataset) -> str:
        if not self._should_prefix_control_frequency():
            return instruction
        fps = dataset.lerobot_info_meta.get("fps", 1)
        prefix = f"FPS: {fps:g}. "
        if isinstance(instruction, str) and (instruction.startswith(prefix) or "FPS: " in instruction):
            return instruction
        return f"{prefix}{instruction}"

    def _format_instruction_with_prompt_prefixes(self, instruction: str, dataset) -> str:
        instruction = self._format_instruction_with_control_frequency(instruction, dataset)
        return self._format_instruction_with_embodiment(instruction, dataset)

    def _format_instruction_with_embodiment(self, instruction: str, dataset) -> str:
        if not self._should_prefix_embodiment():
            return instruction
        prefix = (
            f"Robot: {self._infer_prompt_robot_name(dataset)}, "
            f"Action: {self._infer_prompt_action_name(dataset)}. "
        )
        if isinstance(instruction, str) and instruction.startswith("Robot: "):
            return instruction
        return f"{prefix}{instruction}"

    @staticmethod
    def _infer_prompt_robot_name(dataset) -> str:
        robot_type = str(getattr(dataset, "robot_type", "") or getattr(dataset, "tag", ""))
        robot_type_lower = robot_type.lower()
        if any(name in robot_type_lower for name in ("agilex", "robotwin", "split_aloha", "robocoin")):
            return "AgileX"
        if any(name in robot_type_lower for name in ("franka", "libero", "oxe_droid", "molmoact")):
            return "Franka"
        return robot_type.split(".")[-1] if robot_type else "Unknown"

    @staticmethod
    def _infer_prompt_action_name(dataset) -> str:
        data_cfg = getattr(dataset, "data_cfg", None) or {}
        action_type = str(data_cfg.get("action_type", "") or "").lower()
        action_target_mode = str(data_cfg.get("action_target_mode", "") or "").lower()
        action_mode = str(data_cfg.get("action_mode", "") or "").lower()
        action_keys = [
            str(key).lower()
            for key in getattr(dataset, "modality_keys", {}).get("action", [])
        ]
        robot_type = str(getattr(dataset, "robot_type", "") or getattr(dataset, "tag", "")).lower()

        if (
            action_type in {"delta_ee", "delta_eef"}
            or action_target_mode == "delta_eef_velocity"
            or any(name in robot_type for name in ("libero", "vla_arena"))
        ):
            return "Delta EEF"
        if "joint" in action_type or any("joint" in key for key in action_keys):
            return "Delta joint" if action_mode == "delta" else "Absolute joint"
        if action_type:
            return action_type.replace("_", " ").title()
        return "Unknown"

    def save_dataset_statistics(self, save_path: Path | str, format: str = "json") -> None:
        """
        Save merged dataset statistics to specified path in the required format.
        Only includes statistics for keys that are actually used in the datasets.
        Gripper-related keys will be placed at the end.
        
        Args:
            save_path (Path | str): Path to save the statistics file
            format (str): Save format, currently only supports "json"
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build the data structure to save
        statistics_data = {}
        
        # Collect actually used keys from all datasets
        all_used_action_keys = []
        all_used_state_keys = []
        
        for dataset in self.datasets:
            used_action_keys, used_state_keys = get_used_modality_keys(dataset.modality_keys)
            for used_action_key in used_action_keys:
                if used_action_key not in all_used_action_keys:
                    all_used_action_keys.append(used_action_key)
            for used_state_key in used_state_keys:
                if used_state_key not in all_used_state_keys:
                    all_used_state_keys.append(used_state_key)
        
        # Organize statistics by tag
        for tag, merged_metadata in self.merged_metadata.items():
            tag_stats = {}
            
            # Process action statistics
            if hasattr(merged_metadata.statistics, 'action') and merged_metadata.statistics.action:
                action_stats = merged_metadata.statistics.action
                
                # Filter and reorder keys - iterate in all_used_action_keys order
                non_gripper_keys = []
                gripper_keys = []
                
                for key in all_used_action_keys:
                    if key in action_stats:
                        non_gripper_keys.append(key)
                
                reordered_keys = non_gripper_keys + gripper_keys
                
                filtered_action_stats = {}
                for key in reordered_keys:
                    filtered_action_stats[key] = action_stats[key]
                
                if filtered_action_stats:
                    combined_action_stats = combine_modality_stats(filtered_action_stats)
                    
                    mask = generate_action_mask_for_used_keys(
                        merged_metadata.modalities.action, filtered_action_stats.keys()
                    )
                    combined_action_stats["mask"] = mask
                    
                    tag_stats["action"] = combined_action_stats
            
            # Process state statistics
            if hasattr(merged_metadata.statistics, 'state') and merged_metadata.statistics.state:
                state_stats = merged_metadata.statistics.state
                
                # Filter and reorder keys - iterate in all_used_state_keys order
                # Filter and reorder keys - iterate in all_used_state_keys order
                non_gripper_keys = []
                gripper_keys = []
                
                for key in all_used_state_keys:
                    if key in state_stats:
                        non_gripper_keys.append(key)
                
                reordered_keys = non_gripper_keys + gripper_keys
                
                filtered_state_stats = {}
                for key in reordered_keys:
                    filtered_state_stats[key] = state_stats[key]
                
                if filtered_state_stats:
                    combined_state_stats = combine_modality_stats(filtered_state_stats)
                    tag_stats["state"] = combined_state_stats
            
            # Add dataset counts
            tag_stats.update(self._get_dataset_counts(tag))
            
            statistics_data[tag] = tag_stats
        
        # Save file
        if format.lower() == "json":
            if not str(save_path).endswith('.json'):
                save_path = save_path.with_suffix('.json')
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(statistics_data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}. Currently only 'json' is supported.")
        
        print(f"Merged dataset statistics saved to: {save_path}")
        print(f"Used action keys (reordered): {list(all_used_action_keys)}")
        print(f"Used state keys (reordered): {list(all_used_state_keys)}")


    def _combine_modality_stats(self, modality_stats: dict) -> dict:
        """Backward compatibility wrapper."""
        return combine_modality_stats(modality_stats)

    def _generate_action_mask_for_used_keys(self, action_modalities: dict, used_action_keys_ordered) -> list[bool]:
        """Backward compatibility wrapper."""
        return generate_action_mask_for_used_keys(action_modalities, used_action_keys_ordered)

    def _get_dataset_counts(self, tag: str) -> dict:
        """
        Get dataset count information for specified normalization group.
        
        Args:
            tag (str): normalization group key
            
        Returns:
            dict: Dictionary containing num_transitions and num_trajectories
        """
        num_transitions = 0
        num_trajectories = 0
        
        # Count dataset information belonging to this normalization group
        for dataset in self.datasets:
            if dataset.normalization_group == tag:
                num_transitions += len(dataset)
                num_trajectories += len(dataset.trajectory_ids)
        
        return {
            "num_transitions": num_transitions,
            "num_trajectories": num_trajectories
        }

    @classmethod
    def load_merged_statistics(cls, load_path: Path | str) -> dict:
        """
        Load merged dataset statistics from file.
        
        Args:
            load_path (Path | str): Path to the statistics file
            
        Returns:
            dict: Dictionary containing merged statistics
        """
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"Statistics file not found: {load_path}")
        
        if load_path.suffix.lower() == '.json':
            with open(load_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif load_path.suffix.lower() == '.pkl':
            import pickle
            with open(load_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {load_path.suffix}")

    def apply_cached_statistics(self, cached_statistics: dict) -> None:
        """
        Apply cached statistics to avoid recomputation.
        
        Args:
            cached_statistics (dict): Statistics loaded from file
        """
        if not cached_statistics:
            raise ValueError("Cached statistics are empty.")

        cached_tags = [tag for tag in cached_statistics.keys() if tag != "metadata"]
        if not cached_tags:
            raise ValueError("Cached statistics contain no normalization-group entries.")

        all_used_action_keys = []
        all_used_state_keys = []
        for dataset in self.datasets:
            used_action_keys, used_state_keys = get_used_modality_keys(dataset.modality_keys)
            for used_action_key in used_action_keys:
                if used_action_key not in all_used_action_keys:
                    all_used_action_keys.append(used_action_key)
            for used_state_key in used_state_keys:
                if used_state_key not in all_used_state_keys:
                    all_used_state_keys.append(used_state_key)

        def build_per_key_statistics(
            flat_stats: dict,
            modality: str,
            ref_metadata: DatasetMetadata,
            used_keys: list[str],
        ) -> dict[str, dict]:
            if not flat_stats:
                raise KeyError(f"Missing cached `{modality}` statistics.")

            stat_names = ["mean", "std", "min", "max", "q01", "q99"]
            lengths = {stat_name: len(flat_stats[stat_name]) for stat_name in stat_names}
            if len(set(lengths.values())) != 1:
                raise ValueError(f"Inconsistent cached `{modality}` statistic lengths: {lengths}")

            ref_stats = getattr(ref_metadata.statistics, modality)
            ordered_keys = [key for key in used_keys if key in ref_stats]
            expected_dim = sum(len(ref_stats[key].mean) for key in ordered_keys)
            actual_dim = lengths["mean"]
            if actual_dim != expected_dim:
                raise ValueError(
                    f"Cached `{modality}` stats dim mismatch: expected {expected_dim}, got {actual_dim}. "
                    f"Current keys: {ordered_keys}"
                )

            per_key_stats = {}
            offset = 0
            for key in ordered_keys:
                dim = len(ref_stats[key].mean)
                per_key_stats[key] = {
                    stat_name: flat_stats[stat_name][offset : offset + dim] for stat_name in stat_names
                }
                offset += dim
            return per_key_stats

        self.merged_metadata = {}
        current_tags = []
        for dataset in self.datasets:
            if dataset.normalization_group not in current_tags:
                current_tags.append(dataset.normalization_group)

        for tag in current_tags:
            tag_datasets = [dataset for dataset in self.datasets if dataset.normalization_group == tag]
            ref_metadata = tag_datasets[0].metadata

            stats_data = cached_statistics.get(tag)
            if stats_data is None and len(cached_tags) == 1:
                stats_data = cached_statistics[cached_tags[0]]
                print(
                    f"Cached statistics tag `{cached_tags[0]}` reused for current tag `{tag}`."
                )
            if stats_data is None:
                raise KeyError(f"Cached statistics do not contain tag `{tag}`.")

            metadata_dict = {
                "embodiment_tag": ref_metadata.embodiment_tag,
                "statistics": {
                    "action": build_per_key_statistics(
                        stats_data.get("action", {}),
                        "action",
                        ref_metadata,
                        all_used_action_keys,
                    ),
                    "state": build_per_key_statistics(
                        stats_data.get("state", {}),
                        "state",
                        ref_metadata,
                        all_used_state_keys,
                    ),
                },
                "modalities": ref_metadata.modalities.model_dump(),
            }
            self.merged_metadata[tag] = DatasetMetadata.model_validate(metadata_dict)

        for dataset in self.datasets:
            if dataset.normalization_group in self.merged_metadata:
                dataset.set_transforms_metadata(
                    self._metadata_for_dataset_transforms(
                        dataset, self.merged_metadata[dataset.normalization_group]
                    ),
                    original_metadata=dataset.metadata,
                )

        print(f"Applied cached statistics for {len(self.merged_metadata)} normalization groups.")
