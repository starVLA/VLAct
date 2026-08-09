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

import functools
import random
from typing import Any, ClassVar

import numpy as np
import pytorch3d.transforms as pt
import torch
from pydantic import Field, PrivateAttr, field_validator, model_validator

from ..schema import DatasetMetadata, DatasetStatisticalValues, RotationType, StateActionMetadata
from .base import InvertibleModalityTransform, ModalityTransform


class RotationTransform:
    """Adapted from https://github.com/real-stanford/diffusion_policy/blob/548a52bbb105518058e27bf34dcf90bf6f73681a/diffusion_policy/model/common/rotation_transformer.py"""

    valid_reps = ["axis_angle", "euler_angles", "quaternion", "rotation_6d", "matrix"]

    def __init__(self, from_rep="axis_angle", to_rep="rotation_6d"):
        """
        Valid representations

        Always use matrix as intermediate representation.
        """
        if from_rep.startswith("euler_angles"):
            from_convention = from_rep.split("_")[-1]
            from_rep = "euler_angles"
            from_convention = from_convention.replace("r", "X").replace("p", "Y").replace("y", "Z")
        else:
            from_convention = None
        if to_rep.startswith("euler_angles"):
            to_convention = to_rep.split("_")[-1]
            to_rep = "euler_angles"
            to_convention = to_convention.replace("r", "X").replace("p", "Y").replace("y", "Z")
        else:
            to_convention = None
        assert from_rep != to_rep, f"from_rep and to_rep cannot be the same: {from_rep}"
        assert from_rep in self.valid_reps, f"Invalid from_rep: {from_rep}"
        assert to_rep in self.valid_reps, f"Invalid to_rep: {to_rep}"

        forward_funcs = list()
        inverse_funcs = list()

        if from_rep != "matrix":
            funcs = [getattr(pt, f"{from_rep}_to_matrix"), getattr(pt, f"matrix_to_{from_rep}")]
            if from_convention is not None:
                funcs = [functools.partial(func, convention=from_convention) for func in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        if to_rep != "matrix":
            funcs = [getattr(pt, f"matrix_to_{to_rep}"), getattr(pt, f"{to_rep}_to_matrix")]
            if to_convention is not None:
                funcs = [functools.partial(func, convention=to_convention) for func in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        inverse_funcs = inverse_funcs[::-1]

        self.forward_funcs = forward_funcs
        self.inverse_funcs = inverse_funcs

    @staticmethod
    def _apply_funcs(x: torch.Tensor, funcs: list) -> torch.Tensor:
        assert isinstance(x, torch.Tensor)
        for func in funcs:
            x = func(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(
            x, torch.Tensor
        ), f"Unexpected input type: {type(x)}. Expected type: {torch.Tensor}"
        return self._apply_funcs(x, self.forward_funcs)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(
            x, torch.Tensor
        ), f"Unexpected input type: {type(x)}. Expected type: {torch.Tensor}"
        return self._apply_funcs(x, self.inverse_funcs)


class Normalizer:
    valid_modes = ["q99", "mean_std", "min_max", "binary", "wrap"]

    def __init__(
        self,
        mode: str,
        statistics: dict,
        binary_threshold: float | None = None,
        output_range: tuple[float, float] = (-1.0, 1.0),
    ):
        self.mode = mode
        self.statistics = statistics
        self.binary_threshold = binary_threshold
        self.output_range = output_range
        for key, value in self.statistics.items():
            self.statistics[key] = torch.tensor(value)
        if mode == "binary":
            self.binary_threshold = self._resolve_binary_threshold(binary_threshold)

    def _resolve_binary_threshold(self, default_threshold: float | None):
        if default_threshold is not None:
            return default_threshold
        min_values = self.statistics.get("min")
        max_values = self.statistics.get("max")
        if min_values is None or max_values is None or min_values.shape != max_values.shape:
            return 0.5
        return min_values + (max_values - min_values) / 2

    def _get_binary_threshold(self, x: torch.Tensor):
        if isinstance(self.binary_threshold, torch.Tensor):
            return self.binary_threshold.to(device=x.device, dtype=x.dtype)
        return self.binary_threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(
            x, torch.Tensor
        ), f"Unexpected input type: {type(x)}. Expected type: {torch.Tensor}"
        out_min, out_max = self.output_range

        # Normalize the tensor
        if self.mode == "q99":
            # Range of q99 defaults to [-1, 1], but gripper dims can use [0, 1].
            q01 = self.statistics["q01"].to(x.dtype)
            q99 = self.statistics["q99"].to(x.dtype)

            # In the case of q01 == q99, the normalization will be undefined
            # So we set the normalized values to the original values
            mask = q01 != q99
            normalized = torch.zeros_like(x)

            # Normalize the values where q01 != q99
            # Formula: ((x - q01) / (q99 - q01)) * (out_max - out_min) + out_min
            normalized[..., mask] = (x[..., mask] - q01[..., mask]) / (
                q99[..., mask] - q01[..., mask]
            )
            normalized[..., mask] = normalized[..., mask] * (out_max - out_min) + out_min

            # Set the normalized values to the original values where q01 == q99
            normalized[..., ~mask] = x[..., ~mask].to(x.dtype)

            # Clip the normalized values to the configured output range.
            normalized = torch.clamp(normalized, out_min, out_max)

        elif self.mode == "mean_std":
            # Range of mean_std is not fixed, but can be positive or negative
            mean = self.statistics["mean"].to(x.dtype)
            std = self.statistics["std"].to(x.dtype)

            # In the case of std == 0, the normalization will be undefined
            # So we set the normalized values to the original values
            mask = std != 0
            normalized = torch.zeros_like(x)

            # Normalize the values where std != 0
            # Formula: (x - mean) / std
            normalized[..., mask] = (x[..., mask] - mean[..., mask]) / std[..., mask]

            # Set the normalized values to the original values where std == 0
            normalized[..., ~mask] = x[..., ~mask].to(x.dtype)

        elif self.mode == "min_max":
            # Range of min_max defaults to [-1, 1], but gripper dims can use [0, 1].
            min = self.statistics["min"].to(x.dtype)
            max = self.statistics["max"].to(x.dtype)

            # In the case of min == max, the normalization will be undefined
            # So we set the normalized values to the original values
            mask = min != max
            normalized = torch.zeros_like(x)

            # Normalize the values where min != max
            # Formula: ((x - min) / (max - min)) * (out_max - out_min) + out_min
            normalized[..., mask] = (x[..., mask] - min[..., mask]) / (
                max[..., mask] - min[..., mask]
            )
            normalized[..., mask] = normalized[..., mask] * (out_max - out_min) + out_min

            # Set the normalized values to the original values where min == max
            # normalized[..., ~mask] = x[..., ~mask].to(x.dtype)
            # Set the normalized values to 0 where min == max
            normalized[..., ~mask] = 0

        elif self.mode == "scale":
            # Range of scale is [0, 1]
            min = self.statistics["min"].to(x.dtype)
            max = self.statistics["max"].to(x.dtype)
            abs_max = torch.max(torch.abs(min), torch.abs(max))
            mask = abs_max != 0
            normalized = torch.zeros_like(x)
            normalized[..., mask] = x[..., mask] / abs_max[..., mask]
            normalized[..., ~mask] = 0

        elif self.mode == "binary":
            # Range of binary is [0, 1]
            normalized = (x > self._get_binary_threshold(x)).to(x.dtype)
        elif self.mode == "wrap":
            period = x.new_tensor(2 * np.pi)
            normalized = torch.remainder(x + np.pi, period) - np.pi
        else:
            raise ValueError(f"Invalid normalization mode: {self.mode}")

        return normalized

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(
            x, torch.Tensor
        ), f"Unexpected input type: {type(x)}. Expected type: {torch.Tensor}"
        out_min, out_max = self.output_range
        if self.mode == "q99":
            q01 = self.statistics["q01"].to(x.dtype)
            q99 = self.statistics["q99"].to(x.dtype)
            return (x - out_min) / (out_max - out_min) * (q99 - q01) + q01
        elif self.mode == "mean_std":
            mean = self.statistics["mean"].to(x.dtype)
            std = self.statistics["std"].to(x.dtype)
            return x * std + mean
        elif self.mode == "min_max":
            min = self.statistics["min"].to(x.dtype)
            max = self.statistics["max"].to(x.dtype)
            return (x - out_min) / (out_max - out_min) * (max - min) + min
        elif self.mode == "binary":
            return (x > self._get_binary_threshold(x)).to(x.dtype)
        elif self.mode == "wrap":
            period = x.new_tensor(2 * np.pi)
            return torch.remainder(x + np.pi, period) - np.pi
        else:
            raise ValueError(f"Invalid normalization mode: {self.mode}")


class StateActionToTensor(InvertibleModalityTransform):
    """
    Transforms states and actions to tensors.
    """

    input_dtypes: dict[str, np.dtype] = Field(
        default_factory=dict, description="The input dtypes for each state key."
    )
    output_dtypes: dict[str, torch.dtype] = Field(
        default_factory=dict, description="The output dtypes for each state key."
    )

    def model_dump(self, *args, **kwargs):
        if kwargs.get("mode", "python") == "json":
            include = {"apply_to"}
        else:
            include = kwargs.pop("include", None)

        return super().model_dump(*args, include=include, **kwargs)

    @field_validator("input_dtypes", "output_dtypes", mode="before")
    def validate_dtypes(cls, v):
        for key, dtype in v.items():
            if isinstance(dtype, str):
                if dtype.startswith("torch."):
                    dtype_split = dtype.split(".")[-1]
                    v[key] = getattr(torch, dtype_split)
                elif dtype.startswith("np.") or dtype.startswith("numpy."):
                    dtype_split = dtype.split(".")[-1]
                    v[key] = np.dtype(dtype_split)
                else:
                    raise ValueError(f"Invalid dtype: {dtype}")
        return v

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                continue
            value = data[key]
            assert isinstance(
                value, np.ndarray
            ), f"Unexpected input type: {type(value)}. Expected type: {np.ndarray}"
            data[key] = torch.from_numpy(value)
            if key in self.output_dtypes:
                data[key] = data[key].to(self.output_dtypes[key])
        return data

    def unapply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                continue
            value = data[key]
            assert isinstance(
                value, torch.Tensor
            ), f"Unexpected input type: {type(value)}. Expected type: {torch.Tensor}"
            data[key] = value.numpy()
            if key in self.input_dtypes:
                data[key] = data[key].astype(self.input_dtypes[key])
        return data


class StateActionSignFlipTransform(InvertibleModalityTransform):
    """Flip the sign of selected state/action dimensions."""

    flip_dims: dict[str, list[int]] = Field(
        default_factory=dict,
        description="Per-key last-axis indices to multiply by -1.",
    )

    def model_dump(self, *args, **kwargs):
        if kwargs.get("mode", "python") == "json":
            include = {"apply_to", "flip_dims"}
        else:
            include = kwargs.pop("include", None)
        return super().model_dump(*args, include=include, **kwargs)

    @field_validator("flip_dims", mode="before")
    def validate_flip_dims(cls, v):
        normalized = {}
        for key, dims in (v or {}).items():
            normalized[str(key)] = [int(dim) for dim in dims]
        return normalized

    def _flip_value(self, key: str, value: Any) -> Any:
        dims = self.flip_dims.get(key, [])
        if not dims:
            return value
        if not isinstance(value, (np.ndarray, torch.Tensor)):
            raise TypeError(f"Unexpected input type for {key}: {type(value)}")
        if value.ndim == 0:
            raise ValueError(f"Cannot flip scalar value for {key}")
        max_dim = max(dims)
        if max_dim >= value.shape[-1]:
            raise ValueError(
                f"Flip dim {max_dim} out of bounds for {key} with shape {tuple(value.shape)}"
            )
        value[..., dims] = -value[..., dims]
        return value

    def _flip_statistics(
        self,
        stats: DatasetStatisticalValues,
        dims: list[int],
    ) -> DatasetStatisticalValues:
        flipped = stats.model_copy(deep=True)
        mean = np.array(flipped.mean, copy=True)
        std = np.array(flipped.std, copy=True)
        min_values = np.array(flipped.min, copy=True)
        max_values = np.array(flipped.max, copy=True)
        q01 = np.array(flipped.q01, copy=True)
        q99 = np.array(flipped.q99, copy=True)

        mean[dims] = -mean[dims]
        orig_min = min_values[dims].copy()
        orig_max = max_values[dims].copy()
        min_values[dims] = -orig_max
        max_values[dims] = -orig_min

        orig_q01 = q01[dims].copy()
        orig_q99 = q99[dims].copy()
        q01[dims] = -orig_q99
        q99[dims] = -orig_q01

        return DatasetStatisticalValues(
            mean=mean,
            std=std,
            min=min_values,
            max=max_values,
            q01=q01,
            q99=q99,
        )

    def set_metadata(
        self,
        dataset_metadata: DatasetMetadata,
        original_metadata: DatasetMetadata | None = None,
    ):
        flipped_metadata = dataset_metadata.model_copy(deep=True)
        for key in self.apply_to:
            dims = self.flip_dims.get(key, [])
            if not dims or "." not in key:
                continue
            modality, subkey = key.split(".", 1)
            if modality not in {"state", "action"}:
                continue
            stats_by_modality = getattr(flipped_metadata.statistics, modality)
            if subkey not in stats_by_modality:
                continue
            stats_by_modality[subkey] = self._flip_statistics(stats_by_modality[subkey], dims)
        self.dataset_metadata = flipped_metadata

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key in data:
                data[key] = self._flip_value(key, data[key])
        return data

    def unapply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key in data:
                data[key] = self._flip_value(key, data[key])
        return data


class StateActionTransform(InvertibleModalityTransform):
    """
    Class for state or action transform.

    Args:
        apply_to (list[str]): The keys in the modality to load and transform.
        normalization_modes (dict[str, str]): The normalization modes for each state key.
            If a state key in apply_to is not present in the dictionary, it will not be normalized.
        target_rotations (dict[str, str]): The target representations for each state key.
            If a state key in apply_to is not present in the dictionary, it will not be rotated.
    """

    # Configurable attributes
    apply_to: list[str] = Field(..., description="The keys in the modality to load and transform.")
    normalization_modes: dict[str, str] = Field(
        default_factory=dict, description="The normalization modes for each state key."
    )
    target_rotations: dict[str, str] = Field(
        default_factory=dict, description="The target representations for each state key."
    )
    normalization_statistics: dict[str, dict] = Field(
        default_factory=dict, description="The statistics for each state key."
    )
    binary_threshold: float | None = Field(
        default=None,
        description="Optional threshold for binary normalization mode. If None, infer it from min/max statistics.",
    )
    gripper_cluster_split_value: float = Field(
        default=1.0,
        description="Dataset-level split value used to separate low-scale and high-scale gripper families.",
    )
    gripper_cluster_family_metric: str = Field(
        default="max",
        description="Statistic name used to assign a dataset to a gripper family.",
    )
    gripper_cluster_low_close: float = Field(
        default=-0.01,
        description="Low-scale family close prototype for clustered gripper normalization.",
    )
    gripper_cluster_low_open: float = Field(
        default=0.43,
        description="Low-scale family open prototype for clustered gripper normalization.",
    )
    gripper_cluster_high_close: float = Field(
        default=0.16,
        description="High-scale family close prototype for clustered gripper normalization.",
    )
    gripper_cluster_high_open: float = Field(
        default=5.25,
        description="High-scale family open prototype for clustered gripper normalization.",
    )
    gripper_cluster_low_binary_threshold: float = Field(
        default=0.10,
        description="Low-scale family threshold for clustered binary gripper normalization.",
    )
    gripper_cluster_high_binary_threshold: float = Field(
        default=1.0,
        description="High-scale family threshold for clustered binary gripper normalization.",
    )
    modality_metadata: dict[str, StateActionMetadata] = Field(
        default_factory=dict, description="The modality metadata for each state key."
    )

    # Model variables
    _rotation_transformers: dict[str, RotationTransform] = PrivateAttr(default_factory=dict)
    _normalizers: dict[str, Normalizer] = PrivateAttr(default_factory=dict)
    _input_dtypes: dict[str, np.dtype | torch.dtype] = PrivateAttr(default_factory=dict)
    _gripper_cluster_family_by_key: dict[str, str] = PrivateAttr(default_factory=dict)
    _original_normalization_statistics: dict[str, dict] = PrivateAttr(default_factory=dict)

    # Model constants
    _DEFAULT_MIN_MAX_STATISTICS: ClassVar[dict] = {
        "rotation_6d": {
            "min": [-1, -1, -1, -1, -1, -1],
            "max": [1, 1, 1, 1, 1, 1],
        },
        "euler_angles": {
            "min": [-np.pi, -np.pi, -np.pi],
            "max": [np.pi, np.pi, np.pi],
        },
        "quaternion": {
            "min": [-1, -1, -1, -1],
            "max": [1, 1, 1, 1],
        },
        "axis_angle": {
            "min": [-np.pi, -np.pi, -np.pi],
            "max": [np.pi, np.pi, np.pi],
        },
    }
    def model_dump(self, *args, **kwargs):
        if kwargs.get("mode", "python") == "json":
            include = {
                "apply_to",
                "normalization_modes",
                "target_rotations",
                "gripper_cluster_split_value",
                "gripper_cluster_family_metric",
                "gripper_cluster_low_close",
                "gripper_cluster_low_open",
                "gripper_cluster_high_close",
                "gripper_cluster_high_open",
                "gripper_cluster_low_binary_threshold",
                "gripper_cluster_high_binary_threshold",
            }
        else:
            include = kwargs.pop("include", None)

        return super().model_dump(*args, include=include, **kwargs)

    @field_validator("modality_metadata", mode="before")
    def validate_modality_metadata(cls, v):
        for modality_key, config in v.items():
            if isinstance(config, dict):
                config = StateActionMetadata.model_validate(config)
            else:
                assert isinstance(
                    config, StateActionMetadata
                ), f"Invalid source rotation config: {config}"
            v[modality_key] = config
        return v

    @model_validator(mode="after")
    def validate_normalization_statistics(self):
        for modality_key, normalization_statistics in self.normalization_statistics.items():
            if modality_key in self.normalization_modes:
                normalization_mode = self.normalization_modes[modality_key]
                if normalization_mode == "min_max":
                    assert (
                        "min" in normalization_statistics and "max" in normalization_statistics
                    ), f"Min and max statistics are required for min_max normalization, but got {normalization_statistics}"
                    assert len(normalization_statistics["min"]) == len(
                        normalization_statistics["max"]
                    ), f"Min and max statistics must have the same length, but got {normalization_statistics['min']} and {normalization_statistics['max']}"
                elif normalization_mode == "mean_std":
                    assert (
                        "mean" in normalization_statistics and "std" in normalization_statistics
                    ), f"Mean and std statistics are required for mean_std normalization, but got {normalization_statistics}"
                    assert len(normalization_statistics["mean"]) == len(
                        normalization_statistics["std"]
                    ), f"Mean and std statistics must have the same length, but got {normalization_statistics['mean']} and {normalization_statistics['std']}"
                elif normalization_mode == "q99":
                    assert (
                        "q01" in normalization_statistics and "q99" in normalization_statistics
                    ), f"q01 and q99 statistics are required for q99 normalization, but got {normalization_statistics}"
                    assert len(normalization_statistics["q01"]) == len(
                        normalization_statistics["q99"]
                    ), f"q01 and q99 statistics must have the same length, but got {normalization_statistics['q01']} and {normalization_statistics['q99']}"
                elif normalization_mode == "binary":
                    assert (
                        len(normalization_statistics) == 1
                    ), f"Binary normalization should only have one value, but got {normalization_statistics}"
                    assert normalization_statistics[0] in [
                        0,
                        1,
                    ], f"Binary normalization should only have 0 or 1, but got {normalization_statistics[0]}"
                elif normalization_mode == "wrap":
                    pass
                else:
                    raise ValueError(f"Invalid normalization mode: {normalization_mode}")
        return self

    def set_metadata(
        self,
        dataset_metadata: DatasetMetadata,
        original_metadata: DatasetMetadata | None = None,
    ):
        dataset_statistics = dataset_metadata.statistics
        modality_metadata = dataset_metadata.modalities
        original_statistics = (
            original_metadata.statistics if original_metadata is not None else dataset_statistics
        )
        self._gripper_cluster_family_by_key = {}
        self._original_normalization_statistics = {}

        # Check that all state keys specified in apply_to have their modality_metadata
        for key in self.apply_to:
            split_key = key.split(".", 1)
            assert len(split_key) == 2, "State keys should have two parts: 'modality.key'"
            if key not in self.modality_metadata:
                modality, state_key = split_key
                assert hasattr(modality_metadata, modality), f"{modality} config not found"
                assert state_key in getattr(
                    modality_metadata, modality
                ), f"{state_key} config not found"
                self.modality_metadata[key] = getattr(modality_metadata, modality)[state_key]

        # Check that all state keys specified in normalization_modes have their statistics in state_statistics
        for key in self.normalization_modes:
            split_key = key.split(".", 1)
            assert len(split_key) == 2, "State keys should have two parts: 'modality.key'"
            modality, state_key = split_key
            assert hasattr(dataset_statistics, modality), f"{modality} statistics not found"
            assert state_key in getattr(
                dataset_statistics, modality
            ), f"{state_key} statistics not found"
            assert (
                len(getattr(modality_metadata, modality)[state_key].shape) == 1
            ), f"{getattr(modality_metadata, modality)[state_key].shape=}"
            self.normalization_statistics[key] = getattr(dataset_statistics, modality)[
                state_key
            ].model_dump()
            self._original_normalization_statistics[key] = getattr(original_statistics, modality)[
                state_key
            ].model_dump()

        # Initialize the rotation transformers
        for key in self.target_rotations:
            # Get the original representation of the state
            from_rep = self.modality_metadata[key].rotation_type
            assert from_rep is not None, f"Source rotation type not found for {key}"

            # Get the target representation of the state, will raise an error if the target representation is not valid
            to_rep = RotationType(self.target_rotations[key])

            # If the original representation is not the same as the target representation, initialize the rotation transformer
            if from_rep != to_rep:
                self._rotation_transformers[key] = RotationTransform(
                    from_rep=from_rep.value, to_rep=to_rep.value
                )

        # Initialize the normalizers
        for key in self.normalization_modes:
            modality, state_key = key.split(".", 1)
            normalization_mode = self.normalization_modes[key]
            # If the state has a nontrivial rotation, we need to handle it more carefully
            # For absolute rotations, we need to convert them to the target representation and normalize them using min_max mode,
            # since we can infer the bounds by the representation
            # For relative rotations, we cannot normalize them as we don't know the bounds
            if key in self._rotation_transformers:
                # Case 1: Absolute rotation
                if self.modality_metadata[key].absolute:
                    # Check that the normalization mode is valid
                    assert (
                        normalization_mode == "min_max"
                    ), "Absolute rotations that are converted to other formats must be normalized using `min_max` mode"
                    rotation_type = RotationType(self.target_rotations[key]).value
                    # If the target representation is euler angles, we need to parse the convention
                    if rotation_type.startswith("euler_angles"):
                        rotation_type = "euler_angles"
                    # Get the statistics for the target representation
                    statistics = self._DEFAULT_MIN_MAX_STATISTICS[rotation_type]
                # Case 2: Relative rotation
                else:
                    raise ValueError(
                        f"Cannot normalize relative rotations: {key} that's converted to {self.target_rotations[key]}"
                    )
            # If the state is not continuous, we should not use normalization modes other than binary
            elif (
                not self.modality_metadata[key].continuous
                and normalization_mode != "binary"
            ):
                raise ValueError(
                    f"{key} is not continuous, so it should be normalized using `binary` mode"
                )
            # Initialize the normalizer
            else:
                statistics = self.normalization_statistics[key]
            normalizer_mode = normalization_mode
            binary_threshold = self.binary_threshold
            if normalization_mode in {"cluster_binary", "cluster_continuous"}:
                original_statistics = self._original_normalization_statistics[key]
                family = self._resolve_gripper_cluster_family(key, original_statistics)
                statistics = self._get_gripper_cluster_statistics(family)
                if normalization_mode == "cluster_binary":
                    normalizer_mode = "binary"
                    binary_threshold = self._get_gripper_cluster_binary_threshold(family)
                else:
                    normalizer_mode = "min_max"
                    binary_threshold = None
            output_range = (-1.0, 1.0)
            if "gripper" in state_key.lower() and normalizer_mode in {"min_max", "q99"}:
                output_range = (0.0, 1.0)
            self._normalizers[key] = Normalizer(
                mode=normalizer_mode,
                statistics=statistics,
                binary_threshold=binary_threshold,
                output_range=output_range,
            )

        self.dataset_metadata = dataset_metadata

    def _resolve_gripper_cluster_family(self, key: str, statistics: dict[str, list[float]]) -> str:
        if key in self._gripper_cluster_family_by_key:
            return self._gripper_cluster_family_by_key[key]
        metric_name = self.gripper_cluster_family_metric
        metric_values = statistics.get(metric_name)
        if metric_values is None:
            raise KeyError(
                f"Gripper cluster family metric `{metric_name}` not found in statistics for {key}."
            )
        if len(metric_values) != 1:
            raise ValueError(
                f"Clustered gripper normalization expects scalar statistics for {key}, got {metric_values}."
            )
        metric_value = float(metric_values[0])
        family = "high" if metric_value >= self.gripper_cluster_split_value else "low"
        self._gripper_cluster_family_by_key[key] = family
        return family

    def _get_gripper_cluster_statistics(self, family: str) -> dict[str, list[float]]:
        if family == "low":
            return {
                "min": [self.gripper_cluster_low_close],
                "max": [self.gripper_cluster_low_open],
            }
        if family == "high":
            return {
                "min": [self.gripper_cluster_high_close],
                "max": [self.gripper_cluster_high_open],
            }
        raise ValueError(f"Unsupported gripper cluster family: {family}")

    def _get_gripper_cluster_binary_threshold(self, family: str) -> float:
        if family == "low":
            return self.gripper_cluster_low_binary_threshold
        if family == "high":
            return self.gripper_cluster_high_binary_threshold
        raise ValueError(f"Unsupported gripper cluster family: {family}")

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                # We allow some keys to be missing in the data, and only process the keys that are present
                continue
            if key not in self._input_dtypes:
                input_dtype = data[key].dtype
                assert isinstance(
                    input_dtype, torch.dtype
                ), f"Unexpected input dtype: {input_dtype}. Expected type: {torch.dtype}"
                self._input_dtypes[key] = input_dtype
            else:
                assert (
                    data[key].dtype == self._input_dtypes[key]
                ), f"All states corresponding to the same key must be of the same dtype, input dtype: {data[key].dtype}, expected dtype: {self._input_dtypes[key]}"
            # Rotate the state
            state = data[key]
            if key in self._rotation_transformers:
                state = self._rotation_transformers[key].forward(state)
            # Normalize the state
            if key in self._normalizers:
                state = self._normalizers[key].forward(state)
            data[key] = state
        return data

    def unapply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                continue
            state = data[key]
            assert isinstance(
                state, torch.Tensor
            ), f"Unexpected state type: {type(state)}. Expected type: {torch.Tensor}"
            # Unnormalize the state
            if key in self._normalizers:
                state = self._normalizers[key].inverse(state)
            # Change the state back to its original representation
            if key in self._rotation_transformers:
                state = self._rotation_transformers[key].inverse(state)
            assert isinstance(
                state, torch.Tensor
            ), f"State should be tensor after unapplying transformations, but got {type(state)}"
            # Only convert back to the original dtype if it's known, i.e. `apply` was called before
            # If not, we don't know the original dtype, so we don't convert
            if key in self._input_dtypes:
                original_dtype = self._input_dtypes[key]
                if isinstance(original_dtype, np.dtype):
                    state = state.numpy().astype(original_dtype)
                elif isinstance(original_dtype, torch.dtype):
                    state = state.to(original_dtype)
                else:
                    raise ValueError(f"Invalid input dtype: {original_dtype}")
            data[key] = state
        return data


class StateActionPerturbation(ModalityTransform):
    """
    Class for state or action perturbation.

    Args:
        apply_to (list[str]): The keys in the modality to load and transform.
        std (float): Standard deviation of the noise to be added to the state or action.
    """

    # Configurable attributes
    std: float = Field(
        ..., description="Standard deviation of the noise to be added to the state or action."
    )

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        if not self.training:
            # Don't perturb the data in eval mode
            return data
        if self.std < 0:
            # If the std is negative, we don't add any noise
            return data
        for key in self.apply_to:
            state = data[key]
            assert isinstance(state, torch.Tensor)
            transformed_data_min = torch.min(state)
            transformed_data_max = torch.max(state)
            noise = torch.randn_like(state) * self.std
            state += noise
            # Clip to the original range
            state = torch.clamp(state, transformed_data_min, transformed_data_max)
            data[key] = state
        return data


class StateActionDropout(ModalityTransform):
    """
    Class for state or action dropout.

    Args:
        apply_to (list[str]): The keys in the modality to load and transform.
        dropout_prob (float): Probability of dropping out a state or action.
    """

    # Configurable attributes
    dropout_prob: float = Field(..., description="Probability of dropping out a state or action.")

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        if not self.training:
            # Don't drop out the data in eval mode
            return data
        if self.dropout_prob < 0:
            # If the dropout probability is negative, we don't drop out any states
            return data
        if self.dropout_prob > 1e-9 and random.random() < self.dropout_prob:
            for key in self.apply_to:
                state = data[key]
                assert isinstance(state, torch.Tensor)
                state = torch.zeros_like(state)
                data[key] = state
        return data


class StateActionSinCosTransform(ModalityTransform):
    """
    Class for state or action sin-cos transform.

    Args:
        apply_to (list[str]): The keys in the modality to load and transform.
    """

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            state = data[key]
            assert isinstance(state, torch.Tensor)
            sin_state = torch.sin(state)
            cos_state = torch.cos(state)
            data[key] = torch.cat([sin_state, cos_state], dim=-1)
        return data
