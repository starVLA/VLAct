# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Fangjing Wang/ SUST University] in [2025]. 
# Modification: [return raw data and suport multi-dataset mixture].
# Modified by [Jinhui YE/ HKUST University] in [2025]. 
# Modification: [suport topdowm processing, suport param from config].

import inspect
from pathlib import Path
from typing import Any, Sequence
from omegaconf import OmegaConf

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset, LeRobotMixtureDataset
from starVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
from starVLA.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import ROBOT_TYPE_TO_EMBODIMENT_TAG, EmbodimentTag


def _resolve_pretrained_statistics_path(data_cfg) -> Path | None:
    """Resolve a config value to the saved dataset statistics file."""
    stats_source = data_cfg.get("pretrained_stats_path", None)
    if not stats_source:
        return None

    source_path = Path(str(stats_source))
    if source_path.is_dir():
        resolved_path = source_path / "dataset_statistics.json"
    elif source_path.suffix.lower() in {".pt", ".bin", ".safetensors"}:
        resolved_path = source_path.parents[1] / "dataset_statistics.json"
    elif source_path.name in {"config.yaml", "config.yml"}:
        resolved_path = source_path.parent / "dataset_statistics.json"
    else:
        resolved_path = source_path

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Unable to find pretrained statistics file from `{stats_source}`. "
            f"Expected `{resolved_path}` to exist."
        )
    return resolved_path


def collate_fn(batch):
    return batch


def _merge_robot_type_data_cfg_defaults(data_config, data_cfg) -> dict[str, Any] | None:
    defaults_getter = getattr(data_config, "dataset_defaults", None)
    defaults = defaults_getter() if callable(defaults_getter) else None
    if not defaults:
        return data_cfg

    resolved_cfg = {}
    if data_cfg is not None:
        resolved_cfg = OmegaConf.to_container(data_cfg, resolve=True) if OmegaConf.is_config(data_cfg) else dict(data_cfg)

    merged_cfg = dict(defaults)
    merged_cfg.update(resolved_cfg)
    return merged_cfg


def _is_lerobot_dataset_dir(dataset_dir: Path) -> bool:
    return (
        dataset_dir.is_dir()
        and (dataset_dir / "meta" / "info.json").exists()
        and (dataset_dir / "meta" / "modality.json").exists()
        and (dataset_dir / "data").is_dir()
    )


def _expand_lerobot_dataset_names(data_root_dir: Path, data_name: str) -> list[str]:
    """Expand a task-level folder into the LeRobot leaf datasets under it."""
    dataset_dir = data_root_dir / data_name
    if not dataset_dir.is_dir() or _is_lerobot_dataset_dir(dataset_dir):
        return [data_name]

    leaf_dirs = sorted(path for path in dataset_dir.rglob("*") if _is_lerobot_dataset_dir(path))
    if not leaf_dirs:
        return [data_name]

    return [str(path.relative_to(data_root_dir)) for path in leaf_dirs]


def _is_skippable_missing_dataset(data_mix: str, data_name: str) -> bool:
    if not data_mix.startswith("domino"):
        return False

    split_name = data_name.split("/", 1)[0]
    return split_name in {"Clean_Dynamic", "Random_Dynamic", "Clean", "Randomized"}


def make_LeRobotSingleDataset(
    data_root_dir: Path | str,
    data_name: str,
    robot_type: str,
    delete_pause_frame: bool = False,
    data_cfg: dict | None = None,
) -> LeRobotSingleDataset:
    """
    Make a LeRobotSingleDataset object.

    :param data_root_dir: The root directory of the dataset.
    :param data_name: The name of the dataset.
    :param robot_type: The robot type config to use.
    :param crop_obs_camera: Whether to crop the observation camera images.
    :return: A LeRobotSingleDataset object.
    """
    
    data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
    effective_data_cfg = _merge_robot_type_data_cfg_defaults(data_config, data_cfg)
    modality_config = data_config.modality_config()
    transforms = (
        data_config.transform(data_cfg=effective_data_cfg)
        if "data_cfg" in inspect.signature(data_config.transform).parameters
        else data_config.transform()
    )
    dataset_path = data_root_dir / data_name
    if robot_type not in ROBOT_TYPE_TO_EMBODIMENT_TAG:
        print(f"Warning: Robot type {robot_type} not found in ROBOT_TYPE_TO_EMBODIMENT_TAG, using {EmbodimentTag.NEW_EMBODIMENT} as default")
        embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    else:
        embodiment_tag = ROBOT_TYPE_TO_EMBODIMENT_TAG[robot_type]
    normalization_group = embodiment_tag.value
    
    video_backend = effective_data_cfg.get("video_backend", "decord") if effective_data_cfg else "torchvision_av"
    return LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        robot_type=robot_type,
        normalization_group=normalization_group,
        video_backend=video_backend, # decord is more efficiency | torchvision_av for video.av1
        delete_pause_frame=delete_pause_frame,
        data_cfg=effective_data_cfg,
    )

def get_vla_dataset(
    data_cfg: dict,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    seed: int = 42,
    **kwargs: dict,
) -> LeRobotMixtureDataset:
    """
    Get a LeRobotMixtureDataset object.
    """
    data_root_dir = data_cfg.data_root_dir
    data_mix = data_cfg.data_mix
    delete_pause_frame = data_cfg.get("delete_pause_frame", False)
    balance_dataset_weights = data_cfg.get("balance_dataset_weights", balance_dataset_weights)
    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]
    included_datasets, filtered_mixture_spec = set(), []
    root_path = Path(data_root_dir)
    for d_name, d_weight, robot_type in mixture_spec:
        dataset_path = root_path / d_name
        if not dataset_path.exists() and _is_skippable_missing_dataset(data_mix, d_name):
            print(f"Skipping optional missing dataset in `{data_mix}`: {d_name}")
            continue

        expanded_names = _expand_lerobot_dataset_names(root_path, d_name)
        if (
            expanded_names == [d_name]
            and dataset_path.is_dir()
            and not _is_lerobot_dataset_dir(dataset_path)
            and _is_skippable_missing_dataset(data_mix, d_name)
        ):
            print(f"Skipping dataset folder with no LeRobot leaf datasets in `{data_mix}`: {d_name}")
            continue

        for expanded_name in expanded_names:
            dataset_key = (expanded_name, robot_type)
            if dataset_key in included_datasets:
                print(f"Skipping Duplicate Dataset: `{(expanded_name, d_weight, robot_type)}`")
                continue

            included_datasets.add(dataset_key)
            filtered_mixture_spec.append((expanded_name, d_weight, robot_type))

    dataset_mixture = []
    total_original_trajectories = 0
    total_filtered_trajectories = 0
    total_original_steps = 0
    total_filtered_steps = 0
    skipped_empty_datasets = []
    for d_name, d_weight, robot_type in filtered_mixture_spec:
        dataset = make_LeRobotSingleDataset(
            Path(data_root_dir),
            d_name,
            robot_type,
            delete_pause_frame=delete_pause_frame,
            data_cfg=data_cfg,
        )
        total_original_trajectories += getattr(dataset, "original_trajectory_count", len(dataset.trajectory_ids))
        total_filtered_trajectories += getattr(dataset, "filtered_trajectory_count", len(dataset.trajectory_ids))
        total_original_steps += getattr(dataset, "original_step_count", int(dataset.trajectory_lengths.sum()))
        total_filtered_steps += getattr(dataset, "filtered_step_count", len(dataset))

        if len(dataset) == 0:
            skipped_empty_datasets.append(d_name)
            print(
                f"Skipping empty dataset after filtering: {d_name} "
                f"(kept {getattr(dataset, 'filtered_trajectory_count', 0)}/"
                f"{getattr(dataset, 'original_trajectory_count', len(dataset.trajectory_ids))} trajectories)"
            )
            continue

        dataset_mixture.append((dataset, d_weight))

    if skipped_empty_datasets:
        print(
            f"Skipped {len(skipped_empty_datasets)} empty datasets in mixture `{data_mix}`: "
            + ", ".join(skipped_empty_datasets)
        )
    print(
        f"Dataset mixture `{data_mix}` trajectories kept "
        f"{total_filtered_trajectories}/{total_original_trajectories}, steps kept "
        f"{total_filtered_steps}/{total_original_steps}"
    )
    if not dataset_mixture:
        raise ValueError(
            f"No non-empty datasets remain in mixture `{data_mix}` after filtering. "
            f"Original trajectories: {total_original_trajectories}, kept: {total_filtered_trajectories}."
        )

    metadata_config = {
        "percentile_mixing_method": "min_max",
    }
    cached_statistics_path = _resolve_pretrained_statistics_path(data_cfg)

    return LeRobotMixtureDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        seed=seed,
        metadata_config=metadata_config,
        cached_statistics_path=cached_statistics_path,
        data_cfg=data_cfg,
        **kwargs,
    )



if __name__ == "__main__":

    # import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./starVLA/config/training/starvla_cotrain_behavior.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    # debugpy.listen(("0.0.0.0", 10092))
    # print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    # debugpy.wait_for_client()
    args.config_yaml = "./examples/MultiRobot/train_files/starvla_cotrain_multiRobot.yaml"
    cfg = OmegaConf.load(args.config_yaml)
    # cfg.datasets.vla_data.data_mix = "robotwin"
    vla_dataset_cfg = cfg.datasets.vla_data
    # cfg.datasets.vla_data.include_state = True
    vla_dataset_cfg.task_id = 1
    for task_id in ["all"]:
        vla_dataset_cfg.task_id = task_id
        print(f"Testing Task ID: {task_id}")
        dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
        # dataset
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1, # For Debug
        collate_fn=collate_fn,
    )

    cfg.output_dir = "./results/debug"
    output_dir = Path(cfg.output_dir)
    dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")

    from tqdm import tqdm
    count = 0
    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        # print(batch)
        # print(1)
        if count > 100:
            break
        count += 1
        pass