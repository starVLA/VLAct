import re

from pathlib import Path

# You can add multimodal datasets here and register a short nickname to ${data_dict}.
# The data format should follow the general multimodal VLM format, for example:
# https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-finetune/README.md

json_root = f"./playground/Datasets/LLaVA-OneVision-COCO/llava_jsons"
image_root = f"./playground/Datasets/LLaVA-OneVision-COCO/images"

SHAREGPT4V_COCO = {
    "annotation_path": f"{json_root}/sharegpt4v_coco.json",
    "data_path": f"{image_root}/",
}

data_dict = {
    "sharegpt4v_coco": SHAREGPT4V_COCO,
}

def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def _resolve_path_dataset(dataset_name):
    """Allow simple path-based dataset specs in addition to registered names."""
    if "::" in dataset_name:
        annotation_path, data_path = dataset_name.split("::", 1)
        return {
            "annotation_path": annotation_path,
            "data_path": data_path,
        }

    dataset_path = Path(dataset_name)
    if dataset_path.is_file():
        if dataset_path.suffix == ".parquet":
            return {
                "annotation_path": str(dataset_path),
                "data_path": "",
                "file_format": "parquet",
            }
        data_path = dataset_path.parent
        if data_path.name == "llava_jsons":
            image_root = data_path.parent / "images"
            data_path = image_root if image_root.exists() else data_path.parent
        return {
            "annotation_path": str(dataset_path),
            "data_path": str(data_path),
        }

    if dataset_path.is_dir():
        candidates = []
        for pattern in ("llava_jsons/*train*.jsonl", "llava_jsons/*train*.json", "*.jsonl", "*.json"):
            candidates.extend(sorted(dataset_path.glob(pattern)))
        if candidates:
            annotation_path = candidates[0]
            image_root = dataset_path / "images"
            data_path = image_root if image_root.exists() else dataset_path
            return {
                "annotation_path": str(annotation_path),
                "data_path": str(data_path),
            }
        parquet_files = sorted(dataset_path.glob("*.parquet"))
        if parquet_files:
            return {
                "annotation_path": str(dataset_path),
                "data_path": "",
                "file_format": "parquet",
            }
        raise ValueError(f"do not find json/jsonl/parquet annotation under {dataset_name}")

    return None

def data_list(dataset_names):
    if dataset_names == ["all"]:
        dataset_names = list(data_dict.keys())
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            config = _resolve_path_dataset(dataset_name)
            if config is None:
                raise ValueError(f"do not find {dataset_name}")
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
    return config_list

if __name__ == "__main__":
    print(data_list)
    
