import json
from glob import glob
from os.path import dirname, join
from pathlib import Path

from torch.utils.data import Dataset


def detect_dataset_kind(jsonl_path):
    path_lower = str(jsonl_path).lower()
    if "clearpose" in path_lower:
        return "clearpose"
    if "hammer" in path_lower:
        return "hammer"
    if "transcg" in path_lower:
        return "transcg"
    if "std_cat" in path_lower or "dreds" in path_lower:
        return "dreds"
    raise ValueError(f"Invalid dataset: {jsonl_path}")


def load_test_dataset(jsonl_path, raw_type="d435"):
    dataset_kind = detect_dataset_kind(jsonl_path)
    if dataset_kind == "clearpose":
        if raw_type.lower() != "d435":
            raise ValueError("ClearPose dataset only supports raw-type=d435")
        return ClearPoseDataset(jsonl_path), dataset_kind
    if dataset_kind == "hammer":
        return HAMMERDataset(jsonl_path, raw_type), dataset_kind
    if dataset_kind == "dreds":
        return DREDSDataset(jsonl_path), dataset_kind
    if dataset_kind == "transcg":
        if raw_type.lower() != "d435":
            raise ValueError("TransCG dataset only supports raw-type=d435")
        return TransCGDataset(jsonl_path), dataset_kind
    raise ValueError(f"Invalid dataset kind: {dataset_kind}")


def sample_name_for_dataset(dataset_kind, rgb_path):
    path = Path(str(rgb_path))
    parts = path.parts

    if dataset_kind == "hammer":
        scene_name = parts[-4] if len(parts) >= 4 else path.parent.name or "unknown"
        return f"{scene_name}#{path.stem}"
    if dataset_kind in ("clearpose", "dreds"):
        if len(parts) < 3:
            raise ValueError(f"Cannot derive {dataset_kind} sample name from: {rgb_path}")
        return f"{parts[-3]}#{parts[-2]}#{path.stem}"
    if dataset_kind == "transcg":
        if len(parts) >= 3:
            return f"{parts[-3]}_{parts[-2]}"
        return path.stem
    raise ValueError(f"Invalid dataset kind: {dataset_kind}")


def sample_name_for_sample(dataset_kind, sample):
    if dataset_kind == "transcg" and len(sample) >= 4 and sample[3]:
        return str(sample[3])
    return sample_name_for_dataset(dataset_kind, sample[0])


class HAMMERDataset(Dataset):
    def __init__(self, jsonl_path, raw_type="d435"):
        self.jsonl_path = jsonl_path
        self.dataset_name = "hammer"
        self.root = dirname(jsonl_path)
        self.data = []

        with open(jsonl_path, "r", encoding="utf-8") as file:
            for line in file:
                self.data.append(json.loads(line))

        self.raw_type = raw_type
        self.depth_range = self.data[0]["depth-range"]
        self.depth_scale = 1000.0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        rgb = join(self.root, item["rgb"])
        raw_type = self.raw_type.lower()

        if raw_type == "d435":
            raw_depth = join(self.root, item["d435_depth"])
        elif raw_type == "l515":
            raw_depth = join(self.root, item["l515_depth"])
        elif raw_type == "tof":
            raw_depth = join(self.root, item["tof_depth"])
        else:
            raise ValueError(f"Invalid raw type: {self.raw_type}")

        gt_depth = join(self.root, item["depth"])
        return rgb, raw_depth, gt_depth


class ClearPoseDataset(Dataset):
    def __init__(self, jsonl_path, max_length_each_sequence=300):
        self.jsonl_path = jsonl_path
        self.dataset_name = "clearpose"
        self.root = dirname(jsonl_path)
        self.data = []
        self.rgbs = []
        self.raw_depths = []
        self.gt_depths = []

        depth_range = None

        with open(jsonl_path, "r", encoding="utf-8") as file:
            for line in file:
                item = json.loads(line)
                if depth_range is None:
                    depth_range = item["depth-range"]

                rgb = sorted(
                    glob(join(self.root, item["rgb"], "*" + item["rgb-suffix"]))
                )[:max_length_each_sequence]
                raw_depth = sorted(
                    glob(join(self.root, item["rgb"], "*" + item["raw_depth-suffix"]))
                )[:max_length_each_sequence]
                gt_depth = sorted(
                    glob(join(self.root, item["rgb"], "*" + item["depth-suffix"]))
                )[:max_length_each_sequence]

                self.rgbs.extend(rgb)
                self.raw_depths.extend(raw_depth)
                self.gt_depths.extend(gt_depth)
                self.data.append(item)

        self.depth_range = depth_range
        self.depth_scale = 1000.0

    def __len__(self):
        return len(self.rgbs)

    def __getitem__(self, idx):
        return self.rgbs[idx], self.raw_depths[idx], self.gt_depths[idx]


class DREDSDataset(Dataset):
    def __init__(self, jsonl_path, max_length_each_sequence=50):
        self.jsonl_path = jsonl_path
        self.dataset_name = "dreds"
        self.root = dirname(jsonl_path)
        self.data = []
        self.rgbs = []
        self.raw_depths = []
        self.gt_depths = []

        depth_range = None

        with open(jsonl_path, "r", encoding="utf-8") as file:
            for line in file:
                item = json.loads(line)
                if depth_range is None:
                    depth_range = item["depth-range"]

                rgb = sorted(
                    glob(join(self.root, item["rgb"], "*" + item["rgb-suffix"]))
                )[:max_length_each_sequence]
                raw_depth = sorted(
                    glob(join(self.root, item["rgb"], "*" + item["raw_depth-suffix"]))
                )[:max_length_each_sequence]
                gt_depth = sorted(
                    glob(join(self.root, item["rgb"], "*" + item["depth-suffix"]))
                )[:max_length_each_sequence]

                self.rgbs.extend(rgb)
                self.raw_depths.extend(raw_depth)
                self.gt_depths.extend(gt_depth)
                self.data.append(item)

        self.depth_range = depth_range
        self.depth_scale = 1.0

    def __len__(self):
        return len(self.rgbs)

    def __getitem__(self, idx):
        return self.rgbs[idx], self.raw_depths[idx], self.gt_depths[idx]


class TransCGDataset(Dataset):
    def __init__(self, jsonl_path, default_depth_range=(0.1, 6.0)):
        self.jsonl_path = jsonl_path
        self.dataset_name = "transcg"
        self.root = dirname(jsonl_path)
        self.data = []

        depth_range = None

        with open(jsonl_path, "r", encoding="utf-8") as file:
            for line_no, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue

                item = json.loads(line)
                missing_keys = [key for key in ("rgb", "depth") if key not in item]
                if "d435_depth" not in item and "raw_depth" not in item:
                    missing_keys.append("d435_depth")
                if missing_keys:
                    raise KeyError(
                        f"TransCG jsonl row {line_no} is missing keys: {missing_keys}"
                    )

                if depth_range is None and "depth-range" in item:
                    depth_range = item["depth-range"]

                self.data.append(item)

        if not self.data:
            raise ValueError(f"TransCG jsonl is empty: {jsonl_path}")

        self.depth_range = (
            depth_range if depth_range is not None else list(default_depth_range)
        )
        self.depth_scale = 1000.0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        rgb = join(self.root, item["rgb"])
        raw_depth = join(self.root, item.get("d435_depth", item.get("raw_depth")))
        gt_depth = join(self.root, item["depth"])
        sample_name = item.get("sample_name") or sample_name_for_dataset("transcg", rgb)

        return rgb, raw_depth, gt_depth, sample_name


def load_dataset_for_eval(dataset_path, raw_type):
    dataset, _ = load_test_dataset(dataset_path, raw_type)
    return dataset


def limit_dataset_for_eval(dataset, max_samples):
    if max_samples <= 0:
        return dataset
    if isinstance(dataset, (ClearPoseDataset, DREDSDataset)):
        dataset.rgbs = dataset.rgbs[:max_samples]
        dataset.raw_depths = dataset.raw_depths[:max_samples]
        dataset.gt_depths = dataset.gt_depths[:max_samples]
        return dataset
    if isinstance(dataset, HAMMERDataset):
        dataset.data = dataset.data[:max_samples]
        return dataset
    if isinstance(dataset, TransCGDataset):
        dataset.data = dataset.data[:max_samples]
        return dataset
    raise TypeError(f"Unsupported dataset type: {type(dataset).__name__}")
