from pathlib import Path


def resolve_sample_name(rgb_path: str, dataset_path: str) -> str:
    path = Path(rgb_path)
    parts = path.parts
    dataset_lower = dataset_path.lower()

    if "hammer" in dataset_lower:
        if len(parts) >= 4:
            scene_name = parts[-4]
        else:
            scene_name = path.parent.name or "unknown"
        return f"{scene_name}#{path.stem}"

    if "clearpose" in dataset_lower:
        return f"{parts[-3]}#{parts[-2]}#{path.stem}"

    raise ValueError(f"Invalid dataset: {dataset_path}")


def sample_id_from_rgb_path(rgb_path: str) -> str:
    """Return the HAMMER sample id used by infer.py and eval.py."""
    return resolve_sample_name(rgb_path, "hammer")
