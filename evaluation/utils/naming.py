from pathlib import Path


def sample_id_from_rgb_path(rgb_path: str) -> str:
    """Return the HAMMER sample id used by infer.py and eval.py."""
    path = Path(rgb_path)
    parts = path.parts
    if len(parts) >= 4:
        scene_name = parts[-4]
    else:
        scene_name = path.parent.name or "unknown"
    return f"{scene_name}#{path.stem}"
