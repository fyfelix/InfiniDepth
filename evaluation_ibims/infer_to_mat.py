#!/usr/bin/env python3
"""Run InfiniDepth_DepthSensor iBims inference and save official *_results.mat files."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from scipy.io import savemat
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MPLCONFIGDIR = Path(os.environ.get("MPLCONFIGDIR", Path(tempfile.gettempdir()) / "infinidepth-matplotlib"))
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

from InfiniDepth.utils.io_utils import (  # noqa: E402
    depth_to_disparity,
    filter_depth_noise_numpy,
    load_image,
)
from InfiniDepth.utils.model_utils import build_model  # noqa: E402
from InfiniDepth.utils.sampling_utils import SAMPLING_METHODS  # noqa: E402


IBIMS_DEPTH_MAX_M = 50.0
IBIMS_DEPTH_SCALE = 65535.0 / IBIMS_DEPTH_MAX_M
SYNTHETIC_RAW_DIR_NAME = "ibims1_synthetic_raw_depth"
EXPECTED_SHAPE = (480, 640)
DEFAULT_MODEL_PATH = "checkpoints/depth/infinidepth_depthsensor.ckpt"
DEFAULT_MODEL_TYPE = "InfiniDepth_DepthSensor"
DEFAULT_ENCODER = "vitl16"
DEFAULT_INPUT_SIZE = (768, 1024)
DEFAULT_PROMPT_SAMPLES = 1500


def parse_hw(value: str) -> tuple[int, int]:
    if "x" in value.lower():
        h_str, w_str = value.lower().split("x", 1)
    elif "," in value:
        h_str, w_str = value.split(",", 1)
    else:
        raise argparse.ArgumentTypeError("Expected HxW or H,W, for example 768x1024.")

    try:
        height = int(h_str)
        width = int(w_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Input size must contain integer height and width.") from exc

    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError("Input height and width must be positive.")
    return height, width


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run InfiniDepth_DepthSensor inference for iBims and write official MAT files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", required=True, help="iBims JSONL manifest path")
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Local InfiniDepth_DepthSensor checkpoint path",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Prediction directory; defaults to evaluation_ibims/output/ibims_<model>_<timestamp>/predictions/<level>",
    )
    parser.add_argument(
        "--model-type",
        default=DEFAULT_MODEL_TYPE,
        choices=[DEFAULT_MODEL_TYPE],
        help="Fixed InfiniDepth model variant for metric RGB-D evaluation.",
    )
    parser.add_argument(
        "--encoder",
        default=DEFAULT_ENCODER,
        choices=["vitl16", "vith16plus"],
        help="DINOv3 encoder used by the InfiniDepth checkpoint.",
    )
    parser.add_argument(
        "--input-size",
        type=parse_hw,
        default=DEFAULT_INPUT_SIZE,
        help="InfiniDepth input size as HxW.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Kept for wrapper compatibility. InfiniDepth inference runs one sample at a time.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Use only the first N manifest rows for smoke testing. 0 means all samples.",
    )
    parser.add_argument(
        "--prompt-samples",
        type=int,
        default=DEFAULT_PROMPT_SAMPLES,
        help="Maximum number of valid raw-depth prompt pixels used by InfiniDepth.",
    )
    parser.add_argument(
        "--prompt-min-depth",
        type=float,
        default=None,
        help="Minimum valid raw-depth prompt in meters. Defaults to manifest depth-range min.",
    )
    parser.add_argument(
        "--prompt-max-depth",
        type=float,
        default=None,
        help="Maximum valid raw-depth prompt in meters. Defaults to manifest depth-range max.",
    )
    parser.add_argument(
        "--enable-noise-filter",
        action="store_true",
        help="Apply strict depth noise filtering before sampling raw-depth prompts.",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=None,
        help="Raw depth scale; defaults to each manifest row depth_scale.",
    )
    return parser.parse_args()


def resolve_root(path: str | Path) -> Path:
    path_obj = Path(path).expanduser()
    if not path_obj.is_absolute():
        path_obj = Path.cwd() / path_obj
    return path_obj.resolve()


def resolve_path(base: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return (base / path).resolve()


def load_model(model_path: str | Path, encoder: str, model_type: str = DEFAULT_MODEL_TYPE):
    model_path = resolve_root(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to initialize InfiniDepth models.")
    model = build_model(
        model_type,
        model_path=str(model_path),
        encoder=encoder,
    )
    print(f"Model: {model_type} encoder={encoder} ({model_path})")
    print("Device: cuda")
    return model


def load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(manifest_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("dataset") != "ibims":
                raise ValueError(f"{manifest_path}:{line_number} is not an iBims row")
            for key in ("sample_id", "rgb", "raw_depth"):
                if key not in row:
                    raise ValueError(f"{manifest_path}:{line_number} missing required key: {key}")
            rows.append(row)

    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return rows


def infer_difficulty(manifest_path: Path, rows: list[dict[str, Any]]) -> str:
    difficulty = rows[0].get("difficulty")
    if difficulty:
        return str(difficulty)
    stem = manifest_path.stem
    return stem[len("ibims_") :] if stem.startswith("ibims_") else stem


def default_output_dir(manifest_path: Path, rows: list[dict[str, Any]], model_path: Path) -> Path:
    difficulty = infer_difficulty(manifest_path, rows)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_stem = model_path.stem
    return (
        PROJECT_ROOT
        / "evaluation_ibims"
        / "output"
        / f"ibims_{model_stem}_{timestamp}"
        / "predictions"
        / difficulty
    )


def row_depth_scale(row: dict[str, Any], cli_depth_scale: float | None) -> float:
    if cli_depth_scale is not None:
        return cli_depth_scale
    return float(row.get("depth_scale", IBIMS_DEPTH_SCALE))


def row_depth_range(row: dict[str, Any]) -> tuple[float, float]:
    depth_range = row.get("depth-range", [0.01, IBIMS_DEPTH_MAX_M])
    return float(depth_range[0]), float(depth_range[1])


def row_prompt_min_depth(row: dict[str, Any], cli_min_depth: float | None) -> float:
    if cli_min_depth is not None:
        return cli_min_depth
    return row_depth_range(row)[0]


def row_prompt_max_depth(row: dict[str, Any], cli_max_depth: float | None) -> float:
    if cli_max_depth is not None:
        return cli_max_depth
    return row_depth_range(row)[1]


def read_single_channel_depth(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read depth image: {path}")
    image = np.asarray(image)
    if image.ndim == 2:
        return image.astype(np.float32)
    if image.ndim == 3:
        if image.shape[2] == 1:
            return image[:, :, 0].astype(np.float32)
        ch0, ch1, ch2 = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        if np.array_equal(ch0, ch1) and np.array_equal(ch1, ch2):
            return ch0.astype(np.float32)
        return ch0.astype(np.float32)
    raise ValueError(f"Unsupported depth image shape {image.shape}: {path}")


def sample_sparse_prompt(depth: np.ndarray, depth_mask: np.ndarray, prompt_samples: int) -> np.ndarray:
    valid_depth = depth * depth_mask
    if prompt_samples <= 0:
        return valid_depth.astype(np.float32, copy=False)
    if (valid_depth > 0.1).sum() <= prompt_samples:
        return valid_depth.astype(np.float32, copy=False)

    height, width = depth.shape
    sample_depth = valid_depth.reshape(-1).copy()
    nonzero_index = np.flatnonzero(sample_depth > 0.1)
    index = np.random.permutation(nonzero_index)[:prompt_samples]
    sample_mask = np.ones_like(sample_depth)
    sample_mask[index] = 0.0
    sample_depth[sample_mask.astype(bool)] = 0.0
    return sample_depth.reshape(height, width).astype(np.float32, copy=False)


def load_scaled_raw_depth(
    raw_depth_path: Path,
    *,
    input_size: tuple[int, int],
    depth_scale: float,
    prompt_min_depth: float,
    prompt_max_depth: float,
    prompt_samples: int,
    enable_noise_filter: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
    depth = read_single_channel_depth(raw_depth_path).astype(np.float32) / float(depth_scale)
    original_shape = depth.shape[:2]
    depth = cv2.resize(depth, input_size[::-1], interpolation=cv2.INTER_NEAREST)
    valid = np.isfinite(depth) & (depth > prompt_min_depth) & (depth < prompt_max_depth)
    depth = np.where(valid, depth, 0.0).astype(np.float32)
    depth_mask = valid.astype(np.float32)

    if enable_noise_filter:
        depth, depth_mask = filter_depth_noise_numpy(
            depth=depth,
            depth_mask=depth_mask,
            std_threshold=0.8,
            median_threshold=0.5,
            gradient_threshold=0.5,
            min_neighbors=5,
            bilateral_d=7,
            bilateral_sigma_color=1.0,
            bilateral_sigma_space=10.0,
            verbose=False,
        )
        depth = np.where(depth_mask > 0, depth, 0.0).astype(np.float32)
        depth_mask = depth_mask.astype(np.float32)

    prompt_depth = sample_sparse_prompt(depth, depth_mask, prompt_samples)

    depth_ts = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).float().cuda(non_blocking=True)
    prompt_ts = torch.from_numpy(prompt_depth).unsqueeze(0).unsqueeze(0).float().cuda(non_blocking=True)
    mask_ts = torch.from_numpy(depth_mask).unsqueeze(0).unsqueeze(0).float().cuda(non_blocking=True)
    return depth_ts, prompt_ts, mask_ts, original_shape


def normalize_prediction(pred_depth: np.ndarray, sample_id: str) -> np.ndarray:
    pred = np.asarray(pred_depth, dtype=np.float32)
    if pred.ndim == 3 and pred.shape[0] == 1:
        pred = pred[0]
    if pred.ndim != 2:
        raise ValueError(f"{sample_id}: expected HxW prediction, got shape {pred.shape}")
    if pred.shape != EXPECTED_SHAPE:
        raise ValueError(f"{sample_id}: expected prediction shape {EXPECTED_SHAPE}, got {pred.shape}")
    invalid = ~np.isfinite(pred) | (pred <= 0.0)
    pred[invalid] = np.nan
    return pred.astype(np.float32, copy=False)


def iter_batches(rows: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


@torch.inference_mode()
def infer_one_sample(
    model,
    *,
    sample_id: str,
    rgb_path: Path,
    raw_depth_path: Path,
    input_size: tuple[int, int],
    depth_scale: float,
    prompt_min_depth: float,
    prompt_max_depth: float,
    prompt_samples: int,
    enable_noise_filter: bool,
) -> np.ndarray:
    _, image, (org_h, org_w) = load_image(str(rgb_path), input_size)
    image = image.cuda(non_blocking=True)

    gt_depth, prompt_depth, gt_depth_mask, raw_shape = load_scaled_raw_depth(
        raw_depth_path,
        input_size=input_size,
        depth_scale=depth_scale,
        prompt_min_depth=prompt_min_depth,
        prompt_max_depth=prompt_max_depth,
        prompt_samples=prompt_samples,
        enable_noise_filter=enable_noise_filter,
    )
    if raw_shape != (org_h, org_w):
        raise ValueError(
            f"RGB/depth shape mismatch for {sample_id}: rgb={(org_h, org_w)}, depth={raw_shape}"
        )

    query_coord = SAMPLING_METHODS["2d_uniform"]((org_h, org_w)).unsqueeze(0).cuda(non_blocking=True)
    pred_depth, _ = model.inference(
        image=image,
        query_coord=query_coord,
        gt_depth=depth_to_disparity(gt_depth),
        gt_depth_mask=gt_depth_mask,
        prompt_depth=depth_to_disparity(prompt_depth),
        prompt_mask=prompt_depth > 0,
    )
    pred_depthmap = pred_depth.permute(0, 2, 1).reshape(1, 1, org_h, org_w)
    pred_np = pred_depthmap.squeeze().detach().cpu().numpy().astype(np.float32)
    return normalize_prediction(pred_np, sample_id)


@torch.inference_mode()
def run_manifest_inference(
    manifest_path: str | Path,
    output_dir: str | Path,
    model,
    *,
    batch_size: int = 1,
    input_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
    prompt_samples: int = DEFAULT_PROMPT_SAMPLES,
    prompt_min_depth: float | None = None,
    prompt_max_depth: float | None = None,
    enable_noise_filter: bool = False,
    depth_scale: float | None = None,
    max_samples: int | None = None,
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be greater than 0")
    if batch_size != 1:
        print("[Warning] InfiniDepth iBims adapter processes one sample at a time; --batch-size is recorded only.")
    if max_samples is not None and max_samples < 0:
        raise ValueError("--max-samples must be >= 0")

    manifest_path = resolve_root(manifest_path)
    output_dir = resolve_root(output_dir)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    rows = load_manifest(manifest_path)
    if max_samples is not None and max_samples > 0:
        rows = rows[:max_samples]
    difficulty = infer_difficulty(manifest_path, rows)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    progress = tqdm(total=len(rows), desc=f"iBims {difficulty} inference")
    try:
        for batch_rows in iter_batches(rows, batch_size):
            for row in batch_rows:
                sample_id = str(row["sample_id"])
                rgb_path = resolve_path(manifest_path.parent, row["rgb"])
                raw_depth_path = resolve_path(manifest_path.parent, row["raw_depth"])

                pred_depth = infer_one_sample(
                    model,
                    sample_id=sample_id,
                    rgb_path=rgb_path,
                    raw_depth_path=raw_depth_path,
                    input_size=input_size,
                    depth_scale=row_depth_scale(row, depth_scale),
                    prompt_min_depth=row_prompt_min_depth(row, prompt_min_depth),
                    prompt_max_depth=row_prompt_max_depth(row, prompt_max_depth),
                    prompt_samples=prompt_samples,
                    enable_noise_filter=enable_noise_filter,
                )
                savemat(
                    output_dir / f"{sample_id}_results.mat",
                    {"pred_depths": pred_depth.astype(np.float32, copy=False)},
                )
                written += 1
                progress.update(1)
    finally:
        progress.close()

    stats = {
        "difficulty": difficulty,
        "manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "num_predictions": written,
        "prediction_shape": list(EXPECTED_SHAPE),
    }
    metadata = dict(run_metadata or {})
    metadata.update(stats)
    metadata["input_size"] = list(input_size)
    with open(output_dir / "infer_args.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False, sort_keys=True, default=str)

    return stats


def main() -> None:
    args = parse_args()
    manifest_path = resolve_root(args.manifest)
    model_path = resolve_root(args.model_path)
    rows = load_manifest(manifest_path)
    output_dir = (
        resolve_root(args.output_dir)
        if args.output_dir
        else default_output_dir(manifest_path, rows, model_path)
    )
    model = load_model(model_path, args.encoder, args.model_type)

    stats = run_manifest_inference(
        manifest_path,
        output_dir,
        model,
        batch_size=args.batch_size,
        input_size=args.input_size,
        prompt_samples=args.prompt_samples,
        prompt_min_depth=args.prompt_min_depth,
        prompt_max_depth=args.prompt_max_depth,
        enable_noise_filter=args.enable_noise_filter,
        depth_scale=args.depth_scale,
        max_samples=args.max_samples,
        run_metadata={
            **vars(args),
            "model_path": str(model_path),
            "resolved_model_class": args.model_type,
            "encoder": args.encoder,
            "output_kind": "metric_depth_meter",
            "alignment": "none",
            "raw_depth_decoder": "manifest_depth_scale",
            "device_resolved": "cuda",
        },
    )
    print(f"Wrote {stats['num_predictions']} official iBims predictions to: {output_dir}")


if __name__ == "__main__":
    main()
