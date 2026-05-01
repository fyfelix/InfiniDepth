#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from tqdm import tqdm

from dataset import HAMMERDataset
from utils.naming import sample_id_from_rgb_path


EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from InfiniDepth.utils.io_utils import depth_to_disparity, load_depth, load_image, read_depth_array
from InfiniDepth.utils.model_utils import build_model
from InfiniDepth.utils.sampling_utils import SAMPLING_METHODS


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


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="InfiniDepth_DepthSensor inference for HAMMER evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to ckpts/infinidepth_depthsensor.ckpt.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="HAMMER test JSONL path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/output",
        help="Run metadata directory for args.json. Predictions fall back here when --prediction-dir is omitted.",
    )
    parser.add_argument(
        "--prediction-dir",
        type=str,
        default=None,
        help="Directory for per-sample .npy predictions. Defaults to --output.",
    )
    parser.add_argument(
        "--visualization-dir",
        type=str,
        default=None,
        help="Directory for optional visualizations. Defaults to --output.",
    )
    parser.add_argument(
        "--raw-type",
        type=str,
        required=True,
        choices=["d435", "l515", "tof"],
        help="HAMMER raw depth field used as the depth sensor input.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="InfiniDepth_DepthSensor",
        choices=["InfiniDepth_DepthSensor"],
        help="Fixed InfiniDepth model variant for metric RGB-D evaluation.",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="vitl16",
        choices=["vitl16", "vith16plus"],
        help="DINOv3 encoder used by the InfiniDepth checkpoint.",
    )
    parser.add_argument(
        "--input-size",
        type=parse_hw,
        default=(768, 1024),
        help="Model input size as HxW. The official depth demo uses 768x1024.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Kept for wrapper compatibility. InfiniDepth inference runs one HAMMER sample at a time.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Kept for wrapper compatibility. The adapted script uses a single-process sample loop.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum number of HAMMER samples to process. 0 means all samples.",
    )
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="Save RGB/raw-depth/pred-depth/GT-depth visualization grids.",
    )
    parser.add_argument(
        "--prompt-samples",
        type=int,
        default=1500,
        help="Maximum number of valid raw-depth prompt pixels used by InfiniDepth load_depth().",
    )
    parser.add_argument(
        "--prompt-min-depth",
        type=float,
        default=None,
        help="Minimum valid raw-depth prompt in meters. Defaults to HAMMER depth-range min.",
    )
    parser.add_argument(
        "--prompt-max-depth",
        type=float,
        default=None,
        help="Maximum valid raw-depth prompt in meters. Defaults to HAMMER depth-range max.",
    )
    parser.add_argument(
        "--enable-noise-filter",
        action="store_true",
        help="Apply strict depth noise filtering before sampling raw-depth prompts.",
    )
    return parser.parse_args()


def validate_inputs(args) -> None:
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")
    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"HAMMER dataset JSONL not found: {args.dataset}")
    if args.batch_size != 1:
        print("[Warning] This InfiniDepth adapter processes one sample at a time; --batch-size is recorded only.")
    if args.num_workers != 0:
        print("[Warning] This InfiniDepth adapter uses a single-process sample loop; --num-workers is recorded only.")
    if args.max_samples < 0:
        raise ValueError("--max-samples must be >= 0.")
    if not torch.cuda.is_available():
        raise RuntimeError("InfiniDepth model initialization requires CUDA. This Mac smoke check can parse/import only.")


def load_infinidepth_model(args):
    model = build_model(
        args.model_type,
        model_path=args.model_path,
        encoder=args.encoder,
    )
    print(f"Loaded {args.model_type} with encoder={args.encoder} from {args.model_path}")
    return model


def resolve_output_dirs(args) -> tuple[str, str, str]:
    output_dir = args.output
    prediction_dir = args.prediction_dir or args.output
    visualization_dir = args.visualization_dir or args.output
    return output_dir, prediction_dir, visualization_dir


def limit_dataset(dataset: HAMMERDataset, max_samples: int) -> HAMMERDataset:
    if max_samples > 0:
        dataset.data = dataset.data[:max_samples]
    return dataset


def save_args(
    args,
    dataset: HAMMERDataset,
    *,
    prediction_dir: str,
    visualization_dir: str,
) -> None:
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(prediction_dir, exist_ok=True)
    if args.save_vis:
        os.makedirs(visualization_dir, exist_ok=True)
    args_dict = vars(args).copy()
    args_dict["input_size"] = list(args.input_size)
    args_dict["dataset_depth_range"] = list(dataset.depth_range)
    args_dict["resolved_output_dir"] = args.output
    args_dict["resolved_prediction_dir"] = prediction_dir
    args_dict["resolved_visualization_dir"] = visualization_dir
    args_dict["actual_num_samples"] = len(dataset)
    args_dict["resolved_model_class"] = args.model_type
    args_dict["prediction_kind"] = "metric_depth_meter"
    with open(os.path.join(args.output, "args.json"), "w", encoding="utf-8") as file:
        json.dump(args_dict, file, indent=2)


@torch.no_grad()
def infer_one_sample(
    model,
    rgb_path: str,
    raw_depth_path: str,
    input_size: tuple[int, int],
    prompt_samples: int,
    prompt_min_depth: float,
    prompt_max_depth: float,
    enable_noise_filter: bool,
) -> np.ndarray:
    _, image, (org_h, org_w) = load_image(rgb_path, input_size)
    image = image.cuda(non_blocking=True)

    gt_depth, prompt_depth, gt_depth_mask = load_depth(
        raw_depth_path,
        input_size,
        num_samples=prompt_samples,
        min_prompt=prompt_min_depth,
        max_prompt=prompt_max_depth,
        enable_noise_filter=enable_noise_filter,
        verbose=False,
    )
    gt_depth = gt_depth.cuda(non_blocking=True)
    prompt_depth = prompt_depth.cuda(non_blocking=True)
    gt_depth_mask = gt_depth_mask.cuda(non_blocking=True)

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
    pred_np[~np.isfinite(pred_np)] = 0.0
    return pred_np


def load_gt_depth_for_vis(depth_path: str, depth_scale: float = 1000.0) -> np.ndarray:
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f"Failed to read GT depth for visualization: {depth_path}")
    return np.asarray(depth).astype(np.float32) / depth_scale


def resize_depth_for_vis(depth: np.ndarray, height: int, width: int) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32).squeeze()
    if depth.shape[:2] == (height, width):
        return depth
    return cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)


def colorize_depth(depth: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32).squeeze()
    valid = np.isfinite(depth) & (depth > 0)
    denom = max(float(max_depth) - float(min_depth), 1e-6)
    normalized = np.clip((depth - float(min_depth)) / denom, 0.0, 1.0)
    normalized[~valid] = 0.0
    colored = cv2.applyColorMap((normalized * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    colored[~valid] = 0
    return colored


def save_visualization(
    rgb_path: str,
    raw_depth_path: str,
    gt_depth_path: str,
    pred_depth: np.ndarray,
    output_path: str,
    min_depth: float,
    max_depth: float,
) -> None:
    with Image.open(rgb_path) as pil_image:
        rgb = np.asarray(ImageOps.exif_transpose(pil_image).convert("RGB"))

    height, width = pred_depth.shape[:2]
    if rgb.shape[:2] != (height, width):
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)

    raw_depth = resize_depth_for_vis(read_depth_array(raw_depth_path), height, width)
    gt_depth = resize_depth_for_vis(load_gt_depth_for_vis(gt_depth_path), height, width)

    panels = [
        rgb,
        colorize_depth(raw_depth, min_depth, max_depth),
        colorize_depth(pred_depth, min_depth, max_depth),
        colorize_depth(gt_depth, min_depth, max_depth),
    ]
    top = np.concatenate(panels[:2], axis=1)
    bottom = np.concatenate(panels[2:], axis=1)
    grid = np.concatenate([top, bottom], axis=0)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(grid).save(output_path, quality=95)


def inference(args) -> None:
    validate_inputs(args)

    dataset = HAMMERDataset(args.dataset, args.raw_type)
    dataset = limit_dataset(dataset, args.max_samples)
    prompt_min_depth = args.prompt_min_depth
    prompt_max_depth = args.prompt_max_depth
    if prompt_min_depth is None:
        prompt_min_depth = float(dataset.depth_range[0])
    if prompt_max_depth is None:
        prompt_max_depth = float(dataset.depth_range[1])

    output_dir, prediction_dir, visualization_dir = resolve_output_dirs(args)
    args.output = output_dir
    save_args(
        args,
        dataset,
        prediction_dir=prediction_dir,
        visualization_dir=visualization_dir,
    )
    model = load_infinidepth_model(args)
    if args.enable_noise_filter:
        print("[Info] Depth noise filtering is enabled before sampling raw-depth prompts.")
    else:
        print("[Info] Depth noise filtering is disabled; raw depth prompts will be used directly.")

    for rgb_path, raw_depth_path, gt_depth_path in tqdm(dataset, desc="InfiniDepth HAMMER inference"):
        sample_id = sample_id_from_rgb_path(rgb_path)
        pred_path = os.path.join(prediction_dir, f"{sample_id}.npy")
        pred_depth = infer_one_sample(
            model=model,
            rgb_path=rgb_path,
            raw_depth_path=raw_depth_path,
            input_size=args.input_size,
            prompt_samples=args.prompt_samples,
            prompt_min_depth=prompt_min_depth,
            prompt_max_depth=prompt_max_depth,
            enable_noise_filter=args.enable_noise_filter,
        )
        np.save(pred_path, pred_depth)
        if args.save_vis:
            vis_path = os.path.join(visualization_dir, f"{sample_id}_promptda_vis.jpg")
            save_visualization(
                rgb_path=rgb_path,
                raw_depth_path=raw_depth_path,
                gt_depth_path=gt_depth_path,
                pred_depth=pred_depth,
                output_path=vis_path,
                min_depth=prompt_min_depth,
                max_depth=prompt_max_depth,
            )

    print(f"Saved predictions to: {prediction_dir}")


if __name__ == "__main__":
    inference(parse_arguments())
