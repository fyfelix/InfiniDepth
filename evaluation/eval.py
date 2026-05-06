import argparse
import os
from datetime import datetime
from os.path import exists, join

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset import limit_dataset_for_eval, load_test_dataset, sample_name_for_dataset
from utils.metric import abs_relative_difference, rmse_linear, delta1_acc, mae_linear, delta4_acc_105, delta5_acc110


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="InfiniDepth RGB-D depth evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--encoder",
        type=str,
        choices=["vits", "vitb", "vitl", "vitg", "vitl16", "vith16plus"],
        default="vitl16",
        help="Model encoder type",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model checkpoint file",
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="HAMMER, ClearPose, or DREDS JSONL path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_dir",
        help="Evaluation metadata and metrics output directory",
    )
    parser.add_argument(
        "--prediction-dir",
        type=str,
        default=None,
        help="Directory containing per-sample .npy predictions. Defaults to --output/predictions.",
    )
    parser.add_argument(
        "--raw-type",
        type=str,
        required=True,
        choices=["d435", "l515", "tof"],
        help="Raw type. ClearPose only supports d435.",
    )
    parser.add_argument(
        "--input-size", type=int, default=518, help="Input size for inference"
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1000.0,
        help="Scale factor for depth values",
    )
    parser.add_argument(
        "--max-depth", type=float, default=6.0, help="Maximum valid depth value"
    )
    parser.add_argument(
        "--image-min", type=float, default=0.1, help="Minimum valid depth value"
    )
    parser.add_argument(
        "--image-max", type=float, default=5.0, help="Maximum valid depth value"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum number of samples to evaluate. 0 means all samples.",
    )
    return parser.parse_args()


def load_gt_depth(depth_path, depth_scale, max_depth,min_depth):
    depth_GT = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_GT is None:
        raise ValueError(f"Could not load GT depth from {depth_path}")
    depth_GT = np.asarray(depth_GT).astype(np.float32) / depth_scale
    valid_mask = (depth_GT >= min_depth) & (depth_GT <= max_depth)
    depth_GT[~valid_mask] = min_depth
    return depth_GT, valid_mask


def align_prediction_shape(pred, gt_shape, dataset_kind, name):
    if pred.shape == gt_shape:
        return pred
    if dataset_kind != "dreds":
        raise ValueError(
            f"Prediction/GT shape mismatch for {name}: "
            f"dataset_kind={dataset_kind}, pred_shape={pred.shape}, gt_shape={gt_shape}"
        )
    gt_height, gt_width = gt_shape
    return cv2.resize(
        pred.astype(np.float32, copy=False),
        (gt_width, gt_height),
        interpolation=cv2.INTER_NEAREST,
    )


class EvalDataset(Dataset):
    def __init__(self, dataset, output_path, args, depth_scale, align=False):
        self.dataset = dataset
        self.prediction_path = args.prediction_dir or join(output_path, "predictions")
        self.legacy_prediction_path = output_path
        self.args = args
        self.depth_scale = depth_scale
        self.align = align

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        depth_GT, valid_mask = load_gt_depth(sample[2], self.depth_scale, self.args.max_depth, self.args.min_depth)

        name = sample_name_for_dataset(self.args.dataset_kind, sample[0])

        pred_path = join(self.prediction_path, name + ".npy")
        if not exists(pred_path):
            pred_path = join(self.legacy_prediction_path, name + ".npy")
        if not exists(pred_path):
            raise FileNotFoundError(
                f"Prediction for {name} not found in "
                f"{self.prediction_path} or {self.legacy_prediction_path}"
            )

        pred = np.load(pred_path)
        pred = align_prediction_shape(pred, depth_GT.shape, self.args.dataset_kind, name)

        pred_invalid_mask = np.logical_or(np.isnan(pred), np.isinf(pred))
        if pred_invalid_mask.sum() > 0:
            # print(f"Invalid mask: {name} {pred_invalid_mask.sum()}")
            valid_mask = valid_mask & ~pred_invalid_mask

        if self.align:
            depth_GT_reshaped = depth_GT[valid_mask].reshape((-1, 1))
            pred_reshaped = pred[valid_mask].reshape((-1, 1))

            _ones = np.ones_like(pred_reshaped)
            A = np.concatenate([pred_reshaped, _ones], axis=-1)
            X = np.linalg.lstsq(A, depth_GT_reshaped, rcond=None)[0]
            scale, shift = X
            pred_reshaped = scale * pred_reshaped + shift
            pred_reshaped = np.clip(pred_reshaped, a_min=self.args.min_depth, a_max=None)

            # For ALIGN=True, shapes are variable (N_valid, 1), cannot simple stack in default collate
            # We return them as is, but batch_size should be 1 or custom collate used
            return {
                'name': name,
                'pred': pred_reshaped.astype(np.float32),
                'gt': depth_GT_reshaped.astype(np.float32),
                'mask': np.ones_like(pred_reshaped, dtype=bool),
                'is_aligned': True
            }
        else:
            return {
                'name': name,
                'pred': pred.astype(np.float32),
                'gt': depth_GT.astype(np.float32),
                'mask': valid_mask.astype(bool),
                'is_aligned': False
            }


def main():
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args = parse_arguments()

    if args.max_samples < 0:
        raise ValueError("--max-samples must be >= 0.")

    os.makedirs(args.output, exist_ok=True)

    dataset, dataset_kind = load_test_dataset(args.dataset, args.raw_type)
    args.dataset_kind = dataset_kind
    if hasattr(dataset, "depth_scale"):
        args.depth_scale = dataset.depth_scale
    dataset = limit_dataset_for_eval(dataset, args.max_samples)

    depth_scale = args.depth_scale
    args.resolved_prediction_dir = args.prediction_dir or join(args.output, "predictions")
    args.actual_num_samples = len(dataset)

    min_depth = dataset.depth_range[0]
    max_depth = dataset.depth_range[1]

    args.min_depth = min_depth
    args.max_depth = max_depth

    with open(join(args.output, 'eval_args.json'), 'w', encoding="utf-8") as f:
        json.dump(vars(args), f)

    print('min depth is updated and set to ', min_depth, 'and max depth is updated and set to ', max_depth)
    print(f'evaluation device: {DEVICE}')

    all_metrics = []

    ALIGN = False

    # Use DataLoader for acceleration
    eval_dataset = EvalDataset(dataset, args.output, args, depth_scale, align=ALIGN)

    # If ALIGN is True, we can't batch variable sized tensors easily without padding.
    # Since ALIGN=False is default and target for optimization, we use batch > 1 only when ALIGN=False.
    batch_size = 1 if ALIGN else 32
    num_workers = 0 if ALIGN or DEVICE != "cuda" else 8

    loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=DEVICE == "cuda",
    )

    for batch in tqdm(loader):
        names = batch['name']

        # Move tensors to the best available device. Metrics are unchanged.
        pred_depth_ts = batch['pred'].to(DEVICE)
        gt_depth_ts = batch['gt'].to(DEVICE)
        mask_ts = batch['mask'].to(DEVICE)

        # Compute metrics with reduction='none' to get per-sample results
        # All these return (B,) tensors
        l1 = mae_linear(pred_depth_ts, gt_depth_ts, mask_ts, reduction='none')
        rmse = rmse_linear(pred_depth_ts, gt_depth_ts, mask_ts, reduction='none')
        abs_rel = abs_relative_difference(pred_depth_ts, gt_depth_ts, mask_ts, reduction='none')
        d4 = delta4_acc_105(pred_depth_ts, gt_depth_ts, mask_ts, reduction='none')
        d5 = delta5_acc110(pred_depth_ts, gt_depth_ts, mask_ts, reduction='none')
        d1 = delta1_acc(pred_depth_ts, gt_depth_ts, mask_ts, reduction='none')

        # Transfer back to CPU only once per batch
        batch_len = len(names)
        l1_cpu = l1.detach().cpu().numpy()
        rmse_cpu = rmse.detach().cpu().numpy()
        abs_rel_cpu = abs_rel.detach().cpu().numpy()
        d4_cpu = d4.detach().cpu().numpy()
        d5_cpu = d5.detach().cpu().numpy()
        d1_cpu = d1.detach().cpu().numpy()
        for i in range(batch_len):
            metrics = {
                'name': names[i],
                'L1': l1_cpu[i],
                'rmse_linear': rmse_cpu[i],
                'abs_relative_difference': abs_rel_cpu[i],
                'delta4_acc_105': d4_cpu[i],
                'delta5_acc110': d5_cpu[i],
                'delta1_acc': d1_cpu[i],
            }
            all_metrics.append(metrics)

    all_metrics = pd.DataFrame(all_metrics)
    all_metrics_mean = all_metrics.mean(numeric_only=True).to_frame().T

    all_metrics.to_csv(join(args.output, f'all_metrics_{current_time}_{ALIGN}.csv'), index=False)
    all_metrics_mean.to_json(
        join(args.output, f'mean_metrics_{current_time}_{ALIGN}.json'),
        orient='records',
        lines=True,
        force_ascii=False,
    )
    print(f'save dir: {args.output}')


if __name__ == "__main__":
    main()
