"""Train the V2 RGB-only segmentation and keypoint heatmap baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from retrieval_cerebellum.spatial_visual_learning import (
    SpatialVisualSidecar,
    build_spatial_visual_model,
    gaussian_heatmaps_torch,
    heatmap_argmax_uv,
    load_episode_batch,
)
from retrieval_cerebellum.spatial_visual_supervision import CAMERA_KEYS, KEYPOINT_NAMES


DEFAULT_DATASET = Path("/mnt/ssd/datasets/dexjoco_lerobot_datasets/bimanual_assembly")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--supervision-dir", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/retrieval_cerebellum/spatial_visual_model"),
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--input-size", type=int, default=320)
    parser.add_argument("--heatmap-sigma-px", type=float, default=2.0)
    parser.add_argument("--max-train-episodes", type=int, default=None)
    parser.add_argument("--max-validation-episodes", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _sidecar_paths(
    supervision_dir: Path,
    split: str,
    maximum: int | None,
) -> list[Path]:
    paths = []
    for path in sorted((supervision_dir / "episodes").glob("episode_*.npz")):
        if SpatialVisualSidecar.load(path).split == split:
            paths.append(path)
    if maximum is not None:
        paths = paths[:maximum]
    return paths


def _dataset(batch) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(batch.images),
        torch.from_numpy(batch.camera_index),
        torch.from_numpy(batch.semantic_masks),
        torch.from_numpy(batch.keypoints_output_uv),
        torch.from_numpy(batch.keypoint_visible),
    )


def _losses(
    output: dict[str, torch.Tensor],
    semantic_masks: torch.Tensor,
    keypoints_uv: torch.Tensor,
    visible: torch.Tensor,
    *,
    sigma_px: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    segmentation_logits = output["segmentation_logits"]
    heatmap_logits = output["heatmap_logits"]
    visibility_logits = output["visibility_logits"]
    if segmentation_logits.shape[-2:] != semantic_masks.shape[-2:]:
        raise ValueError(
            f"model output {segmentation_logits.shape[-2:]} does not match labels "
            f"{semantic_masks.shape[-2:]}"
        )
    segmentation_loss = nn.functional.cross_entropy(
        segmentation_logits,
        semantic_masks,
        weight=torch.tensor(
            [0.1, 4.0, 1.0],
            device=segmentation_logits.device,
            dtype=segmentation_logits.dtype,
        ),
    )
    target_heatmaps = gaussian_heatmaps_torch(
        keypoints_uv,
        visible,
        height=heatmap_logits.shape[-2],
        width=heatmap_logits.shape[-1],
        sigma_px=sigma_px,
    )
    target_distribution = target_heatmaps.reshape(*target_heatmaps.shape[:2], -1)
    target_distribution = target_distribution / target_distribution.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-8)
    log_probability = nn.functional.log_softmax(
        heatmap_logits.reshape(*heatmap_logits.shape[:2], -1),
        dim=-1,
    )
    point_cross_entropy = -(target_distribution * log_probability).sum(dim=-1)
    if bool(visible.any()):
        heatmap_loss = point_cross_entropy[visible].mean()
    else:
        heatmap_loss = heatmap_logits.sum() * 0.0
    visibility_loss = nn.functional.binary_cross_entropy_with_logits(
        visibility_logits,
        visible.to(visibility_logits.dtype),
    )
    total = segmentation_loss + 0.25 * heatmap_loss + 0.5 * visibility_loss
    return total, {
        "total": float(total.detach()),
        "segmentation": float(segmentation_loss.detach()),
        "heatmap": float(heatmap_loss.detach()),
        "visibility": float(visibility_loss.detach()),
    }


class _Metrics:
    def __init__(self) -> None:
        self.confusion = np.zeros((3, 3), dtype=np.int64)
        self.keypoint_errors: list[float] = []
        self.group_errors = [[[] for _ in KEYPOINT_NAMES] for _ in CAMERA_KEYS]
        self.visibility_correct = 0
        self.visibility_count = 0
        self.visibility_brier_sum = 0.0

    def update(
        self,
        output: dict[str, torch.Tensor],
        semantic_masks: torch.Tensor,
        keypoints_uv: torch.Tensor,
        visible: torch.Tensor,
        camera_index: torch.Tensor,
    ) -> None:
        segmentation = output["segmentation_logits"].argmax(dim=1)
        truth = semantic_masks
        combined = (truth.reshape(-1) * 3 + segmentation.reshape(-1)).detach().cpu()
        self.confusion += np.bincount(combined.numpy(), minlength=9).reshape(3, 3)
        predicted_uv = heatmap_argmax_uv(output["heatmap_logits"])
        errors = torch.linalg.norm(predicted_uv - keypoints_uv, dim=-1)
        self.keypoint_errors.extend(errors[visible].detach().cpu().tolist())
        errors_cpu = errors.detach().cpu().numpy()
        visible_cpu = visible.detach().cpu().numpy()
        camera_cpu = camera_index.detach().cpu().numpy()
        for sample_row, camera_row in enumerate(camera_cpu):
            for keypoint_row in range(len(KEYPOINT_NAMES)):
                if visible_cpu[sample_row, keypoint_row]:
                    self.group_errors[int(camera_row)][keypoint_row].append(
                        float(errors_cpu[sample_row, keypoint_row])
                    )
        probability = torch.sigmoid(output["visibility_logits"])
        predicted_visible = probability >= 0.5
        self.visibility_correct += int((predicted_visible == visible).sum().item())
        self.visibility_count += int(visible.numel())
        self.visibility_brier_sum += float(
            (probability - visible.to(probability.dtype)).square().sum().item()
        )

    def summary(self) -> dict[str, object]:
        iou = {}
        for semantic_class, name in ((1, "peg"), (2, "socket")):
            intersection = int(self.confusion[semantic_class, semantic_class])
            union = int(
                self.confusion[semantic_class, :].sum()
                + self.confusion[:, semantic_class].sum()
                - intersection
            )
            iou[name] = float(intersection / union) if union else 0.0
        errors = np.asarray(self.keypoint_errors, dtype=np.float64)
        by_camera_keypoint = {}
        for camera_row, camera in enumerate(CAMERA_KEYS):
            by_camera_keypoint[camera] = {}
            for keypoint_row, keypoint in enumerate(KEYPOINT_NAMES):
                group = np.asarray(
                    self.group_errors[camera_row][keypoint_row],
                    dtype=np.float64,
                )
                by_camera_keypoint[camera][keypoint] = {
                    "count": int(group.size),
                    "p50": float(np.percentile(group, 50)) if group.size else None,
                    "p90": float(np.percentile(group, 90)) if group.size else None,
                }
        return {
            "segmentation_iou": iou,
            "keypoint_visible_count": int(errors.size),
            "keypoint_error_output_px_p50": (
                float(np.percentile(errors, 50)) if errors.size else None
            ),
            "keypoint_error_output_px_p90": (
                float(np.percentile(errors, 90)) if errors.size else None
            ),
            "visibility_accuracy": (
                float(self.visibility_correct / self.visibility_count)
                if self.visibility_count
                else 0.0
            ),
            "visibility_brier": (
                float(self.visibility_brier_sum / self.visibility_count)
                if self.visibility_count
                else 0.0
            ),
            "keypoint_error_output_px_by_camera": by_camera_keypoint,
        }


def _run_split(
    model,
    paths: list[Path],
    *,
    dataset_root: Path,
    input_size: int,
    batch_size: int,
    device: torch.device,
    sigma_px: float,
    optimizer=None,
) -> tuple[dict[str, float], dict[str, object]]:
    training = optimizer is not None
    model.train(training)
    loss_sums = {name: 0.0 for name in ("total", "segmentation", "heatmap", "visibility")}
    batch_count = 0
    metrics = _Metrics()
    ordered_paths = list(paths)
    if training:
        random.shuffle(ordered_paths)
    for path in ordered_paths:
        _, episode = load_episode_batch(
            dataset_root,
            path,
            input_height=input_size,
            input_width=input_size,
        )
        loader = DataLoader(
            _dataset(episode),
            batch_size=batch_size,
            shuffle=training,
            pin_memory=device.type == "cuda",
        )
        for images, camera_index, masks, keypoints, visible in loader:
            images = images.to(device, non_blocking=True)
            camera_index = camera_index.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            keypoints = keypoints.to(device, non_blocking=True)
            visible = visible.to(device, non_blocking=True)
            with torch.set_grad_enabled(training):
                output = model(images, camera_index)
                total, parts = _losses(
                    output,
                    masks,
                    keypoints,
                    visible,
                    sigma_px=sigma_px,
                )
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    total.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
            for name, value in parts.items():
                loss_sums[name] += value
            batch_count += 1
            metrics.update(output, masks, keypoints, visible, camera_index)
    losses = {
        name: value / max(batch_count, 1) for name, value in loss_sums.items()
    }
    return losses, metrics.summary()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.input_size <= 0:
        raise ValueError("epochs, batch-size, and input-size must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    supervision_dir = (
        args.supervision_dir
        or args.dataset_root / "retrieval_cerebellum_spatial_visual"
    )
    train_paths = _sidecar_paths(
        supervision_dir,
        "train",
        args.max_train_episodes,
    )
    validation_paths = _sidecar_paths(
        supervision_dir,
        "validation",
        args.max_validation_episodes,
    )
    if not train_paths or not validation_paths:
        raise ValueError(
            f"need train and validation sidecars, got {len(train_paths)} and "
            f"{len(validation_paths)}"
        )
    first = SpatialVisualSidecar.load(train_paths[0])
    expected_output = first.semantic_masks.shape[-1]
    if args.input_size // 2 != expected_output:
        raise ValueError(
            f"input-size must be twice mask size {expected_output}, got {args.input_size}"
        )
    device = torch.device(args.device)
    model = build_spatial_visual_model(base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_score = float("inf")
    best_epoch = 0
    checkpoint_path = args.output_dir / "best_model.pt"
    for epoch in range(1, args.epochs + 1):
        train_losses, train_metrics = _run_split(
            model,
            train_paths,
            dataset_root=args.dataset_root,
            input_size=args.input_size,
            batch_size=args.batch_size,
            device=device,
            sigma_px=args.heatmap_sigma_px,
            optimizer=optimizer,
        )
        with torch.no_grad():
            validation_losses, validation_metrics = _run_split(
                model,
                validation_paths,
                dataset_root=args.dataset_root,
                input_size=args.input_size,
                batch_size=args.batch_size,
                device=device,
                sigma_px=args.heatmap_sigma_px,
            )
        score_value = validation_metrics["keypoint_error_output_px_p90"]
        score = float(score_value) if score_value is not None else float("inf")
        record = {
            "epoch": epoch,
            "train_losses": train_losses,
            "train_metrics": train_metrics,
            "validation_losses": validation_losses,
            "validation_metrics": validation_metrics,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if score < best_score:
            best_score = score
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "base_channels": args.base_channels,
                    "input_size": args.input_size,
                    "output_size": expected_output,
                    "mask_downsample": first.mask_downsample,
                    "camera_keys": list(CAMERA_KEYS),
                    "keypoint_names": list(KEYPOINT_NAMES),
                    "epoch": epoch,
                    "validation_metrics": validation_metrics,
                },
                checkpoint_path,
            )
    summary = {
        "stage": "V2 RGB spatial visual baseline",
        "train_episodes": [int(path.stem.rsplit("_", 1)[-1]) for path in train_paths],
        "validation_episodes": [
            int(path.stem.rsplit("_", 1)[-1]) for path in validation_paths
        ],
        "epochs": args.epochs,
        "device": str(device),
        "best_epoch": best_epoch,
        "best_validation_keypoint_error_output_px_p90": best_score,
        "checkpoint": str(checkpoint_path),
        "uses_rgb_only_at_inference": True,
        "uses_teacher_geometry_at_inference": False,
        "approved_for_multiview_fusion": False,
        "history": history,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
