"""RGB-only learning utilities for V2 spatial visual estimation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .spatial_visual_supervision import CAMERA_KEYS, KEYPOINT_NAMES
from .visual_initialization import (
    DEFAULT_CAMERA_KEYS,
    decode_video_frames,
    load_episode_video_reference_series,
)


IMAGE_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class SpatialVisualSidecar:
    episode_index: int
    split: str
    frame_index: np.ndarray
    image_size: tuple[int, int]
    mask_downsample: int
    keypoints_uv: np.ndarray
    keypoint_visible: np.ndarray
    semantic_masks: np.ndarray

    @classmethod
    def load(cls, path: Path) -> "SpatialVisualSidecar":
        with np.load(path, allow_pickle=False) as data:
            camera_keys = tuple(str(value) for value in data["camera_keys"])
            keypoint_names = tuple(str(value) for value in data["keypoint_names"])
            if camera_keys != CAMERA_KEYS:
                raise ValueError(f"unexpected camera keys {camera_keys}")
            if keypoint_names != KEYPOINT_NAMES:
                raise ValueError(f"unexpected keypoint names {keypoint_names}")
            image_size_values = np.asarray(data["image_size"], dtype=np.int64)
            sidecar = cls(
                episode_index=int(data["episode_index"]),
                split=str(data["split"]),
                frame_index=np.asarray(data["frame_index"], dtype=np.int64).copy(),
                image_size=(int(image_size_values[0]), int(image_size_values[1])),
                mask_downsample=int(data["mask_downsample"]),
                keypoints_uv=np.asarray(data["keypoints_uv"], dtype=np.float32).copy(),
                keypoint_visible=np.asarray(
                    data["keypoint_visible"], dtype=bool
                ).copy(),
                semantic_masks=np.asarray(data["semantic_masks"], dtype=np.uint8).copy(),
            )
        sidecar.validate()
        return sidecar

    def validate(self) -> None:
        frame_count = len(self.frame_index)
        camera_count = len(CAMERA_KEYS)
        keypoint_count = len(KEYPOINT_NAMES)
        if self.keypoints_uv.shape != (frame_count, camera_count, keypoint_count, 2):
            raise ValueError(f"unexpected keypoints_uv shape {self.keypoints_uv.shape}")
        if self.keypoint_visible.shape != (frame_count, camera_count, keypoint_count):
            raise ValueError(
                f"unexpected keypoint_visible shape {self.keypoint_visible.shape}"
            )
        if self.semantic_masks.shape[:2] != (frame_count, camera_count):
            raise ValueError(f"unexpected semantic_masks shape {self.semantic_masks.shape}")
        expected_height = (self.image_size[0] + self.mask_downsample - 1) // self.mask_downsample
        expected_width = (self.image_size[1] + self.mask_downsample - 1) // self.mask_downsample
        if self.semantic_masks.shape[2:] != (expected_height, expected_width):
            raise ValueError(
                "semantic mask size does not match image_size/mask_downsample: "
                f"{self.semantic_masks.shape[2:]} vs {(expected_height, expected_width)}"
            )


@dataclass(frozen=True)
class SpatialVisualEpisodeBatch:
    images: np.ndarray
    camera_index: np.ndarray
    semantic_masks: np.ndarray
    keypoints_output_uv: np.ndarray
    keypoint_visible: np.ndarray


def prepare_episode_batch(
    sidecar: SpatialVisualSidecar,
    camera_images: Sequence[Sequence[np.ndarray]],
    *,
    input_height: int,
    input_width: int,
) -> SpatialVisualEpisodeBatch:
    """Flatten one episode into camera-frame samples without teacher 3D state."""
    frame_count = len(sidecar.frame_index)
    if len(camera_images) != len(CAMERA_KEYS):
        raise ValueError(f"expected {len(CAMERA_KEYS)} camera image sequences")
    if any(len(images) != frame_count for images in camera_images):
        raise ValueError("camera image sequence length must match sidecar frames")
    if input_height <= 0 or input_width <= 0:
        raise ValueError("input image size must be positive")
    samples = frame_count * len(CAMERA_KEYS)
    images = np.empty((samples, 3, input_height, input_width), dtype=np.float32)
    camera_index = np.empty(samples, dtype=np.int64)
    semantic_masks = np.empty(
        (samples, *sidecar.semantic_masks.shape[2:]), dtype=np.int64
    )
    keypoints_output_uv = np.empty((samples, len(KEYPOINT_NAMES), 2), dtype=np.float32)
    keypoint_visible = np.empty((samples, len(KEYPOINT_NAMES)), dtype=bool)
    cursor = 0
    for camera_row, camera_frames in enumerate(camera_images):
        for frame_row, image in enumerate(camera_frames):
            rgb = np.asarray(image, dtype=np.uint8)
            if rgb.shape != (*sidecar.image_size, 3):
                raise ValueError(
                    f"image shape {rgb.shape} does not match sidecar {sidecar.image_size}"
                )
            resized = cv2.resize(
                rgb,
                (input_width, input_height),
                interpolation=cv2.INTER_AREA,
            ).astype(np.float32) / 255.0
            normalized = (resized - IMAGE_MEAN) / IMAGE_STD
            images[cursor] = np.transpose(normalized, (2, 0, 1))
            camera_index[cursor] = camera_row
            semantic_masks[cursor] = sidecar.semantic_masks[frame_row, camera_row]
            keypoints_output_uv[cursor] = (
                sidecar.keypoints_uv[frame_row, camera_row]
                / float(sidecar.mask_downsample)
            )
            keypoint_visible[cursor] = sidecar.keypoint_visible[frame_row, camera_row]
            cursor += 1
    return SpatialVisualEpisodeBatch(
        images=images,
        camera_index=camera_index,
        semantic_masks=semantic_masks,
        keypoints_output_uv=keypoints_output_uv,
        keypoint_visible=keypoint_visible,
    )


def load_episode_batch(
    dataset_root: Path,
    sidecar_path: Path,
    *,
    input_height: int,
    input_width: int,
) -> tuple[SpatialVisualSidecar, SpatialVisualEpisodeBatch]:
    sidecar = SpatialVisualSidecar.load(sidecar_path)
    references = load_episode_video_reference_series(
        dataset_root,
        sidecar.episode_index,
        sidecar.frame_index,
    )
    camera_images = [decode_video_frames(references[key]) for key in DEFAULT_CAMERA_KEYS]
    return sidecar, prepare_episode_batch(
        sidecar,
        camera_images,
        input_height=input_height,
        input_width=input_width,
    )


def gaussian_heatmaps_torch(
    keypoints_uv,
    visible,
    *,
    height: int,
    width: int,
    sigma_px: float,
):
    """Vectorized torch Gaussian targets, imported lazily for data-only users."""
    import torch

    if sigma_px <= 0.0:
        raise ValueError("sigma_px must be positive")
    rows = torch.arange(height, device=keypoints_uv.device, dtype=keypoints_uv.dtype)
    columns = torch.arange(width, device=keypoints_uv.device, dtype=keypoints_uv.dtype)
    grid_row, grid_column = torch.meshgrid(rows, columns, indexing="ij")
    delta_column = grid_column[None, None] - keypoints_uv[..., 0, None, None]
    delta_row = grid_row[None, None] - keypoints_uv[..., 1, None, None]
    heatmaps = torch.exp(
        -(delta_column.square() + delta_row.square()) / (2.0 * sigma_px**2)
    )
    return heatmaps * visible[..., None, None].to(heatmaps.dtype)


def heatmap_argmax_uv(heatmap_logits):
    """Decode heatmap logits into output-grid ``(u, v)`` coordinates."""
    import torch

    batch, keypoints, height, width = heatmap_logits.shape
    flattened = heatmap_logits.reshape(batch, keypoints, height * width)
    indices = flattened.argmax(dim=-1)
    rows = torch.div(indices, width, rounding_mode="floor")
    columns = indices % width
    return torch.stack([columns, rows], dim=-1).to(heatmap_logits.dtype)


def build_spatial_visual_model(*, base_channels: int = 24):
    """Construct a small shared-camera encoder-decoder with three task heads."""
    import torch
    from torch import nn

    if base_channels <= 0:
        raise ValueError("base_channels must be positive")

    def block(input_channels: int, output_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(inplace=True),
        )

    class SpatialVisualNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(6, base_channels, 5, stride=2, padding=2, bias=False),
                nn.BatchNorm2d(base_channels),
                nn.SiLU(inplace=True),
            )
            self.encoder1 = block(base_channels, base_channels)
            self.encoder2 = block(base_channels, base_channels * 2)
            self.encoder3 = block(base_channels * 2, base_channels * 4)
            self.pool = nn.MaxPool2d(2)
            self.decoder2 = block(base_channels * 6, base_channels * 2)
            self.decoder1 = block(base_channels * 3, base_channels)
            self.segmentation_head = nn.Conv2d(base_channels, 3, 1)
            self.heatmap_head = nn.Conv2d(base_channels, len(KEYPOINT_NAMES), 1)
            self.visibility_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(base_channels * 4, len(KEYPOINT_NAMES)),
            )

        def forward(self, images, camera_index):
            one_hot = torch.nn.functional.one_hot(
                camera_index,
                num_classes=len(CAMERA_KEYS),
            ).to(images.dtype)
            camera_planes = one_hot[:, :, None, None].expand(
                -1,
                -1,
                images.shape[-2],
                images.shape[-1],
            )
            encoded1 = self.encoder1(self.stem(torch.cat([images, camera_planes], dim=1)))
            encoded2 = self.encoder2(self.pool(encoded1))
            encoded3 = self.encoder3(self.pool(encoded2))
            decoded2 = torch.nn.functional.interpolate(
                encoded3,
                size=encoded2.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            decoded2 = self.decoder2(torch.cat([decoded2, encoded2], dim=1))
            decoded1 = torch.nn.functional.interpolate(
                decoded2,
                size=encoded1.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            decoded1 = self.decoder1(torch.cat([decoded1, encoded1], dim=1))
            return {
                "segmentation_logits": self.segmentation_head(decoded1),
                "heatmap_logits": self.heatmap_head(decoded1),
                "visibility_logits": self.visibility_head(encoded3),
            }

    return SpatialVisualNet()
