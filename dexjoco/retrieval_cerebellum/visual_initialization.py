"""Weak visual initialization from episode-start multi-camera observations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


DEFAULT_CAMERA_KEYS = (
    "observation.images.ego",
    "observation.images.wrist_left",
    "observation.images.wrist_right",
)
CLIP_IMAGE_MEAN = np.asarray(
    [0.48145466, 0.4578275, 0.40821073],
    dtype=np.float32,
)
CLIP_IMAGE_STD = np.asarray(
    [0.26862954, 0.26130258, 0.27577711],
    dtype=np.float32,
)


@dataclass(frozen=True)
class EpisodeVideoReference:
    episode_index: int
    camera_key: str
    path: Path
    timestamp_s: float


def load_episode_video_references(
    dataset_root: Path,
    episode_index: int,
    frame_index: int,
    *,
    camera_keys: Sequence[str] = DEFAULT_CAMERA_KEYS,
) -> list[EpisodeVideoReference]:
    """Resolve one episode-local frame into packed LeRobot video timestamps."""
    import pyarrow.compute as compute
    import pyarrow.parquet as parquet

    root = Path(dataset_root)
    metadata_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    table = parquet.read_table(metadata_path)
    selected = table.filter(compute.equal(table["episode_index"], int(episode_index)))
    if selected.num_rows != 1:
        raise ValueError(
            f"episode {episode_index} expected one metadata row, got {selected.num_rows}"
        )
    row = selected.to_pylist()[0]
    info = json.loads((root / "meta" / "info.json").read_text())
    fps = float(info["fps"])
    if fps <= 0.0:
        raise ValueError("dataset fps must be positive")

    references = []
    for camera_key in camera_keys:
        prefix = f"videos/{camera_key}"
        chunk_index = int(row[f"{prefix}/chunk_index"])
        file_index = int(row[f"{prefix}/file_index"])
        from_timestamp = float(row[f"{prefix}/from_timestamp"])
        to_timestamp = float(row[f"{prefix}/to_timestamp"])
        timestamp = from_timestamp + float(frame_index) / fps
        if timestamp > to_timestamp + 0.5 / fps:
            raise ValueError(
                f"episode {episode_index} frame {frame_index} exceeds {camera_key} video span"
            )
        path = (
            root
            / "videos"
            / camera_key
            / f"chunk-{chunk_index:03d}"
            / f"file-{file_index:03d}.mp4"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        references.append(
            EpisodeVideoReference(
                episode_index=int(episode_index),
                camera_key=camera_key,
                path=path,
                timestamp_s=timestamp,
            )
        )
    return references


def load_episode_video_reference_series(
    dataset_root: Path,
    episode_index: int,
    frame_indices: Sequence[int],
    *,
    camera_keys: Sequence[str] = DEFAULT_CAMERA_KEYS,
) -> dict[str, list[EpisodeVideoReference]]:
    """Resolve multiple episode-local frames without rereading metadata."""
    import pyarrow.compute as compute
    import pyarrow.parquet as parquet

    root = Path(dataset_root)
    metadata_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    table = parquet.read_table(metadata_path)
    selected = table.filter(compute.equal(table["episode_index"], int(episode_index)))
    if selected.num_rows != 1:
        raise ValueError(
            f"episode {episode_index} expected one metadata row, got {selected.num_rows}"
        )
    row = selected.to_pylist()[0]
    info = json.loads((root / "meta" / "info.json").read_text())
    fps = float(info["fps"])
    if fps <= 0.0:
        raise ValueError("dataset fps must be positive")
    indices = [int(value) for value in frame_indices]
    if any(value < 0 for value in indices):
        raise ValueError("frame indices must be non-negative")
    result: dict[str, list[EpisodeVideoReference]] = {}
    for camera_key in camera_keys:
        prefix = f"videos/{camera_key}"
        chunk_index = int(row[f"{prefix}/chunk_index"])
        file_index = int(row[f"{prefix}/file_index"])
        from_timestamp = float(row[f"{prefix}/from_timestamp"])
        to_timestamp = float(row[f"{prefix}/to_timestamp"])
        path = (
            root
            / "videos"
            / camera_key
            / f"chunk-{chunk_index:03d}"
            / f"file-{file_index:03d}.mp4"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        references = []
        for frame_index in indices:
            timestamp = from_timestamp + float(frame_index) / fps
            if timestamp > to_timestamp + 0.5 / fps:
                raise ValueError(
                    f"episode {episode_index} frame {frame_index} exceeds "
                    f"{camera_key} video span"
                )
            references.append(
                EpisodeVideoReference(
                    episode_index=int(episode_index),
                    camera_key=camera_key,
                    path=path,
                    timestamp_s=timestamp,
                )
            )
        result[camera_key] = references
    return result


def decode_video_frame(reference: EpisodeVideoReference) -> np.ndarray:
    """Decode the first RGB frame at or after the requested timestamp."""
    import av

    with av.open(str(reference.path)) as container:
        stream = container.streams.video[0]
        time_base = float(stream.time_base)
        container.seek(
            int(reference.timestamp_s / time_base),
            stream=stream,
            backward=True,
        )
        closest = None
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            frame_timestamp = float(frame.pts * stream.time_base)
            closest = frame
            if frame_timestamp + 1e-6 >= reference.timestamp_s:
                break
        if closest is None:
            raise RuntimeError(
                f"could not decode {reference.path} at {reference.timestamp_s:.6f}s"
            )
        image = closest.to_ndarray(format="rgb24")
    if image.ndim != 3 or image.shape[2] != 3:
        raise RuntimeError(f"unexpected decoded image shape {image.shape}")
    return image


def decode_video_frames(references: Sequence[EpisodeVideoReference]) -> list[np.ndarray]:
    """Decode ordered timestamps from one packed video with a single open."""
    import av

    items = list(references)
    if not items:
        return []
    path = items[0].path
    if any(item.path != path for item in items):
        raise ValueError("all references must point to the same video")
    timestamps = np.asarray([item.timestamp_s for item in items], dtype=np.float64)
    if np.any(np.diff(timestamps) < 0.0):
        raise ValueError("references must be ordered by timestamp")
    decoded: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        time_base = float(stream.time_base)
        container.seek(
            int(timestamps[0] / time_base),
            stream=stream,
            backward=True,
        )
        target_index = 0
        closest = None
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            frame_timestamp = float(frame.pts * stream.time_base)
            closest = frame
            while (
                target_index < len(timestamps)
                and frame_timestamp + 1e-6 >= timestamps[target_index]
            ):
                image = closest.to_ndarray(format="rgb24")
                decoded.append(image)
                target_index += 1
            if target_index == len(timestamps):
                break
    if len(decoded) != len(items):
        raise RuntimeError(
            f"decoded {len(decoded)} of {len(items)} requested frames from {path}"
        )
    return decoded


def preprocess_clip_images(images: Sequence[np.ndarray], image_size: int = 224):
    """Apply deterministic OpenAI CLIP resize, center crop, and normalization."""
    import cv2
    import torch

    processed = []
    for image in images:
        rgb = np.asarray(image, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"image must have shape (H, W, 3), got {rgb.shape}")
        height, width = rgb.shape[:2]
        scale = max(image_size / height, image_size / width)
        resized_height = max(image_size, int(round(height * scale)))
        resized_width = max(image_size, int(round(width * scale)))
        resized = cv2.resize(
            rgb,
            (resized_width, resized_height),
            interpolation=cv2.INTER_CUBIC,
        )
        top = (resized_height - image_size) // 2
        left = (resized_width - image_size) // 2
        crop = resized[top : top + image_size, left : left + image_size]
        normalized = crop.astype(np.float32) / 255.0
        normalized = (normalized - CLIP_IMAGE_MEAN) / CLIP_IMAGE_STD
        processed.append(np.moveaxis(normalized, -1, 0))
    return torch.from_numpy(np.stack(processed))


class ClipVisionEncoder:
    """Offline-only CLIP vision encoder with deterministic manual preprocessing."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: str = "cuda",
        local_files_only: bool = True,
    ) -> None:
        import torch
        from transformers import CLIPVisionModel

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        self.model_name_or_path = str(model_name_or_path)
        self.device = torch.device(device)
        self.model = CLIPVisionModel.from_pretrained(
            self.model_name_or_path,
            local_files_only=local_files_only,
        ).to(self.device)
        self.model.eval()

    def encode(self, images: Sequence[np.ndarray]) -> np.ndarray:
        import torch

        pixels = preprocess_clip_images(images).to(self.device)
        with torch.inference_mode():
            embedding = self.model(pixel_values=pixels).pooler_output
        values = embedding.detach().cpu().numpy().astype(np.float32)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise RuntimeError("CLIP produced invalid embeddings")
        return values


@dataclass(frozen=True)
class TrainOnlyPCA:
    mean: np.ndarray
    components: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray, n_components: int) -> "TrainOnlyPCA":
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] < 2:
            raise ValueError("PCA features must have shape (N, D) with N >= 2")
        if n_components <= 0:
            raise ValueError("n_components must be positive")
        mean = values.mean(axis=0)
        _, _, right = np.linalg.svd(values - mean, full_matrices=False)
        count = min(int(n_components), values.shape[0] - 1, values.shape[1])
        return cls(
            mean=mean.astype(np.float32),
            components=right[:count].astype(np.float32),
        )

    def transform(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.mean.shape[0]:
            raise ValueError(
                f"PCA features expected shape (N, {self.mean.shape[0]}), got {values.shape}"
            )
        projected = (values - self.mean) @ self.components.T
        return projected.astype(np.float32)


@dataclass(frozen=True)
class EpisodeVisualFeatureStore:
    episode_index: np.ndarray
    split: np.ndarray
    features: np.ndarray
    camera_keys: tuple[str, ...]
    source_frame_index: np.ndarray

    @classmethod
    def load(cls, path: Path) -> "EpisodeVisualFeatureStore":
        with np.load(path, allow_pickle=False) as data:
            store = cls(
                episode_index=np.asarray(data["episode_index"], dtype=np.int64),
                split=np.asarray(data["split"], dtype=str),
                features=np.asarray(data["projected_features"], dtype=np.float32),
                camera_keys=tuple(str(value) for value in data["camera_keys"]),
                source_frame_index=np.asarray(
                    data["source_frame_index"], dtype=np.int64
                ),
            )
        store.validate()
        return store

    def validate(self) -> None:
        count = self.episode_index.shape[0]
        if self.episode_index.shape != (count,):
            raise ValueError("episode_index must be one-dimensional")
        if self.split.shape != (count,) or self.source_frame_index.shape != (count,):
            raise ValueError("visual cache metadata lengths differ")
        if self.features.ndim != 2 or self.features.shape[0] != count:
            raise ValueError("projected_features must have shape (N, D)")
        if len(set(self.episode_index.tolist())) != count:
            raise ValueError("visual cache contains duplicate episodes")
        if not np.isfinite(self.features).all():
            raise ValueError("visual cache contains non-finite features")

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])

    def feature_for(self, episode_index: int, split: str) -> np.ndarray:
        matches = np.flatnonzero(self.episode_index == int(episode_index))
        if matches.size != 1:
            raise KeyError(f"episode {episode_index} missing from visual cache")
        row = int(matches[0])
        if str(self.split[row]) != str(split):
            raise ValueError(
                f"episode {episode_index} split mismatch: cache={self.split[row]} data={split}"
            )
        return self.features[row].copy()
