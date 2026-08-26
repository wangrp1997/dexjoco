"""DexJoCo ego JPEG + labels.parquet -> image / 2-ch heatmap."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from gaze_heatmap.utils import heatmap


@dataclass(frozen=True)
class FrameRecord:
    episode_dir: Path
    frame: int
    image_rel: str
    tip_u: float
    tip_v: float
    hole_u: float
    hole_v: float
    tip_visible: bool
    hole_visible: bool
    tip_in_frame: bool
    hole_in_frame: bool


def discover_episodes(data_root: Path) -> list[Path]:
    eps = sorted(p for p in data_root.glob("episode_*") if (p / "labels.parquet").exists())
    if not eps:
        raise FileNotFoundError(f"no episode_*/labels.parquet under {data_root}")
    return eps


def load_index(data_root: Path, episode_dirs: list[Path] | None = None) -> list[FrameRecord]:
    episode_dirs = episode_dirs or discover_episodes(data_root)
    rows: list[FrameRecord] = []
    for ep_dir in episode_dirs:
        table = pq.read_table(ep_dir / "labels.parquet")
        for rec in table.to_pylist():
            rows.append(
                FrameRecord(
                    episode_dir=ep_dir,
                    frame=int(rec["frame"]),
                    image_rel=str(rec["image"]),
                    tip_u=float(rec["tip_u"]),
                    tip_v=float(rec["tip_v"]),
                    hole_u=float(rec["hole_u"]),
                    hole_v=float(rec["hole_v"]),
                    tip_visible=bool(rec["tip_visible"]),
                    hole_visible=bool(rec["hole_visible"]),
                    tip_in_frame=bool(rec["tip_in_frame"]),
                    hole_in_frame=bool(rec["hole_in_frame"]),
                )
            )
    return rows


class GazeSpiralDataset(Dataset):
    """Channel 0 = hole, channel 1 = tip (matches reference repo)."""

    def __init__(
        self,
        records: list[FrameRecord],
        *,
        image_size: int = 224,
        sigma: float = 5.0,
        train: bool = False,
    ) -> None:
        self.records = records
        self.image_size = int(image_size)
        self.sigma = float(sigma)
        self.train = train

    def __len__(self) -> int:
        return len(self.records)

    def _resize_points(
        self,
        img: np.ndarray,
        tip_uv: tuple[float, float],
        hole_uv: tuple[float, float],
    ) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
        h0, w0 = img.shape[:2]
        s = self.image_size
        sx, sy = s / w0, s / h0
        out = cv2.resize(img, (s, s), interpolation=cv2.INTER_LINEAR)
        tip = (tip_uv[0] * sx, tip_uv[1] * sy)
        hole = (hole_uv[0] * sx, hole_uv[1] * sy)
        return out, tip, hole

    def _heatmaps(
        self,
        hole_uv: tuple[float, float],
        tip_uv: tuple[float, float],
        *,
        hole_ok: bool,
        tip_ok: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        s = self.image_size
        rep = np.zeros((2, s, s), dtype=np.float32)
        mask = np.zeros((2, s, s), dtype=np.float32)
        if hole_ok:
            rep[0] = heatmap(self.sigma, s, s, [hole_uv])
            mask[0] = 1.0
        if tip_ok:
            rep[1] = heatmap(self.sigma, s, s, [tip_uv])
            mask[1] = 1.0
        return rep, mask

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self.records[idx]
        img_path = rec.episode_dir / rec.image_rel
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise FileNotFoundError(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if self.train and np.random.rand() < 0.5:
            rgb = rgb[:, ::-1].copy()
            w = rgb.shape[1]
            tip_uv = (w - 1 - rec.tip_u, rec.tip_v)
            hole_uv = (w - 1 - rec.hole_u, rec.hole_v)
        else:
            tip_uv = (rec.tip_u, rec.tip_v)
            hole_uv = (rec.hole_u, rec.hole_v)

        rgb, tip_uv, hole_uv = self._resize_points(rgb, tip_uv, hole_uv)
        if self.train:
            if np.random.rand() < 0.5:
                rgb = np.clip(rgb.astype(np.float32) * np.random.uniform(0.7, 1.2), 0, 255).astype(np.uint8)
            if np.random.rand() < 0.3:
                k = np.random.choice([3, 5])
                rgb = cv2.GaussianBlur(rgb, (k, k), 0)

        hole_ok = rec.hole_in_frame and rec.hole_visible
        tip_ok = rec.tip_in_frame and rec.tip_visible
        rep, mask = self._heatmaps(hole_uv, tip_uv, hole_ok=hole_ok, tip_ok=tip_ok)

        rgb_f = rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb_f = (rgb_f - mean) / std
        x = torch.from_numpy(rgb_f.transpose(2, 0, 1))
        y = torch.from_numpy(rep)
        m = torch.from_numpy(mask)
        return {"image": x, "heatmap": y, "mask": m}
