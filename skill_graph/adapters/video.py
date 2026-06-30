"""Record ego-camera mp4 from AssemblySim."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np


class SimVideoRecorder:
    def __init__(self, sim, out_path: Path, *, fps: int = 30, max_frames: int = 1500) -> None:
        self._sim = sim
        self._out_path = Path(out_path)
        self._fps = int(fps)
        self._max_frames = int(max_frames)
        self._frames: list[np.ndarray] = []
        self._out_path.parent.mkdir(parents=True, exist_ok=True)

    def capture(self) -> None:
        if len(self._frames) >= self._max_frames:
            return
        frame = self._sim.env.render()
        if isinstance(frame, (list, tuple)):
            frame = frame[0]
        arr = np.asarray(frame)
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
        if arr.mean() < 1.0:
            return
        self._frames.append(np.ascontiguousarray(arr))

    def __enter__(self) -> "SimVideoRecorder":
        self._sim.set_frame_callback(self.capture)
        return self

    def __exit__(self, *_args) -> None:
        self._sim.set_frame_callback(None)
        if self._frames:
            imageio.mimsave(str(self._out_path), self._frames, fps=self._fps)
