"""Record dexjoco env.render() — match dexquery eval (30fps, max 1500 frames / 50s)."""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

EVAL_FPS = 30
EVAL_MAX_FRAMES = 1500

_sim_step: contextvars.ContextVar[int] = contextvars.ContextVar("sim_step", default=0)
_exec_capture: contextvars.ContextVar[Callable[[], None] | None] = contextvars.ContextVar(
    "exec_capture", default=None
)


def sim_step_count() -> int:
    return int(_sim_step.get())


def reset_sim_step() -> None:
    _sim_step.set(0)


def maybe_capture_frame() -> None:
    """Count one control motion step; capture video frame if recording."""
    _sim_step.set(_sim_step.get() + 1)
    cb = _exec_capture.get()
    if cb is not None:
        cb()


@contextmanager
def exec_recording(on_frame: Callable[[], None] | None) -> Iterator[None]:
    if on_frame is None:
        yield
        return
    token = _exec_capture.set(on_frame)
    try:
        yield
    finally:
        _exec_capture.reset(token)


def _frame_to_uint8(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    return np.ascontiguousarray(arr)


class DexEnvVideoRecorder:
    """ego camera via unwrapped.render()[0]; hard cap 50s; reject black frames."""

    def __init__(
        self,
        env,
        out_path: Path,
        *,
        fps: int = EVAL_FPS,
        camera_index: int = 0,
        max_frames: int = EVAL_MAX_FRAMES,
        min_frame_mean: float = 8.0,
    ) -> None:
        self._env = env
        self._raw = env.unwrapped
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = int(fps)
        self.camera_index = int(camera_index)
        self.max_frames = int(max_frames)
        self.min_frame_mean = float(min_frame_mean)
        self._writer = None
        self._frame_count = 0
        self._primed = False

    def _prime(self) -> None:
        if self._primed:
            return
        prime = getattr(self._raw, "_prime_rgb_array_renderer", None)
        if callable(prime):
            prime()
        self._primed = True

    def _grab_frame(self) -> np.ndarray:
        self._prime()
        for attempt in range(2):
            rendered = self._raw.render()
            if isinstance(rendered, (list, tuple)):
                frame = rendered[self.camera_index]
            else:
                frame = rendered
            out = _frame_to_uint8(frame)
            if float(out.mean()) >= self.min_frame_mean:
                return out
            if attempt == 0:
                self._primed = False
                self._prime()
        raise RuntimeError(
            f"render returned black frame (mean={float(out.mean()):.2f}); "
            "EGL/render context likely broken"
        )

    def capture(self) -> None:
        if self._frame_count >= self.max_frames:
            return
        frame = self._grab_frame()
        if self._writer is None:
            from dexjoco.data.video_writer import Mp4VideoWriter

            self._writer = Mp4VideoWriter.create_h264(fps=self.fps)
            self._writer.start(str(self.out_path))
        self._writer.write_frame(frame)
        self._frame_count += 1

    def close(self) -> Path:
        if self._writer is not None:
            self._writer.stop()
            self._writer = None
        if self._frame_count == 0:
            raise RuntimeError(f"No frames captured for {self.out_path}")
        return self.out_path

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def duration_s(self) -> float:
        return self._frame_count / float(self.fps)
