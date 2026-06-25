"""Episode video recording with optional GPU (NVENC) encoding at episode end."""

from __future__ import annotations

import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import numpy as np

EncoderMode = Literal["auto", "nvenc", "cpu"]
ResolvedEncoder = Literal["nvenc", "cpu"]


def _system_ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def _gpu_indices() -> list[str]:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _nvenc_in_ffmpeg() -> bool:
    ffmpeg = _system_ffmpeg()
    if ffmpeg is None:
        return False
    try:
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return "h264_nvenc" in proc.stdout


def _probe_nvenc(*, cuda_device: str) -> bool:
    ffmpeg = _system_ffmpeg()
    if ffmpeg is None:
        return False
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        "256x256",
        "-pix_fmt",
        "rgb24",
        "-r",
        "30",
        "-i",
        "-",
        "-frames:v",
        "1",
        "-an",
        "-c:v",
        "h264_nvenc",
        "-f",
        "null",
        "-",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_device
    try:
        proc = subprocess.run(
            cmd,
            input=frame.tobytes(),
            capture_output=True,
            timeout=10,
            check=False,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def _find_nvenc_device() -> str | None:
    if not _nvenc_in_ffmpeg():
        return None
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip() != "":
        if _probe_nvenc(cuda_device=visible.split(",")[0].strip()):
            return visible.split(",")[0].strip()
        return None
    for idx in _gpu_indices():
        if _probe_nvenc(cuda_device=idx):
            return idx
    return None


def resolve_encoder(mode: EncoderMode) -> tuple[ResolvedEncoder, str | None]:
    if mode == "cpu":
        return "cpu", None
    nvenc_device = _find_nvenc_device()
    if mode == "nvenc":
        if nvenc_device is None:
            print("NVENC unavailable; falling back to libx264 CPU.", flush=True)
            return "cpu", None
        return "nvenc", nvenc_device
    if nvenc_device is not None:
        return "nvenc", nvenc_device
    return "cpu", None


def _encode_mp4(
    path: Path,
    frames: list[np.ndarray],
    *,
    fps: int,
    encoder: ResolvedEncoder,
    nvenc_device: str | None,
) -> None:
    if not frames:
        return
    ffmpeg = _system_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH")

    height, width = frames[0].shape[:2]
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
    ]
    if encoder == "nvenc":
        cmd += ["-c:v", "h264_nvenc", "-preset", "p4", "-pix_fmt", "yuv420p"]
    else:
        cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    cmd += ["-v", "warning", str(path)]

    env = os.environ.copy()
    if encoder == "nvenc" and nvenc_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = nvenc_device

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    assert proc.stdin is not None
    try:
        for frame in frames:
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    finally:
        proc.stdin.close()
    stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed ({rc}) for {path}: {stderr[-500:]}")


class EpisodeVideoRecorder:
    """Buffer frames during rollout; encode all cameras in parallel at episode end."""

    def __init__(
        self,
        video_dir: Path,
        camera_names: list[str],
        *,
        fps: int = 30,
        encoder: EncoderMode = "auto",
    ) -> None:
        self.video_dir = video_dir
        self.fps = fps
        self.camera_names = camera_names
        self.encoder, self.nvenc_device = resolve_encoder(encoder)
        self._frames: dict[str, list[np.ndarray]] = {name: [] for name in camera_names}
        if self.encoder == "nvenc":
            print(
                f"Video encoder: h264_nvenc (GPU {self.nvenc_device}, encode at episode end)",
                flush=True,
            )
        else:
            print("Video encoder: libx264 ultrafast (CPU, encode at episode end)", flush=True)

    def append(self, raw_images: dict[str, np.ndarray]) -> None:
        for name in self.camera_names:
            self._frames[name].append(np.ascontiguousarray(raw_images[name]))

    def close(self) -> None:
        jobs = [
            (self.video_dir / f"{name}.mp4", self._frames[name])
            for name in self.camera_names
            if self._frames[name]
        ]
        if not jobs:
            return

        label = f"NVENC GPU {self.nvenc_device}" if self.encoder == "nvenc" else "CPU"
        print(f"Encoding episode videos ({label}, parallel)...", flush=True)

        def _encode_job(job: tuple[Path, list[np.ndarray]]) -> None:
            path, frames = job
            try:
                _encode_mp4(
                    path,
                    frames,
                    fps=self.fps,
                    encoder=self.encoder,
                    nvenc_device=self.nvenc_device,
                )
            except RuntimeError:
                if self.encoder != "nvenc":
                    raise
                print(f"NVENC failed for {path.name}; retrying with libx264 CPU.", flush=True)
                _encode_mp4(
                    path,
                    frames,
                    fps=self.fps,
                    encoder="cpu",
                    nvenc_device=None,
                )

        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            futures = [pool.submit(_encode_job, job) for job in jobs]
            for future in as_completed(futures):
                future.result()

        for name in self.camera_names:
            self._frames[name].clear()
