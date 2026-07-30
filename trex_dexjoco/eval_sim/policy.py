"""In-process T-Rex policy for DexJoCo sim eval (action 44 + tactile [8,3])."""

from __future__ import annotations

import argparse
import io
import os
import sys
from typing import Any

import numpy as np
import torch
from PIL import Image

_TREX_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.join(_TREX_ROOT, "scripts")
for p in (_TREX_ROOT, _SCRIPTS):
    if p not in sys.path:
        sys.path.insert(0, p)

from adapters.prep_vqvae_data import forces_to_tactile_f6  # noqa: E402
from utils.lerobot_common import F6_PER_FINGER, N_FINGERS  # noqa: E402


def _pil_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _uint8_to_pil(arr: np.ndarray) -> Image.Image:
    a = np.asarray(arr)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[-1] not in (1, 3):
        a = np.transpose(a, (1, 2, 0))
    return Image.fromarray(a).convert("RGB")


def force_obs_to_tactile_f6(force_flat: np.ndarray) -> np.ndarray:
    """OpenPI finger force (R12+L12) → DexJoCo tactile [8,3] (left then right)."""
    f = np.asarray(force_flat, dtype=np.float32).reshape(-1)
    if f.size < 24:
        raise ValueError(f"expected finger force dim>=24, got {f.size}")
    right_12, left_12 = f[:12], f[12:24]
    return forces_to_tactile_f6(left_12[None], right_12[None])[0]


def resolve_trex_ckpt_label(checkpoint: str | os.PathLike) -> str:
    """``checkpoint-13-44646`` → ``ckpt000013`` (ForceVLA-style suffix)."""
    name = os.path.basename(os.path.abspath(str(checkpoint)))
    if name.startswith("checkpoint-"):
        parts = name.split("-")
        if len(parts) >= 2 and parts[1].isdigit():
            return f"ckpt{int(parts[1]):06d}"
    if name.isdigit():
        return f"ckpt{int(name):06d}"
    return f"ckpt_{name}"


def build_load_args(
    checkpoint: str,
    *,
    cuda: int = 0,
    image_size: tuple[int, int] = (384, 288),
    base_model_path: str = "",
) -> argparse.Namespace:
    return argparse.Namespace(
        checkpoint_path=checkpoint,
        base_model_path=base_model_path
        or "/mnt/hdd/checkpoints/trex/Qwen3-VL-2B-Instruct",
        stats_path="",
        dataset_name="dexjoco",
        cuda=cuda,
        action_dim=44,
        action_chunk=16,
        use_robot_state=0,
        use_tactile_vec=1,
        use_tactile_deform=0,
        use_tactile_code=0,
        use_tactile_vqvae=1,
        vqvae_ckpt="",
        vqvae_codebook_size=64,
        vqvae_config=None,
        tactile_intermediate_size=1536,
        n_flare_tokens_per_frame=4,
        n_flare_steps=8,
        cascaded_total_steps=10,
        cascaded_split_step=6,
        image_size=list(image_size),
        disable_tactile=0,
        port=0,
    )


def _patch_server_for_dexjoco(server, per_finger_dim: int = F6_PER_FINGER):
    """Make CascadedServer accept DexJoCo tactile [8,3] instead of [10,6]."""
    import scripts.test as test_mod

    def _rolling_f6_window(tactile_f6_input):
        if tactile_f6_input is None:
            return None
        arr = np.asarray(tactile_f6_input, dtype=np.float32)
        w = server.vqvae_window
        nf, d = N_FINGERS, F6_PER_FINGER
        if arr.ndim == 3:
            if arr.shape[0] >= w:
                arr = arr[-w:]
            else:
                head = np.repeat(arr[:1], w - arr.shape[0], axis=0)
                arr = np.concatenate([head, arr], axis=0)
        else:
            f6 = arr.reshape(nf, d)
            server.f6_buffer.append(f6)
            if len(server.f6_buffer) > w:
                server.f6_buffer = server.f6_buffer[-w:]
            if len(server.f6_buffer) < w:
                head = [server.f6_buffer[0]] * (w - len(server.f6_buffer))
                arr = np.stack(head + server.f6_buffer, axis=0)
            else:
                arr = np.stack(server.f6_buffer, axis=0)
        return arr

    def _encode_tactile_f6(tactile_f6_input, statistic, device):
        if tactile_f6_input is None:
            return None
        arr = np.asarray(tactile_f6_input, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[-1]
        tacf6 = arr.reshape(-1)
        norm = test_mod._normalize(
            tacf6, statistic["tacf6_mask"],
            statistic["tacf6_min"], statistic["tacf6_max"],
        )
        return (
            torch.tensor(norm.reshape(-1, per_finger_dim), dtype=torch.bfloat16)
            .unsqueeze(0).to(device)
        )

    server._rolling_f6_window = _rolling_f6_window
    test_mod._encode_tactile_f6 = _encode_tactile_f6
    return server


class TrexPolicy:
    """Sync policy: each call returns one action chunk [T, 44]."""

    def __init__(
        self,
        checkpoint: str,
        *,
        cuda: int = 0,
        image_size: tuple[int, int] = (384, 288),
        base_model_path: str = "",
        prompt: str = "Assemble the peg into the tray.",
    ):
        from scripts.test import CascadedServer, model_load

        args = build_load_args(
            checkpoint, cuda=cuda, image_size=image_size,
            base_model_path=base_model_path,
        )
        print(f"[TrexPolicy] loading {checkpoint}", flush=True)
        model, processor, statistic = model_load(args)
        per_dim = int(getattr(model, "tacf6_dim", F6_PER_FINGER))
        raw = CascadedServer(args, model, processor, statistic)
        self.server = _patch_server_for_dexjoco(raw, per_finger_dim=per_dim)
        self.prompt = prompt
        self.action_chunk = int(args.action_chunk)
        self._hist: list[np.ndarray] = []
        self.vqvae_window = int(getattr(self.server, "vqvae_window", 16))

    def reset(self):
        self._hist.clear()
        self.server.f6_buffer = []
        self.server.cached_kv = None
        self.server.x_split = None
        self.server.last_actions = None

    def _update_hist(self, tac: np.ndarray) -> np.ndarray:
        self._hist.append(np.asarray(tac, dtype=np.float32))
        w = self.vqvae_window
        if len(self._hist) > w:
            self._hist = self._hist[-w:]
        if len(self._hist) < w:
            head = [self._hist[0]] * (w - len(self._hist))
            return np.stack(head + self._hist, axis=0)
        return np.stack(self._hist, axis=0)

    def infer_chunk(self, obs: dict[str, Any]) -> np.ndarray:
        ego = _uint8_to_pil(obs["base"])
        # Match DexJoCoSftDataset: fast cams = [wrist_left, wrist_right]
        # (CascadedServer.predict() uses right-then-left; do not call it here.)
        wl = _uint8_to_pil(obs["wrist_left"])
        wr = _uint8_to_pil(obs["wrist_right"])
        if "force" not in obs:
            raise KeyError("TrexPolicy needs force_mode=finger so obs has 'force'")
        tac = force_obs_to_tactile_f6(obs["force"])
        hist = self._update_hist(tac)
        prompt = obs.get("prompt") or self.prompt

        out = self._predict_left_first(
            ego=ego, wrist_left=wl, wrist_right=wr,
            tactile_f6=hist, task_description=prompt,
        )
        actions = np.asarray(out["actions"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        if actions.shape[0] == 0:
            raise RuntimeError("T-Rex returned empty action chunk")
        return actions

    def _predict_left_first(
        self,
        *,
        ego: Image.Image,
        wrist_left: Image.Image,
        wrist_right: Image.Image,
        tactile_f6,
        task_description: str,
    ) -> dict:
        """slow_and_fast with fast cams in training order [left, right]."""
        import time

        server = self.server
        fast_list = [wrist_left, wrist_right]
        with server.lock, torch.inference_mode():
            server.model = server.model.to(server.device).eval()
            t0 = time.time()
            server._run_slow(
                task_description, [ego], fast_list, tactile_f6, None, None
            )
            actions, cid = server._run_fast(tactile_f6, None)
            latency_ms = (time.time() - t0) * 1000.0
        return {
            "status": "success",
            "mode": "slow_and_fast",
            "actions": actions,
            "chunk_id": cid,
            "latency_ms": latency_ms,
        }