"""DexJoCoSftDataset — post-train loader for native LeRobot + force_labels.

Mirrors `scripts/train.SftDataset.collate_fn` batch contract so cascaded MoT
training is unchanged. Differences vs upstream JSON:
  * action = absolute chunk [16, 44] (Allegro/Panda), not 62-D delta-base
  * tactile = [8, 3] Allegro contact forces (left then right)
  * cameras = ego + wrist_left + wrist_right (ego used as slow/head view)
  * no deform images
"""

from __future__ import annotations

import copy
import json
import os
from typing import Dict, List, Optional

import numpy as np
import PIL.Image
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from adapters.compute_norm_stats import build_action_chunks, load_actions
from adapters.dexjoco_schema import (
    ACTION_CHUNK,
    ACTION_DIM,
    DEFAULT_INSTRUCTION,
    F6_DIM,
    F6_PER_FINGER,
    N_FINGERS,
    SRC_EGO,
    SRC_WRIST_L,
    SRC_WRIST_R,
    STATS_KEY,
)
from adapters.prep_vqvae_data import _list_to_arr, forces_to_tactile_f6


def _normalize(values, mask, vmin, vmax):
    return np.where(
        mask,
        np.clip(2 * (values - vmin) / (vmax - vmin + 1e-8) - 1, -1, 1),
        values,
    )


class DexJoCoSftDataset(torch.utils.data.Dataset):
    def __init__(self, config, processor, accelerator, _indices=None):
        self.config = config
        self.processor = processor
        self.accelerator = accelerator

        root = config.lerobot_root
        self.root = root
        force_path = config.force_labels_path
        stats_path = config.norm_stats_path
        self.instruction = getattr(
            config, "language_instruction", None
        ) or DEFAULT_INSTRUCTION

        with open(stats_path) as f:
            self.stats_data = json.load(f)
        block = self.stats_data[STATS_KEY]

        def _arr(key, sub):
            return np.array(block[key][sub])

        self.action_mask = _arr("action", "mask")
        self.action_min = _arr("action", "q01")
        self.action_max = _arr("action", "q99")
        self.tacf6_mask = _arr("tactile_f6", "mask")
        self.tacf6_min = _arr("tactile_f6", "q01")
        self.tacf6_max = _arr("tactile_f6", "q99")

        self.image_size = (
            tuple(config.image_size) if getattr(config, "image_size", None) else None
        )
        self.use_flare = bool(getattr(config, "use_flare", 0))
        self.n_flare_steps = (
            int(getattr(config, "n_flare_steps", 0)) if self.use_flare else 0
        )
        self.flare_stride = int(getattr(config, "flare_frame_stride", 1))
        self.use_tactile_vec = bool(getattr(config, "use_tactile_vec", 0))
        self.use_tactile_vqvae = bool(getattr(config, "use_tactile_vqvae", 0))
        self.vqvae_window = int(getattr(config, "vqvae_window", 16))
        self.action_dim = int(getattr(config, "action_dim", ACTION_DIM))

        # Cache actions + forces in frame order
        self.actions, self.ep_index = load_actions(root)
        ft = pq.read_table(force_path)
        if ft.num_rows != self.actions.shape[0]:
            raise ValueError(
                f"force rows {ft.num_rows} != action rows {self.actions.shape[0]}"
            )
        left = _list_to_arr(ft["left_finger_force"].to_pylist(), ft.num_rows, 12)
        right = _list_to_arr(ft["right_finger_force"].to_pylist(), ft.num_rows, 12)
        self.tactile = forces_to_tactile_f6(left, right)  # [N,8,3]
        self.chunks = build_action_chunks(self.actions, self.ep_index)

        # Per-episode frame ranges for history / flare
        self._ep_starts = {}
        self._ep_ends = {}
        for i, e in enumerate(self.ep_index):
            e = int(e)
            if e not in self._ep_starts:
                self._ep_starts[e] = i
            self._ep_ends[e] = i + 1

        # LeRobot for cameras (lazy)
        repo_id = getattr(config, "lerobot_repo_id", "") or os.path.basename(
            root.rstrip("/")
        )
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        video_backend = getattr(config, "video_backend", None) or None
        if video_backend == "":
            video_backend = None
        self.ds = LeRobotDataset(
            repo_id, root=root, episodes=None, video_backend=video_backend
        )
        if len(self.ds) != self.actions.shape[0]:
            accelerator.print(
                f"[DexJoCo] warn: LeRobot len {len(self.ds)} != actions "
                f"{self.actions.shape[0]}"
            )

        n = len(self.ds)
        self._indices = list(range(n)) if _indices is None else list(_indices)
        accelerator.print(
            f"[DexJoCo] {n} frames, action={self.action_dim}, "
            f"tactile=[{N_FINGERS},{F6_PER_FINGER}], samples={len(self._indices)}"
        )

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, i):
        return self._indices[i]  # global frame index

    def create_val_split(self, val_ratio=0.05, seed=42):
        eps = sorted(self._ep_starts.keys())
        rng = np.random.RandomState(seed)
        perm = rng.permutation(eps)
        n_val = max(1, int(len(eps) * val_ratio))
        val_eps = set(perm[:n_val].tolist())
        train_idx, val_idx = [], []
        for i in range(len(self.ds)):
            if int(self.ep_index[i]) in val_eps:
                val_idx.append(i)
            else:
                train_idx.append(i)
        val = copy.copy(self)
        val._indices = val_idx
        self._indices = train_idx
        self.accelerator.print(
            f"[DexJoCo] train/val episodes: {len(eps) - n_val}/{n_val}"
        )
        return val

    def _img_to_pil(self, img_t: torch.Tensor) -> PIL.Image.Image:
        arr = (img_t.detach().float().clamp(0, 1) * 255.0).to(torch.uint8)
        arr = arr.permute(1, 2, 0).cpu().numpy()
        img = PIL.Image.fromarray(arr, mode="RGB")
        if self.image_size is not None:
            img = img.resize(self.image_size, PIL.Image.LANCZOS)
        return img

    def _f6_history(self, global_i: int) -> np.ndarray:
        W = self.vqvae_window
        ep = int(self.ep_index[global_i])
        start = self._ep_starts[ep]
        local = global_i - start
        hist = []
        for k in range(W):
            j = start + max(0, local - (W - 1 - k))
            hist.append(self.tactile[j])
        return np.stack(hist, axis=0).astype(np.float32)

    def _flare_pil(self, global_i: int, k: int):
        ep = int(self.ep_index[global_i])
        end = self._ep_ends[ep]
        j = min(global_i + (k + 1) * self.flare_stride, end - 1)
        item = self.ds[j]
        return self._img_to_pil(item[SRC_EGO])

    def collate_fn(self, batch_indices: List[int]) -> Dict:
        cfg = self.config
        B = len(batch_indices)
        items = [self.ds[i] for i in batch_indices]

        actions = np.stack([self.chunks[i] for i in batch_indices], axis=0)
        norm_actions = torch.tensor(
            _normalize(actions, self.action_mask, self.action_min, self.action_max),
            dtype=torch.bfloat16,
        )
        beta = torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.0))
        time = (beta.sample((B,)) * 0.999 + 0.001).to(torch.bfloat16)
        t_ = time[:, None, None]
        noise = torch.randn_like(norm_actions)
        x_t = t_ * noise + (1 - t_) * norm_actions
        u_t = noise - norm_actions
        time_r = (beta.sample((B,)) * 0.999 + 0.001).to(torch.bfloat16)
        eps_r = torch.randn_like(norm_actions)

        norm_tacf6 = None
        tactile_f6_history_tensor = None
        if self.use_tactile_vec or self.use_tactile_vqvae:
            hist = np.stack(
                [self._f6_history(i) for i in batch_indices], axis=0
            )  # [B,W,8,3]
            if self.use_tactile_vqvae:
                tactile_f6_history_tensor = torch.from_numpy(hist.astype(np.float32))
            if self.use_tactile_vec:
                cur = hist[:, -1].reshape(B, F6_DIM)
                norm_tacf6 = torch.tensor(
                    _normalize(
                        cur, self.tacf6_mask, self.tacf6_min, self.tacf6_max
                    ).reshape(B, N_FINGERS, F6_PER_FINGER),
                    dtype=torch.bfloat16,
                )

        all_input_ids, all_pixel_values, all_grid_thw = [], [], []
        n_slow_images = 1
        for item in items:
            pil_slow = [self._img_to_pil(item[SRC_EGO])]
            pil_fast = [
                self._img_to_pil(item[SRC_WRIST_L]),
                self._img_to_pil(item[SRC_WRIST_R]),
            ]
            all_pil = pil_slow + pil_fast
            content = [{"type": "image"} for _ in pil_slow]
            content.append({"type": "text", "text": self.instruction})
            content.extend({"type": "image"} for _ in pil_fast)
            messages = [{"role": "user", "content": content}]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inp = self.processor(
                text=text, images=all_pil, return_tensors="pt", padding=False
            )
            all_input_ids.append(inp.input_ids[0])
            all_pixel_values.append(inp.pixel_values)
            all_grid_thw.append(inp.image_grid_thw)

        flare_pixel_values = flare_grid_thw = None
        if self.n_flare_steps > 0:
            flare_pils = []
            for gi in batch_indices:
                for k in range(self.n_flare_steps):
                    flare_pils.append(self._flare_pil(gi, k))
            flare_inp = self.processor.image_processor(
                flare_pils, return_tensors="pt"
            )
            flare_pixel_values = flare_inp.pixel_values.to(torch.bfloat16)
            flare_grid_thw = flare_inp.image_grid_thw

        pad_id = self.processor.tokenizer.pad_token_id or 0
        max_len = max(ids.shape[0] for ids in all_input_ids)
        padded_ids, attention_ms = [], []
        for ids in all_input_ids:
            pad_len = max_len - ids.shape[0]
            padded_ids.append(F.pad(ids, (pad_len, 0), value=pad_id))
            attn = torch.ones(max_len, dtype=torch.long)
            if pad_len > 0:
                attn[:pad_len] = 0
            attention_ms.append(attn)

        return {
            "input_ids": torch.stack(padded_ids),
            "attention_mask": torch.stack(attention_ms),
            "pixel_values": torch.cat(all_pixel_values, dim=0),
            "image_grid_thw": torch.cat(all_grid_thw, dim=0),
            "n_slow_images": n_slow_images,
            "noisy_actions": x_t,
            "target": u_t,
            "timesteps": time,
            "norm_actions": norm_actions,
            "tactile_f6s": norm_tacf6,
            "tactile_deforms": None,
            "tactile_f6s_delayed": norm_tacf6,
            "tactile_deforms_delayed": None,
            "tactile_codes": None,
            "tactile_f6_history": tactile_f6_history_tensor,
            "time_r": time_r,
            "eps_r": eps_r,
            "state_raw": None,
            "flare_pixel_values": flare_pixel_values,
            "flare_grid_thw": flare_grid_thw,
        }
