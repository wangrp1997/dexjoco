#!/usr/bin/env python3
"""Export hybrid_insert PBVS insert-phase trajectories (handoff -> openpi success) for FT."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import imageio
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

_REPO = Path(__file__).resolve().parents[1]
_LAI = Path("/home/wangrenpeng/lai")
for p in (_REPO, _REPO / "dexjoco", _REPO / "scripts", _LAI):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from hybrid_insert.assembly_contacts import AssemblyContactLabeler
from hybrid_insert.controller import HybridInsertController
from interaction_retarget.io.zarr_io import load_zarr_episode
from interaction_retarget.sim.replay import make_assembly_env, rotvec_dual_arm_to_policy
from interaction_retarget.sim.settle import read_arm_action
from interaction_retarget.skill_replay.insert import (
    dual_arm23_to_action44,
    demo_replay_to_pre_insert,
)
from pose_insert.pre_insert import resolve_peg_lift_end_frame

from eval_hybrid_openpi_success import (  # noqa: E402
    APPROACH_CAP,
    MAX_STEPS,
    SIDECAR,
    _approach,
    _hybrid_config,
    _manifest,
)

PROMPT = (
    "Grasp the tray with the left hand and the peg with the right hand, "
    "then insert the peg into the hole."
)
CAMERA_NAMES = ("ego", "wrist_left", "wrist_right")
FPS = 30


def _state46(raw) -> np.ndarray:
    state = raw._compute_observation()["state"]
    return np.concatenate(
        [np.asarray(state["tcp_pose"]).ravel(), np.asarray(state["gripper_pose"]).ravel()],
        dtype=np.float64,
    )


class _SegmentRecorder:
    def __init__(self, raw, out_dir: Path, *, force_labeler=None) -> None:
        self._raw = raw
        self._force_labeler = force_labeler
        self._writers = {
            name: imageio.get_writer(out_dir / f"{name}.mp4", fps=FPS)
            for name in CAMERA_NAMES
        }
        self.states: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.insert_ok: list[bool] = []
        self.peg_ok: list[bool] = []
        self.phases: list[str] = []
        self.wrist_ft_right: list[np.ndarray] = []
        self.wrist_ft_left: list[np.ndarray] = []
        self.right_finger_force: list[np.ndarray] = []
        self.left_finger_force: list[np.ndarray] = []

    @property
    def records_force(self) -> bool:
        return self._force_labeler is not None

    def capture(self, *, action44: np.ndarray, phase: str, labeler: AssemblyContactLabeler) -> None:
        frames = self._raw.render()
        for name, frame in zip(CAMERA_NAMES, frames, strict=True):
            arr = np.asarray(frame)
            if arr.dtype != np.uint8:
                if arr.max() <= 1.0:
                    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
                else:
                    arr = arr.astype(np.uint8)
            self._writers[name].append_data(arr)
        outcome = labeler.compute(self._raw)
        self.states.append(_state46(self._raw).astype(np.float32))
        self.actions.append(np.asarray(action44, dtype=np.float32).reshape(44))
        self.insert_ok.append(bool(outcome.insert_ok))
        self.peg_ok.append(bool(outcome.peg_ok))
        self.phases.append(str(phase))
        if self._force_labeler is not None:
            force = self._force_labeler.compute(self._raw)
            self.wrist_ft_right.append(np.asarray(force.wrist_ft_right, dtype=np.float32).reshape(6))
            self.wrist_ft_left.append(np.asarray(force.wrist_ft_left, dtype=np.float32).reshape(6))
            self.right_finger_force.append(
                np.asarray(force.right_finger_force, dtype=np.float32).reshape(12)
            )
            self.left_finger_force.append(
                np.asarray(force.left_finger_force, dtype=np.float32).reshape(12)
            )

    def close(self) -> int:
        n = len(self.states)
        for w in self._writers.values():
            w.close()
        return n

    def save_npz(self, path: Path) -> None:
        phase_labels = sorted(set(self.phases))
        phase_codes = {p: i for i, p in enumerate(phase_labels)}
        payload: dict = {
            "observation_state": np.stack(self.states, axis=0),
            "action": np.stack(self.actions, axis=0),
            "insert_ok": np.asarray(self.insert_ok, dtype=bool),
            "peg_ok": np.asarray(self.peg_ok, dtype=bool),
            "phase": np.asarray([phase_codes[p] for p in self.phases], dtype=np.int16),
            "phase_labels": np.asarray(phase_labels, dtype=object),
            "frame_index": np.arange(len(self.states), dtype=np.int64),
            "fps": np.int32(FPS),
        }
        if self._force_labeler is not None:
            if len(self.wrist_ft_right) != len(self.states):
                raise RuntimeError(
                    f"force/state length mismatch: {len(self.wrist_ft_right)} vs {len(self.states)}"
                )
            wrist_r = np.stack(self.wrist_ft_right, axis=0)
            wrist_l = np.stack(self.wrist_ft_left, axis=0)
            finger_r = np.stack(self.right_finger_force, axis=0)
            finger_l = np.stack(self.left_finger_force, axis=0)
            payload.update(
                {
                    "wrist_ft_right": wrist_r,
                    "wrist_ft_left": wrist_l,
                    "right_finger_force": finger_r,
                    "left_finger_force": finger_l,
                    # ForceVLA both = wrist12 + finger24
                    "force": np.concatenate([wrist_r, wrist_l, finger_r, finger_l], axis=1),
                }
            )
        np.savez(path, **payload)


def export_episode(
    entry: dict,
    *,
    seed: int,
    out_root: Path,
    success_only: bool = False,
) -> dict:
    ep = int(entry["episode_index"])
    peg_lift_end = resolve_peg_lift_end_frame(entry, SIDECAR)
    env = make_assembly_env(seed=seed, randomize=False, render_mode="rgb_array")
    raw = env.unwrapped
    labeler = AssemblyContactLabeler(raw)
    temp_dir = out_root / f"episode_{ep:02d}_temp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    recorder: _SegmentRecorder | None = None
    try:
        _, _, initial_state = load_zarr_episode(Path(entry["zarr_path"]))
        demo_replay_to_pre_insert(
            env,
            raw,
            zarr_path=entry["zarr_path"],
            stop_frame=int(peg_lift_end),
            initial_state=initial_state,
            video_cb=None,
            labeler=labeler,
        )
        handoff_step = int(getattr(raw, "env_step", 0))

        left_hold = np.asarray(read_arm_action(raw, "left"), dtype=np.float64).copy()
        right0 = np.asarray(read_arm_action(raw, "right"), dtype=np.float64).copy()
        right_hand_hold = right0[7:23].copy()

        fail = _approach(env, raw, labeler, left_hold, right_hand_hold)
        if fail:
            return {
                "episode_index": ep,
                "openpi_success": False,
                "fail_reason": fail,
                "num_frames": 0,
            }
        if not bool(labeler.compute(raw).peg_ok):
            return {
                "episode_index": ep,
                "openpi_success": False,
                "fail_reason": "peg_lost_before_hybrid",
                "num_frames": 0,
            }

        insert_budget = MAX_STEPS - int(getattr(raw, "env_step", 0))
        hctrl = HybridInsertController(_hybrid_config(insert_budget))
        hctrl.reset(raw)
        hctrl._peg_rest_z = float(labeler._peg_rest_z)  # noqa: SLF001
        if hctrl._labeler is not None:
            hctrl._labeler._tray_rest_z = float(labeler._tray_rest_z)  # noqa: SLF001
            hctrl._labeler._peg_rest_z = float(labeler._peg_rest_z)  # noqa: SLF001
        left = read_arm_action(raw, "left")
        right = read_arm_action(raw, "right")
        policy44 = dual_arm23_to_action44(left, right)
        hctrl._activate(policy44, raw)  # noqa: SLF001

        recorder = _SegmentRecorder(raw, temp_dir)

        openpi_success = False
        streak = 0
        max_streak = 0
        fail_reason = ""
        for _ in range(insert_budget):
            left = read_arm_action(raw, "left")
            right = read_arm_action(raw, "right")
            policy44 = dual_arm23_to_action44(left, right)
            action44 = hctrl.merge_right_arm(raw, policy44)
            recorder.capture(action44=action44, phase=hctrl.phase_name, labeler=labeler)
            action46 = rotvec_dual_arm_to_policy(np.asarray(action44, dtype=np.float64).reshape(44))
            _, _, terminated, truncated, info = env.step(action46)
            outcome = labeler.compute(raw)
            if bool(outcome.insert_ok):
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
            if bool(info.get("succeed", False)):
                openpi_success = True
                break
            if terminated or truncated:
                openpi_success = bool(info.get("succeed", False))
                if not openpi_success:
                    fail_reason = "env_done_no_success"
                break
        else:
            fail_reason = fail_reason or "max_steps"

        n_frames = recorder.close()
        if not openpi_success:
            if success_only:
                shutil.rmtree(temp_dir, ignore_errors=True)
                return {
                    "episode_index": ep,
                    "openpi_success": False,
                    "fail_reason": fail_reason,
                    "max_insert_streak": max_streak,
                    "num_frames": 0,
                }
        tag = "success" if openpi_success else "failure"
        final_dir = out_root / f"episode_{ep:02d}_{tag}"
        if final_dir.exists():
            shutil.rmtree(final_dir)
        temp_dir.rename(final_dir)
        recorder.save_npz(final_dir / "trajectory.npz")
        meta = {
            "episode_index": ep,
            "openpi_success": bool(openpi_success),
            "fail_reason": fail_reason,
            "max_insert_streak": int(max_streak),
            "num_frames": int(n_frames),
            "handoff_env_step": int(handoff_step),
            "segment": "hybrid_pbvs_insert_only",
            "prompt": PROMPT,
            "observation_state_dim": 46,
            "action_dim": 44,
            "camera_names": list(CAMERA_NAMES),
            "fps": FPS,
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }
        (final_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return {
            "episode_index": ep,
            "openpi_success": bool(openpi_success),
            "fail_reason": fail_reason,
            "max_insert_streak": max_streak,
            "num_frames": n_frames,
            "output_dir": str(final_dir),
        }
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        env.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, nargs="*", default=None)
    p.add_argument("--all", action="store_true", help="Export every manifest episode.")
    p.add_argument("--try-until-success", action="store_true")
    p.add_argument(
        "--success-only",
        action="store_true",
        help="Only write episode folders for openpi-success trajectories.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/mnt/hdd/dexjoco/outputs/hybrid_insert_ft_smoke"),
    )
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.all:
        episodes = None
    elif args.try_until_success:
        episodes = None
    else:
        episodes = args.episodes if args.episodes is not None else [0]
    entries = _manifest(episodes)
    if args.try_until_success:
        entries = entries[:30]
    results = []
    for entry in entries:
        ep = int(entry["episode_index"])
        print(f"[export] ep={ep}", flush=True)
        row = export_episode(
            entry,
            seed=args.seed,
            out_root=args.out_dir,
            success_only=bool(args.success_only),
        )
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if args.try_until_success and row.get("openpi_success"):
            break
    n_ok = sum(1 for r in results if r.get("openpi_success"))
    summary = {
        "protocol": "hybrid_insert_ft_pbvs_segment",
        "openpi_eval": "30step_insert_contact",
        "success_only": bool(args.success_only),
        "n_ok": n_ok,
        "n_total": len(results),
        "results": results,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if n_ok:
        (args.out_dir / f"success_rate_{n_ok}_{len(results)}.txt").touch()
    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
