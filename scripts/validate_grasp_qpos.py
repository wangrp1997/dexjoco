#!/usr/bin/env python3
"""Phase-1: DITTO-warped single-demo approach→grasp at rand_obj layout."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Headless server: avoid GLFW / XDG display errors when recording video.
os.environ.setdefault("MUJOCO_GL", "egl")

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_DEXJOJO_ROOT = _REPO_ROOT / "dexjoco"
if str(_DEXJOJO_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEXJOJO_ROOT))

from interaction_retarget.constants import TASK_ID, default_sidecar_dir
from interaction_retarget.grasp.demo_frame_replay import extract_demo_warp_tracks
from interaction_retarget.grasp.qpos_pipeline import run_bimanual_demo_warp_grasp
from interaction_retarget.io.zarr_io import load_zarr_episode
from interaction_retarget.sim.contact import AssemblyContactDetector
from interaction_retarget.sim.replay import make_assembly_env, raw_flat_to_dict
from interaction_retarget.sim.video import (
    DexEnvVideoRecorder,
    exec_recording,
    maybe_capture_frame,
    reset_sim_step,
)
from interaction_retarget.tpsr.config import TpsrConfig
from dexjoco.tasks import CONFIG_MAPPING
from dexjoco.tasks.state_restorers import has_restorer, restore_initial_state


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sidecar-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0, help="env.reset seed → peg/tray XY (rand_obj)")
    p.add_argument(
        "--rand-visual",
        action="store_true",
        help="Also randomize camera/lighting (NOT needed for rand_obj eval)",
    )
    p.add_argument("--ep", type=int, default=35, help="Demo episode index (zarr + timing from manifest)")
    p.add_argument("--video", action="store_true", help="Save algorithm mp4")
    p.add_argument(
        "--video-dir",
        type=Path,
        default=_REPO_ROOT / "outputs" / "qpos_replay",
    )
    p.add_argument(
        "--record-demo",
        action="store_true",
        help="Privileged demo replay only → outputs/demo_replay/",
    )
    p.add_argument("--demo-dir", type=Path, default=_REPO_ROOT / "outputs" / "demo_replay")
    p.add_argument("--no-squeeze", action="store_true")
    p.add_argument("--no-lift", action="store_true")
    p.add_argument("--strict", action="store_true", help="Exit 1 if lift fails (default: save video anyway)")
    return p.parse_args()


def _manifest_entry(sidecar_dir: Path, episode_index: int) -> dict:
    import json

    manifest = json.loads((sidecar_dir / "manifest.json").read_text(encoding="utf-8"))
    for e in manifest["episodes"]:
        if int(e["episode_index"]) == int(episode_index):
            return e
    raise KeyError(f"episode {episode_index} not in manifest")


def _record_demo_video(env, entry: dict, out_path: Path) -> None:
    actions, _, initial_state = load_zarr_episode(Path(entry["zarr_path"]))
    config = CONFIG_MAPPING["bimanual_assembly"]()
    raw = env.unwrapped
    env.reset()
    if initial_state is not None and has_restorer("bimanual_assembly"):
        restore_initial_state(env, "bimanual_assembly", config, initial_state)
    rec = DexEnvVideoRecorder(env, out_path)
    reset_sim_step()
    prime = getattr(raw, "_prime_rgb_array_renderer", None)
    if callable(prime):
        prime()
    with exec_recording(rec.capture):
        try:
            rec.capture()
        except RuntimeError:
            pass
        for action in actions:
            raw.step(raw_flat_to_dict(action))
            maybe_capture_frame()
    rec.close()
    print(f"demo video: {rec.out_path} ({rec.frame_count} frames, {rec.duration_s:.1f}s)")


def main() -> None:
    args = _parse_args()
    ep = int(args.ep)
    sidecar_dir = args.sidecar_dir if args.sidecar_dir is not None else default_sidecar_dir(TASK_ID)
    entry = _manifest_entry(sidecar_dir, ep)

    # Offline extract before any render env — extra MuJoCo envs break EGL recording.
    print(f"extract warp tracks from ep{ep} ...")
    warp_tracks = extract_demo_warp_tracks(entry)

    if args.record_demo:
        env_d = make_assembly_env(seed=args.seed, randomize=False)
        try:
            out = Path(args.demo_dir) / f"ep{ep}" / "demo.mp4"
            _record_demo_video(env_d, entry, out)
        finally:
            env_d.close()
        if not args.video:
            return

    env = make_assembly_env(seed=args.seed, randomize=args.rand_visual)
    raw = env.unwrapped
    detector = AssemblyContactDetector(raw)
    tpsr_cfg = TpsrConfig(require_qp_fc=False)

    tag = f"ep{ep}_seed{args.seed}"
    rec: DexEnvVideoRecorder | None = None
    if args.video:
        out = Path(args.video_dir) / tag / "algorithm.mp4"
        rec = DexEnvVideoRecorder(env, out)

    try:
        env.reset()
        detector.reset_reference(raw)
        if rec is not None:
            reset_sim_step()
            prime = getattr(raw, "_prime_rgb_array_renderer", None)
            if callable(prime):
                prime()

        def _run():
            nonlocal report
            report = run_bimanual_demo_warp_grasp(
                raw,
                entry,
                sidecar_dir=sidecar_dir,
                detector=detector,
                tracks=warp_tracks,
                tpsr_cfg=tpsr_cfg,
                do_squeeze=not args.no_squeeze,
                do_lift=not args.no_lift,
            )

        report = None
        if rec is not None:
            with exec_recording(rec.capture):
                try:
                    rec.capture()
                except RuntimeError:
                    pass
                _run()
            rec.close()
            print(f"algorithm video: {rec.out_path} ({rec.frame_count} frames, {rec.duration_s:.1f}s)")
        else:
            _run()

        assert report is not None
        t, p = report.tray, report.peg
        print(
            f"[{tag}] tray cc={t.refine.contact_count} rmse={t.refine.contact_rmse_m * 1e3:.1f}mm "
            f"peg cc={p.refine.contact_count} rmse={p.refine.contact_rmse_m * 1e3:.1f}mm"
        )
        print(
            f"[{tag}] lift tray={report.tray_lift_m * 1e3:.1f}mm peg={report.peg_lift_m * 1e3:.1f}mm "
            f"success={report.success}"
        )
        if not report.success and args.strict:
            raise SystemExit(1)
    finally:
        env.close()


if __name__ == "__main__":
    main()
