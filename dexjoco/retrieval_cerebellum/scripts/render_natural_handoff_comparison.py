"""Render fixed-axis versus intent-conditioned natural-handoff execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio
import numpy as np

from dexjoco.tasks import CONFIG_MAPPING
from dexjoco.tasks.state_restorers import restore_initial_state
from dexquery.data.episode_replay import make_assembly_env
from dexquery.data.zarr_io import discover_zarr_demos, load_zarr_episode
from retrieval_cerebellum.assembly_kinematics import (
    apply_bimanual_wrist_twists,
    pose_from_action44,
    world_wrist_twist,
)
from retrieval_cerebellum.geometry_labels import PrivilegedGeometryLabeler
from retrieval_cerebellum.learning_data import state46_to_action44
from retrieval_cerebellum.scripts.run_natural_handoff_baseline import (
    DEFAULT_HANDOFFS,
    DEFAULT_ZARR,
    _handoff_intent_axis,
    _state46,
    _step,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr-input-dir", type=Path, default=DEFAULT_ZARR)
    parser.add_argument("--handoff-summary", type=Path, default=DEFAULT_HANDOFFS)
    parser.add_argument("--episode", type=int, default=31)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--descent-step-mm", type=float, default=0.5)
    parser.add_argument("--intent-horizon", type=int, default=12)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _overlay(
    rgb: np.ndarray,
    *,
    title: str,
    step: int,
    success: bool,
    lateral_error_m: float,
    axis_error_rad: float,
    approach_height_m: float,
) -> np.ndarray:
    image = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    color = (80, 220, 80) if success else (80, 180, 255)
    lines = (
        "POST-HANDOFF ONLY - NO HOLE SEARCH",
        title,
        f"step={step}  success={success}",
        f"eval lateral={lateral_error_m * 1000:.2f} mm",
        f"eval tilt={axis_error_rad:.3f} rad",
        f"eval height={approach_height_m * 1000:.1f} mm",
    )
    cv2.rectangle(image, (0, 0), (image.shape[1], 154), (0, 0, 0), -1)
    for row, text in enumerate(lines):
        cv2.putText(
            image,
            text,
            (14, 26 + 24 * row),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (80, 80, 255) if row == 0 else color if row == 1 else (240, 240, 240),
            2,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _rollout(
    *,
    mode: str,
    episode: int,
    handoff_frame: int,
    actions44: np.ndarray,
    initial_state: np.ndarray,
    steps: int,
    descent_step_m: float,
    intent_horizon: int,
) -> tuple[list[np.ndarray], dict[str, object]]:
    axis = _handoff_intent_axis(actions44, handoff_frame, intent_horizon)
    env = make_assembly_env(seed=episode, randomize=False, render_mode="rgb_array")
    raw_env = env.unwrapped
    evaluator = PrivilegedGeometryLabeler(raw_env)
    frames = []
    success = False
    exit_reason = "horizon_exhausted"
    try:
        env.reset()
        restore_initial_state(
            env,
            "bimanual_assembly",
            CONFIG_MAPPING["bimanual_assembly"](),
            initial_state,
        )
        evaluator.reset_reference(raw_env)
        for action44 in actions44[: handoff_frame + 1]:
            _step(raw_env, action44)
        finger_reference = state46_to_action44(_state46(raw_env))
        initial_truth = evaluator.compute(raw_env)
        frames.append(
            _overlay(
                raw_env._compute_observation()["images"]["ego"],
                title=mode,
                step=0,
                success=False,
                lateral_error_m=initial_truth.lateral_error_m,
                axis_error_rad=initial_truth.axis_error_rad,
                approach_height_m=initial_truth.approach_height_m,
            )
        )
        for step_index in range(steps):
            current_action = state46_to_action44(_state46(raw_env))
            if mode == "pi0.5 action-chunk intent proxy":
                intent_row = handoff_frame + step_index
                if intent_row + 1 >= len(actions44):
                    exit_reason = "intent_exhausted"
                    break
                right_twist = world_wrist_twist(
                    pose_from_action44(actions44[intent_row], side="right"),
                    pose_from_action44(actions44[intent_row + 1], side="right"),
                )
                left_twist = world_wrist_twist(
                    pose_from_action44(actions44[intent_row], side="left"),
                    pose_from_action44(actions44[intent_row + 1], side="left"),
                )
                command = apply_bimanual_wrist_twists(
                    current_action,
                    right_twist,
                    left_twist,
                    finger_reference44=actions44[intent_row + 1],
                )
            else:
                twist = np.concatenate([descent_step_m * axis, np.zeros(3)])
                command = apply_bimanual_wrist_twists(
                    current_action,
                    0.8 * twist,
                    -0.2 * twist,
                    finger_reference44=finger_reference,
                )
            _step(raw_env, command)
            truth = evaluator.compute(raw_env)
            success = bool(truth.insert_ok)
            frames.append(
                _overlay(
                    raw_env._compute_observation()["images"]["ego"],
                    title=mode,
                    step=step_index + 1,
                    success=success,
                    lateral_error_m=truth.lateral_error_m,
                    axis_error_rad=truth.axis_error_rad,
                    approach_height_m=truth.approach_height_m,
                )
            )
            if success:
                exit_reason = "external_success_evaluator"
                break
        return frames, {
            "mode": mode,
            "success": success,
            "exit_reason": exit_reason,
            "frames": len(frames),
        }
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    summary = json.loads(args.handoff_summary.read_text())
    record = next(
        item for item in summary["records"] if int(item["episode_index"]) == args.episode
    )
    handoff_frame = int(record["handoff_frame"])
    paths = discover_zarr_demos(args.zarr_input_dir)
    actions44, initial_state = load_zarr_episode(paths[args.episode])
    if initial_state is None:
        raise ValueError("accurate replay requires initial_state")
    fixed_frames, fixed_result = _rollout(
        mode="fixed-axis guarded descent",
        episode=args.episode,
        handoff_frame=handoff_frame,
        actions44=actions44,
        initial_state=initial_state,
        steps=args.steps,
        descent_step_m=args.descent_step_mm / 1000.0,
        intent_horizon=args.intent_horizon,
    )
    intent_frames, intent_result = _rollout(
        mode="pi0.5 action-chunk intent proxy",
        episode=args.episode,
        handoff_frame=handoff_frame,
        actions44=actions44,
        initial_state=initial_state,
        steps=args.steps,
        descent_step_m=args.descent_step_mm / 1000.0,
        intent_horizon=args.intent_horizon,
    )
    frame_count = max(len(fixed_frames), len(intent_frames))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        args.output,
        fps=args.fps,
        codec="libx264",
        pixelformat="yuv420p",
        ffmpeg_params=["-movflags", "+faststart"],
    )
    try:
        for index in range(frame_count):
            left = fixed_frames[min(index, len(fixed_frames) - 1)]
            right = intent_frames[min(index, len(intent_frames) - 1)]
            writer.append_data(np.concatenate([left, right], axis=1))
    finally:
        writer.close()
    report = {
        "episode": args.episode,
        "handoff_frame": handoff_frame,
        "video": str(args.output),
        "scope": "post_handoff_insertion_execution_only",
        "contains_hole_search": False,
        "contains_online_pi05_rollout": False,
        "intent_proxy_is_recorded_same_episode_continuation": True,
        "fixed_axis": fixed_result,
        "intent_proxy": intent_result,
        "teacher_used_by_controller": False,
        "evaluation_overlay_uses_teacher": True,
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
