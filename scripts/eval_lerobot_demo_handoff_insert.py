#!/usr/bin/env python3
"""Evaluate LeRobot ACT/Diffusion after replaying each demo to hybrid handoff."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
for _proxy in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(_proxy, None)

_REPO = Path(__file__).resolve().parents[1]
_DEXJOCO = _REPO / "dexjoco"
for path in (_REPO, _DEXJOCO, _REPO / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hybrid_insert.assembly_contacts import AssemblyContactLabeler
from hybrid_insert.integration import get_raw_env, state_to_dual_arm_action44
from interaction_retarget.constants import default_sidecar_dir
from interaction_retarget.io.zarr_io import load_zarr_episode
from interaction_retarget.sim.settle import read_arm_action
from interaction_retarget.skill_replay.insert import demo_replay_to_pre_insert
from lerobot.async_inference.configs import RobotClientConfig
from lerobot.transport import services_pb2
from pose_insert.pre_insert import resolve_peg_lift_end_frame

from dexjoco_lerobot_client.async_observation_robot_client import AsyncObservationRobotClient
from dexjoco_lerobot_client.config_dexjoco_robot import DexJoCoRobotConfig  # noqa: F401
from dexjoco_lerobot_client.dexjoco_robot import DexJoCoRobot, _observation_dict_to_state46
from dexjoco_lerobot_client.eval_config import (
    action_dim_from_checkpoint,
    load_eval_yaml,
    resolve_actions_per_chunk,
    write_robot_config_yaml,
)
from dexjoco_lerobot_client.eval_dexjoco_lerobot import reset_client_runtime_state
from eval_hybrid_openpi_success import APPROACH_CAP
from hybrid_insert.geometry import (
    height_along_axis,
    insert_along_hole_delta,
    lateral_error,
    toward_socket_delta,
)
from interaction_retarget.sim.replay import rotvec_dual_arm_to_policy
from interaction_retarget.skill_replay.insert import (
    _insert_geometry,
    dual_arm23_to_action44,
)
from priv_snap_insert.snap import (
    apply_socket_pin,
    capture_o2h,
    capture_socket_pin,
    capture_tray_in_left,
    project_peg_to_o2h,
    project_tray_to_left,
)

EGO_KEY = "ego"


class EgoVideoRecorder:
    """Buffer RGB frames, encode with imageio+ffmpeg like OpenPI eval (h264/yuv420p)."""

    def __init__(self, path: Path, *, fps: int = 30) -> None:
        self.path = path
        self.fps = fps
        self._frames: list[np.ndarray] = []

    def write_rgb(self, frame: np.ndarray) -> None:
        rgb = np.asarray(frame)
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        if rgb.shape[:2] != (640, 640):
            rgb = cv2.resize(rgb, (640, 640), interpolation=cv2.INTER_LINEAR)
        self._frames.append(np.ascontiguousarray(rgb))

    def close(self) -> None:
        if not self._frames:
            self.path.unlink(missing_ok=True)
            return
        import imageio.v3 as iio

        iio.imwrite(
            self.path,
            np.stack(self._frames, axis=0),
            fps=self.fps,
            codec="libx264",
            plugin="pyav",
        )
        self._frames.clear()


def _ego_from_obs(obs: dict[str, Any]) -> np.ndarray | None:
    if EGO_KEY in obs:
        return np.asarray(obs[EGO_KEY])
    images = obs.get("images")
    if isinstance(images, dict) and EGO_KEY in images:
        return np.asarray(images[EGO_KEY])
    return None


def _make_video_cb(recorder: EgoVideoRecorder | None) -> Callable[[dict[str, Any]], None] | None:
    if recorder is None:
        return None

    def _cb(obs: dict[str, Any]) -> None:
        frame = _ego_from_obs(obs)
        if frame is not None:
            recorder.write_rgb(frame)

    return _cb


def _csv_ints(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one episode index")
    return values


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _manifest_entries(sidecar_dir: Path, episodes: list[int]) -> list[dict[str, Any]]:
    manifest = json.loads((sidecar_dir / "manifest.json").read_text(encoding="utf-8"))
    by_episode = {int(entry["episode_index"]): entry for entry in manifest["episodes"]}
    missing = sorted(set(episodes) - set(by_episode))
    if missing:
        raise KeyError(f"episodes missing from manifest: {missing}")
    return [by_episode[episode] for episode in episodes]


def _boost_hand_q(hand16: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return np.asarray(hand16, dtype=np.float64)
    return np.clip(np.asarray(hand16, dtype=np.float64) * float(scale), -2.0, 2.0)


def _boost_left_hold22(left22: np.ndarray, *, grip_scale: float) -> np.ndarray:
    hold = np.asarray(left22, dtype=np.float64).reshape(-1).copy()
    if hold.shape[0] != 22:
        raise ValueError(f"left hold must be 22-dim, got {hold.shape[0]}")
    hold[6:22] = _boost_hand_q(hold[6:22], grip_scale)
    return hold


def _pin_tray(raw) -> dict[str, np.ndarray]:
    pins = capture_socket_pin(raw)
    apply_socket_pin(raw, pins)
    return pins


def _freeze_left_mocap(raw, left_hold22: np.ndarray) -> None:
    """Hard-hold left mocap + Allegro targets (same as priv_snap_insert/run_p0)."""
    from interaction_retarget.sim.replay import policy_dual_arm_to_raw

    left = np.asarray(read_arm_action(raw, "left"), dtype=np.float64)
    right = np.asarray(read_arm_action(raw, "right"), dtype=np.float64)
    hold44 = dual_arm23_to_action44(left, right).copy()
    hold44[22:44] = np.asarray(left_hold22, dtype=np.float64).reshape(22)
    a46 = rotvec_dual_arm_to_policy(hold44)
    raw_dict = policy_dual_arm_to_raw(a46)
    left23 = np.asarray(raw_dict["left"], dtype=np.float64).reshape(23)
    mid = int(raw._mocap_left_id)
    raw._data.mocap_pos[mid] = left23[0:3]
    raw._data.mocap_quat[mid] = left23[3:7]
    ctrl_ids = np.asarray(raw._allegro_ctrl_ids, dtype=int)
    raw._data.ctrl[ctrl_ids[16:32]] = left23[7:23]


@dataclass
class HandoffStabilizer:
    """Tray welded to left palm + frozen left mocap; optional peg o2h snap."""

    t2h: Any
    left_hold22: np.ndarray
    tray_world_fallback: dict[str, np.ndarray]
    o2h: Any | None = None

    @classmethod
    def capture(
        cls,
        raw,
        left_hold22: np.ndarray,
        *,
        snap_peg_o2h: bool = False,
    ) -> HandoffStabilizer:
        return cls(
            t2h=capture_tray_in_left(raw),
            left_hold22=np.asarray(left_hold22, dtype=np.float64).reshape(22).copy(),
            tray_world_fallback=capture_socket_pin(raw),
            o2h=capture_o2h(raw) if snap_peg_o2h else None,
        )

    def apply(self, raw, labeler: AssemblyContactLabeler | None = None) -> None:
        _freeze_left_mocap(raw, self.left_hold22)
        project_tray_to_left(raw, self.t2h, strength=1.0)
        if self.o2h is not None:
            project_peg_to_o2h(raw, self.o2h, strength=1.0)
        if labeler is not None and not bool(labeler.compute(raw).tray_ok):
            apply_socket_pin(raw, self.tray_world_fallback)


def _approach_with_stabilizer(
    env,
    raw,
    labeler,
    left_hold,
    right_hand_hold,
    *,
    stabilizer: HandoffStabilizer | None = None,
    video_cb=None,
) -> str | None:
    best_dist = float(_insert_geometry(raw)[3])
    stall = 0
    for _ in range(APPROACH_CAP):
        if stabilizer is not None:
            stabilizer.apply(raw, labeler)
        outcome = labeler.compute(raw)
        tip, socket, hole, dist = _insert_geometry(raw)
        lat_n, _ = lateral_error(tip, socket, hole)
        along = height_along_axis(tip, socket, hole)
        if not bool(outcome.peg_ok):
            return "peg_lost_approach"
        if dist + 1e-4 < best_dist:
            best_dist = float(dist)
            stall = 0
        else:
            stall += 1
        if lat_n <= 0.010 and along <= 0.13:
            return None
        if stall >= 80 and lat_n <= 0.014 and along <= 0.14:
            return None
        right = np.asarray(read_arm_action(raw, "right"), dtype=np.float64).copy()
        right[7:23] = right_hand_hold
        if along > 0.14 or dist > 0.18:
            delta = toward_socket_delta(tip, socket, gain=0.45, max_step_m=0.0035)
        else:
            _, lat_v = lateral_error(tip, socket, hole)
            delta = -0.8 * lat_v
            if lat_n <= 0.012 and along > 0.06:
                delta = delta + insert_along_hole_delta(hole, step_m=0.0012)
            n = float(np.linalg.norm(delta))
            if n > 0.0035:
                delta = delta * (0.0035 / n)
        right[0:3] = right[0:3] + delta
        policy44 = dual_arm23_to_action44(left_hold, right)
        action46 = rotvec_dual_arm_to_policy(np.asarray(policy44, dtype=np.float64).reshape(44))
        obs = env.step(action46)[0]
        if stabilizer is not None:
            stabilizer.apply(raw, labeler)
        if video_cb is not None:
            video_cb(obs)
    return None


def _install_right_arm_only_policy(robot: DexJoCoRobot, left22: np.ndarray) -> None:
    """44-dim ckpt eval: overwrite left arm [22:44] with handoff hold."""
    hold = np.asarray(left22, dtype=np.float64).reshape(-1).copy()
    if hold.shape[0] != 22:
        raise ValueError(f"left hold must be 22-dim, got {hold.shape[0]}")
    orig_send = robot.send_action

    def send_action(action):  # type: ignore[no-untyped-def]
        keys = list(robot.action_features.keys())
        arr = np.array([float(action[k]) for k in keys], dtype=np.float64)
        arr[22:44] = hold
        patched = {k: float(arr[i]) for i, k in enumerate(keys)}
        return orig_send(patched)

    robot.send_action = send_action  # type: ignore[method-assign]
    robot._right_arm_only_hold = hold  # noqa: SLF001


def _sync_robot_after_handoff(robot: DexJoCoRobot) -> None:
    raw = get_raw_env(robot.env)
    raw_observation = raw._compute_observation()  # noqa: SLF001
    adapted = robot.env.observation(raw_observation) if hasattr(robot.env, "observation") else raw_observation
    robot.observation = robot._process_observation(adapted)
    robot.done = False
    robot.success = False
    if robot.hybrid_insert is not None:
        robot.hybrid_insert.on_reset(robot.env)
        hold44 = state_to_dual_arm_action44(_observation_dict_to_state46(robot.observation))
        robot.hybrid_insert.observe(robot.env, hold44)


def _prepare_handoff(
    robot: DexJoCoRobot,
    entry: dict[str, Any],
    sidecar_dir: Path,
    *,
    video_cb: Callable[[dict[str, Any]], None] | None = None,
    tray_weld: bool = False,
    left_grip_scale: float = 1.0,
    snap_peg_o2h: bool = False,
) -> dict[str, Any]:
    raw = get_raw_env(robot.env)
    labeler = AssemblyContactLabeler(raw)
    peg_lift_end = resolve_peg_lift_end_frame(entry, sidecar_dir)
    _, _, initial_state = load_zarr_episode(Path(entry["zarr_path"]))
    demo_replay_to_pre_insert(
        robot.env,
        raw,
        zarr_path=entry["zarr_path"],
        stop_frame=int(peg_lift_end),
        initial_state=initial_state,
        video_cb=video_cb,
        labeler=labeler,
    )
    replay_end_step = int(getattr(raw, "env_step", 0))
    left_hold = np.asarray(read_arm_action(raw, "left"), dtype=np.float64).copy()
    right = np.asarray(read_arm_action(raw, "right"), dtype=np.float64).copy()
    approach_stabilizer: HandoffStabilizer | None = None
    if tray_weld:
        hold44 = dual_arm23_to_action44(left_hold, right)
        left22 = _boost_left_hold22(hold44[22:44], grip_scale=left_grip_scale)
        approach_stabilizer = HandoffStabilizer.capture(
            raw, left22, snap_peg_o2h=snap_peg_o2h
        )
    setup_failure = _approach_with_stabilizer(
        robot.env,
        raw,
        labeler,
        left_hold,
        right[7:23].copy(),
        stabilizer=approach_stabilizer,
        video_cb=video_cb,
    )
    outcome = labeler.compute(raw)
    _sync_robot_after_handoff(robot)
    if video_cb is not None:
        video_cb(robot.observation)
    return {
        "labeler": labeler,
        "replay_end_step": replay_end_step,
        "handoff_env_step": int(getattr(raw, "env_step", 0)),
        "setup_failure": setup_failure,
        "initial_tray_ok": bool(outcome.tray_ok),
        "initial_peg_ok": bool(outcome.peg_ok),
        "initial_insert_ok": bool(outcome.insert_ok),
    }


def _run_policy(
    client: AsyncObservationRobotClient,
    robot: DexJoCoRobot,
    labeler: AssemblyContactLabeler,
    *,
    task: str,
    ego_recorder: EgoVideoRecorder | None,
    max_policy_steps: int,
    snap_tray: bool = False,
    socket_pin: dict[str, np.ndarray] | None = None,
    stabilizer: HandoffStabilizer | None = None,
) -> dict[str, Any]:
    reset_client_runtime_state(client)
    client.stub.Ready(services_pb2.Empty())  # type: ignore[attr-defined]

    policy_steps = 0
    insert_streak = 0
    max_insert_streak = 0
    peg_lost_streak = 0
    tray_lost_streak = 0
    peg_lost = False
    tray_lost = False
    ever_insert_contact = False

    in_stay_state = False
    episode_start_time = time.time()
    client.must_go.set()
    client.control_loop_observation(task=task)

    while policy_steps < max_policy_steps and not robot.is_done:
        raw = get_raw_env(robot.env)
        if stabilizer is not None:
            stabilizer.apply(raw, labeler)
        elif snap_tray and socket_pin is not None:
            apply_socket_pin(raw, socket_pin)

        if client.actions_available():
            with client.action_queue_lock:
                action = client.action_queue.queue[0]
            if action.get_timestamp() >= episode_start_time:
                action_legal = True
            else:
                with client.action_queue_lock:
                    client.action_queue.get_nowait()
                time.sleep(client.config.environment_dt)
                continue
        else:
            action_legal = False

        if action_legal:
            client.control_loop_action()
            in_stay_state = False
            policy_steps += 1
            raw = get_raw_env(robot.env)
            if stabilizer is not None:
                stabilizer.apply(raw, labeler)
            elif snap_tray and socket_pin is not None:
                apply_socket_pin(raw, socket_pin)
            outcome = labeler.compute(raw)
            if bool(outcome.insert_ok):
                insert_streak += 1
                max_insert_streak = max(max_insert_streak, insert_streak)
                ever_insert_contact = True
            else:
                insert_streak = 0
            peg_lost_streak = 0 if outcome.peg_ok else peg_lost_streak + 1
            tray_lost_streak = 0 if outcome.tray_ok else tray_lost_streak + 1
            peg_lost = peg_lost or peg_lost_streak >= 10
            tray_lost = tray_lost or tray_lost_streak >= 10
        else:
            robot.stay(in_stay_state)
            in_stay_state = True
            time.sleep(client.config.environment_dt)

        if client._ready_to_send_observation():
            client.must_go.set()
            client.control_loop_observation(task=task)

        if ego_recorder is not None:
            obs = robot.get_observation()
            frame = obs.get(EGO_KEY)
            if frame is not None:
                ego_recorder.write_rgb(frame)

        if robot.is_done:
            break

    final_outcome = labeler.compute(get_raw_env(robot.env))
    success = bool(robot.is_success)
    if success:
        failure_reason = "success"
    elif peg_lost:
        failure_reason = "peg_lost_during_insert"
    elif tray_lost:
        failure_reason = "tray_lost_during_insert"
    elif not ever_insert_contact:
        failure_reason = "alignment_or_hole_entry_failed"
    elif max_insert_streak < 30:
        failure_reason = "unstable_or_incomplete_insertion"
    elif robot.is_done:
        failure_reason = "environment_timeout"
    else:
        failure_reason = "policy_step_budget"
    return {
        "success": success,
        "failure_reason": failure_reason,
        "policy_steps": policy_steps,
        "max_insert_streak": max_insert_streak,
        "ever_insert_contact": ever_insert_contact,
        "final_tray_ok": bool(final_outcome.tray_ok),
        "final_peg_ok": bool(final_outcome.peg_ok),
        "final_insert_ok": bool(final_outcome.insert_ok),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=_REPO / "configs/rand_obj/bimanual_assembly.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--policy-type", choices=("act", "diffusion"), required=True)
    parser.add_argument("--episodes", type=_csv_ints, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--replan-ratio", type=float, default=0.8)
    parser.add_argument("--max-policy-steps", type=int, default=900)
    parser.add_argument("--policy-device", default="cuda")
    parser.add_argument("--no-video", action="store_true", help="Skip ego mp4 recording")
    parser.add_argument(
        "--right-arm-only",
        action="store_true",
        help="Freeze left arm at handoff; only right arm follows BC policy",
    )
    parser.add_argument(
        "--snap-tray",
        action="store_true",
        help="Pin tray/socket world pose each policy step (legacy world pin)",
    )
    parser.add_argument(
        "--tray-weld-left",
        action="store_true",
        help="Weld tray to left palm + freeze left mocap from approach through policy",
    )
    parser.add_argument(
        "--snap-peg-o2h",
        action="store_true",
        help="With --tray-weld-left, also snap peg to right-hand o2h each step",
    )
    parser.add_argument(
        "--left-grip-scale",
        type=float,
        default=1.0,
        help="Scale left Allegro closure when freezing left arm (e.g. 1.12)",
    )
    parser.add_argument(
        "--no-hybrid-insert",
        action="store_true",
        help="Disable hybrid PBVS override; pure BC policy on right arm",
    )
    parser.add_argument(
        "--sidecar-dir",
        type=Path,
        default=default_sidecar_dir("bimanual_assembly"),
    )
    args = parser.parse_args()
    if args.snap_tray and args.tray_weld_left:
        raise ValueError("use either --snap-tray or --tray-weld-left, not both")

    checkpoint = args.checkpoint.expanduser().resolve()
    if not (checkpoint / "config.json").exists():
        raise FileNotFoundError(f"checkpoint must contain config.json: {checkpoint}")

    trained_action_dim = action_dim_from_checkpoint(checkpoint)
    freeze_left_arm = args.right_arm_only or trained_action_dim == 22
    right22_trained = trained_action_dim == 22
    if args.right_arm_only and trained_action_dim == 22:
        print("note: 22-dim ckpt already right-arm-only; --right-arm-only is redundant", flush=True)

    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        if args.overwrite:
            shutil.rmtree(output)
        else:
            raise FileExistsError(f"output exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    eval_cfg = load_eval_yaml(args.config)
    task = eval_cfg["prompt"]
    sidecar_dir = args.sidecar_dir.expanduser().resolve()
    entries = _manifest_entries(sidecar_dir, args.episodes)
    actions_per_chunk = resolve_actions_per_chunk(args.policy_type, checkpoint)

    with tempfile.TemporaryDirectory(prefix="dexjoco_lerobot_robot_cfg_") as tmp_dir:
        robot_cfg_path = Path(tmp_dir) / "robot.yaml"
        write_robot_config_yaml(
            eval_cfg,
            robot_cfg_path,
            action_dim=trained_action_dim,
            right_arm_action_only=right22_trained,
        )
        robot_cfg = DexJoCoRobotConfig(
            id=eval_cfg["env_name"],
            config_path=robot_cfg_path,
            seed=args.seed,
            randomize=False,
            randomize_dynamics=False,
            pad_state_dim46=False,
            render_mode="rgb_array",
        )
        robot_client_cfg = RobotClientConfig(
            policy_type=args.policy_type,
            pretrained_name_or_path=str(checkpoint),
            robot=robot_cfg,
            actions_per_chunk=actions_per_chunk,
            task=task,
            server_address=f"{args.host}:{args.port}",
            policy_device=args.policy_device,
            client_device="cpu",
            fps=30,
            aggregate_fn_name="latest_only",
            chunk_size_threshold=args.replan_ratio,
        )
        client = AsyncObservationRobotClient(robot_client_cfg)
        hybrid_mod = __import__("hybrid_insert", fromlist=["EvalHybridInsert", "HybridInsertConfig"])
        hybrid_enabled = freeze_left_arm and not args.no_hybrid_insert
        hybrid_cfg = hybrid_mod.HybridInsertConfig(freeze_left_arm_at_handoff=freeze_left_arm)
        client.robot.hybrid_insert = hybrid_mod.EvalHybridInsert(
            task=eval_cfg["env_name"], enabled=hybrid_enabled, config=hybrid_cfg
        )
        client.start()
        receiver = threading.Thread(target=client.receive_actions, daemon=True)
        receiver.start()
        client.start_barrier.wait()

        rows: list[dict[str, Any]] = []
        try:
            for index, entry in enumerate(entries, start=1):
                episode = int(entry["episode_index"])
                print(f"Episode {index}/{len(entries)} demo={episode}: handoff", flush=True)
                ep_dir = output / f"episode_{episode:02d}_temp"
                ep_dir.mkdir(parents=True, exist_ok=True)
                ego_recorder = None
                row: dict[str, Any] = {"episode_index": episode}
                if not args.no_video:
                    ego_recorder = EgoVideoRecorder(ep_dir / f"{EGO_KEY}.mp4")
                video_cb = _make_video_cb(ego_recorder)
                try:
                    prepared = _prepare_handoff(
                        client.robot,
                        entry,
                        sidecar_dir,
                        video_cb=video_cb,
                        tray_weld=args.tray_weld_left,
                        left_grip_scale=args.left_grip_scale,
                        snap_peg_o2h=args.snap_peg_o2h,
                    )
                    labeler = prepared.pop("labeler")
                    row = {"episode_index": episode, "zarr_path": str(entry["zarr_path"]), **prepared}
                    setup_failure = row.pop("setup_failure")
                    if setup_failure or not row["initial_peg_ok"]:
                        row.update(
                            {
                                "setup_ok": False,
                                "setup_failure": setup_failure or "peg_lost_before_policy",
                                "success": False,
                                "failure_reason": f"setup_failure:{setup_failure or 'peg_lost_before_policy'}",
                                "policy_steps": 0,
                                "max_insert_streak": 0,
                            }
                        )
                        print(f"  setup failed: {row['setup_failure']}", flush=True)
                    else:
                        row["setup_ok"] = True
                        row["setup_failure"] = ""
                        hold44 = state_to_dual_arm_action44(
                            _observation_dict_to_state46(client.robot.observation)
                        )
                        left_hold22 = _boost_left_hold22(
                            hold44[22:44], grip_scale=args.left_grip_scale
                        )
                        raw = get_raw_env(client.robot.env)
                        policy_stabilizer: HandoffStabilizer | None = None
                        socket_pin: dict[str, np.ndarray] | None = None
                        if args.tray_weld_left:
                            policy_stabilizer = HandoffStabilizer.capture(
                                raw, left_hold22, snap_peg_o2h=args.snap_peg_o2h
                            )
                            policy_stabilizer.apply(raw, labeler)
                        elif args.snap_tray:
                            socket_pin = _pin_tray(raw)
                        if right22_trained:
                            client.robot.set_left_arm_hold22(left_hold22)
                        elif args.right_arm_only:
                            _install_right_arm_only_policy(client.robot, left_hold22)
                        row.update(
                            _run_policy(
                                client,
                                client.robot,
                                labeler,
                                task=task,
                                ego_recorder=ego_recorder,
                                max_policy_steps=args.max_policy_steps,
                                snap_tray=args.snap_tray,
                                socket_pin=socket_pin,
                                stabilizer=policy_stabilizer,
                            )
                        )
                        print(
                            f"  {'SUCCESS' if row['success'] else 'FAIL'} steps={row['policy_steps']} "
                            f"streak={row['max_insert_streak']} reason={row['failure_reason']}",
                            flush=True,
                        )
                finally:
                    if ego_recorder is not None:
                        ego_recorder.close()
                suffix = "success" if row.get("success") else "failure"
                if args.no_video and not any(ep_dir.iterdir()):
                    ep_dir.rmdir()
                else:
                    ep_dir.rename(output / f"episode_{episode:02d}_{suffix}")
                rows.append(row)
                _atomic_json(output / f"episode_{episode:02d}.json", row)
        finally:
            client.stop()
            receiver.join(timeout=5)

    evaluable = [row for row in rows if row.get("setup_ok")]
    successes = sum(bool(row["success"]) for row in evaluable)
    summary = {
        "protocol": "demo_replay_to_peg_lift_end_then_hybrid_approach_then_lerobot",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy_type": args.policy_type,
        "checkpoint": str(checkpoint),
        "seed": args.seed,
        "episodes_requested": args.episodes,
        "episodes_evaluable": len(evaluable),
        "setup_failures": len(rows) - len(evaluable),
        "successes": successes,
        "success_rate": successes / max(1, len(evaluable)),
        "failure_counts": dict(Counter(row["failure_reason"] for row in rows if not row["success"])),
        "replan_ratio": args.replan_ratio,
        "max_policy_steps": args.max_policy_steps,
        "actions_per_chunk": actions_per_chunk,
        "video": "ego_mp4_handoff_plus_policy",
        "right_arm_only": freeze_left_arm,
        "snap_tray": bool(args.snap_tray),
        "tray_weld_left": bool(args.tray_weld_left),
        "snap_peg_o2h": bool(args.snap_peg_o2h),
        "left_grip_scale": float(args.left_grip_scale),
        "hybrid_insert": not args.no_hybrid_insert,
        "trained_action_dim": trained_action_dim,
        "episodes": rows,
    }
    _atomic_json(output / "summary.json", summary)
    print(
        f"Success rate: {successes}/{len(evaluable)}; setup failures: {len(rows) - len(evaluable)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
