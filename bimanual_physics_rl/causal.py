from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from dexjoco.tasks.bimanual_assembly.config import TaskConfig
from dexjoco.tasks.state_restorers import restore_initial_state
from interaction_retarget.io.zarr_io import load_zarr_episode
from interaction_retarget.sim.contact import AssemblyContactDetector
from interaction_retarget.sim.replay import zarr_action_to_policy46
from interaction_retarget.transforms import (
    mocap_world_from_object_frame,
    relative_mocap_in_object_frame,
)


DEFAULT_MANIFEST = Path(
    "/mnt/hdd/dexjoco/interaction_sidecar/bimanual_assembly/manifest.json"
)
DEFAULT_TEMPLATES = Path(
    "/mnt/hdd/dexjoco/outputs/bimanual_physics_rl/causal_templates/train_ep001_skill"
)


def build_templates(
    manifest_path: Path,
    output: Path,
    episode: int = 1,
    train_episodes: int = 80,
    lift_steps: int = 60,
) -> None:
    if not 0 <= episode < train_episodes:
        raise ValueError("template episode must be inside the training split")
    if lift_steps < 1:
        raise ValueError("lift_steps must be positive")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    training_entries = manifest["episodes"][:train_episodes]
    entry = next(
        item for item in training_entries if int(item["episode_index"]) == episode
    )
    timing = entry["timing"]
    if timing.get("left_grasp_fallback") or timing.get("right_grasp_fallback"):
        raise ValueError("template episode must have two detected grasps")

    actions, _, initial_state = load_zarr_episode(Path(entry["zarr_path"]))
    if initial_state is None:
        raise ValueError("template episode has no initial state")
    config = TaskConfig()
    env = config.get_environment(
        policy_mode=True,
        render_mode="rgb_array",
        image_obs=False,
        randomize=False,
        seed=episode,
    )
    raw = env.unwrapped
    raw.hz = 0
    raw._prime_rgb_array_renderer = lambda: None
    try:
        env.reset()
        restore_initial_state(env, "bimanual_assembly", config, initial_state)
        specs = {
            "tray": (
                slice(7, 14),
                slice(30, 46),
                raw._socket_body_id,
                int(timing["tray_lift_start"]) + lift_steps,
            ),
            "peg": (
                slice(0, 7),
                slice(14, 30),
                raw._peg_body_id,
                int(timing["peg_lift_start"]) + lift_steps,
            ),
        }
        output.mkdir(parents=True, exist_ok=True)
        skill_steps = {}
        for name, (pose_slice, hand_slice, body_id, end) in specs.items():
            end = min(end, len(actions) - 1)
            object_pose = (
                raw._data.xpos[body_id].copy(),
                raw._data.xquat[body_id].copy(),
            )
            positions = []
            quaternions = []
            hands = []
            for action in actions[: end + 1]:
                policy_action = zarr_action_to_policy46(action)
                pose = policy_action[pose_slice]
                pos, quat = relative_mocap_in_object_frame(
                    pose[:3], pose[3:7], *object_pose
                )
                positions.append(pos)
                quaternions.append(quat)
                hands.append(policy_action[hand_slice])
            np.savez_compressed(
                output / f"{name}.npz",
                mocap_pos_obj=np.stack(positions),
                mocap_quat_obj=np.stack(quaternions),
                hand=np.stack(hands),
                source_episode=np.asarray([episode], dtype=np.int32),
                source_end_frame=np.asarray([end], dtype=np.int32),
            )
            skill_steps[name] = end + 1
    finally:
        env.close()

    (output / "manifest.json").write_text(
        json.dumps(
            {
                "source_manifest": str(manifest_path),
                "training_split": [0, train_episodes],
                "template_episode": episode,
                "lift_steps": lift_steps,
                "skill_steps": skill_steps,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output)


def _pack(right: np.ndarray, left: np.ndarray) -> np.ndarray:
    return np.concatenate([right[:7], left[:7], right[7:], left[7:]]).astype(
        np.float32
    )


def _limit_norm(vector: np.ndarray, limit: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector if norm <= limit else vector * (limit / norm)


def _wxyz(rotation: Rotation) -> np.ndarray:
    return rotation.as_quat()[[3, 0, 1, 2]]


class CausalAssemblyController:
    """Training-skill initialization plus current-state insertion feedback."""

    def __init__(self, raw_env, template_dir: Path = DEFAULT_TEMPLATES) -> None:
        self.raw = raw_env
        self.skills = {
            name: self._load_skill(template_dir / f"{name}.npz")
            for name in ("tray", "peg")
        }
        peg_bottom_z = (
            self.raw._model.geom_pos[self.raw._peg_geom_id, 2]
            - self.raw._model.geom_size[self.raw._peg_geom_id, 1]
        )
        socket_bottom_top_z = (
            self.raw._model.geom_pos[self.raw._socket_bottom_geom_id, 2]
            + self.raw._model.geom_size[self.raw._socket_bottom_geom_id, 2]
        )
        self.target_tip_z = float(
            socket_bottom_top_z
            - peg_bottom_z
            + self.raw._model.site_pos[self.raw._peg_tip_site_id, 2]
        )

    @staticmethod
    def _load_skill(path: Path) -> dict[str, np.ndarray]:
        with np.load(path, allow_pickle=False) as data:
            skill = {
                "position": data["mocap_pos_obj"].copy(),
                "quaternion": data["mocap_quat_obj"].copy(),
                "hand": data["hand"].copy(),
            }
        length = len(skill["position"])
        if skill["quaternion"].shape != (length, 4) or skill["hand"].shape != (
            length,
            16,
        ):
            raise ValueError(f"invalid causal skill: {path}")
        return skill

    def _current_arm(self, side: str) -> np.ndarray:
        data = self.raw._data
        if side == "right":
            mocap_id = self.raw._mocap_right_id
            qpos_ids = self.raw._allegro_dof_right_ids
        else:
            mocap_id = self.raw._mocap_left_id
            qpos_ids = self.raw._allegro_dof_left_ids
        return np.concatenate(
            [data.mocap_pos[mocap_id], data.mocap_quat[mocap_id], data.qpos[qpos_ids]]
        )

    def _world_skill(self, name: str, body_id: int) -> np.ndarray:
        data = self.raw._data
        object_pose = (data.xpos[body_id].copy(), data.xquat[body_id].copy())
        skill = self.skills[name]
        poses = [
            mocap_world_from_object_frame(pos, quat, *object_pose)
            for pos, quat in zip(skill["position"], skill["quaternion"])
        ]
        return np.concatenate(
            [
                np.asarray([pose[0] for pose in poses]),
                np.asarray([pose[1] for pose in poses]),
                skill["hand"],
            ],
            axis=1,
        )

    def reset(self) -> None:
        self.home_right = self._current_arm("right")
        self.left_skill = self._world_skill("tray", self.raw._socket_body_id)
        self.right_skill = self._world_skill("peg", self.raw._peg_body_id)
        self.left_hold = self.left_skill[-1]
        self.right_hold = self.right_skill[-1]
        self._skill_end = len(self.left_skill) + len(self.right_skill)
        self._fine_phase = "align"
        self._fine_step = 0
        self._fine_stable = 0

    def phase(self, step: int) -> tuple[str, float]:
        if step < len(self.left_skill):
            return "left_skill", (step + 1) / len(self.left_skill)
        if step < self._skill_end:
            return "right_skill", (
                step - len(self.left_skill) + 1
            ) / len(self.right_skill)
        return self._fine_phase, 0.0

    def _advance_precision_phase(self) -> None:
        if self._fine_phase == "insert":
            return
        data = self.raw._data
        socket_rotation = data.xmat[self.raw._socket_body_id].reshape(3, 3)
        tip_local = socket_rotation.T @ (
            data.site_xpos[self.raw._peg_tip_site_id]
            - data.xpos[self.raw._socket_body_id]
        )
        if self._fine_phase == "align":
            ready = np.linalg.norm(tip_local[:2]) < 0.001
            timeout = self._fine_step >= 240
            next_phase = "orient"
        else:
            peg_axis = data.xmat[self.raw._peg_body_id].reshape(3, 3)[:, 2]
            angle = np.arccos(
                np.clip(np.dot(peg_axis, socket_rotation[:, 2]), -1.0, 1.0)
            )
            ready = angle < np.deg2rad(1.0) and np.linalg.norm(tip_local[:2]) < 0.002
            timeout = self._fine_step >= 480
            next_phase = "insert"
        self._fine_stable = self._fine_stable + 1 if ready else 0
        if self._fine_stable >= 10 or timeout:
            self._fine_phase = next_phase
            self._fine_step = 0
            self._fine_stable = 0

    def _servo_orient(self, max_angle: float, centering_step: float) -> np.ndarray:
        data = self.raw._data
        socket_rotation = data.xmat[self.raw._socket_body_id].reshape(3, 3)
        right = self._current_arm("right")
        left = self._current_arm("left")
        tip = data.site_xpos[self.raw._peg_tip_site_id].copy()
        self._orient_arms(right, left, max_angle)
        tip_local = socket_rotation.T @ (
            tip - data.xpos[self.raw._socket_body_id]
        )
        desired_tip = data.xpos[self.raw._socket_body_id] + socket_rotation @ np.array(
            [0.0, 0.0, tip_local[2]]
        )
        centering = 0.5 * _limit_norm(desired_tip - tip, centering_step)
        right[0:3] += centering
        left[0:3] -= centering
        right[7:] = self.right_hold[7:]
        left[7:] = self.left_hold[7:]
        return _pack(right, left)

    def _orient_arms(
        self, right: np.ndarray, left: np.ndarray, max_angle: float
    ) -> None:
        data = self.raw._data
        peg_axis = data.xmat[self.raw._peg_body_id].reshape(3, 3)[:, 2]
        socket_axis = data.xmat[self.raw._socket_body_id].reshape(3, 3)[:, 2]
        cross = np.cross(peg_axis, socket_axis)
        angle = float(np.arccos(np.clip(np.dot(peg_axis, socket_axis), -1.0, 1.0)))
        if np.linalg.norm(cross) > 1e-8 and angle > 1e-8:
            correction = Rotation.from_rotvec(
                cross / np.linalg.norm(cross) * 0.5 * min(angle, max_angle)
            )
            tip = data.site_xpos[self.raw._peg_tip_site_id].copy()
            socket = data.site_xpos[self.raw._socket_site_id].copy()
            right[0:3] = tip + correction.apply(right[0:3] - tip)
            wrist = Rotation.from_quat(right[[4, 5, 6, 3]])
            right[3:7] = _wxyz(correction * wrist)
            inverse = correction.inv()
            left[0:3] = socket + inverse.apply(left[0:3] - socket)
            wrist = Rotation.from_quat(left[[4, 5, 6, 3]])
            left[3:7] = _wxyz(inverse * wrist)

    def _servo_insert(
        self, clearance: float, step_size: float, max_angle: float
    ) -> np.ndarray:
        data = self.raw._data
        socket_rotation = data.xmat[self.raw._socket_body_id].reshape(3, 3)
        tip = data.site_xpos[self.raw._peg_tip_site_id]
        tip_local = socket_rotation.T @ (
            tip - data.xpos[self.raw._socket_body_id]
        )
        right = self._current_arm("right")
        left = self._current_arm("left")
        self._orient_arms(right, left, max_angle)
        lateral = socket_rotation @ np.array([-tip_local[0], -tip_local[1], 0.0])
        axial = socket_rotation @ np.array(
            [0.0, 0.0, self.target_tip_z + clearance - tip_local[2]]
        )
        translation = 0.5 * (
            _limit_norm(lateral, step_size) + _limit_norm(axial, step_size)
        )
        right[0:3] += translation
        left[0:3] -= translation
        right[7:] = self.right_hold[7:]
        left[7:] = self.left_hold[7:]
        return _pack(right, left)

    def action(self, step: int, gains: np.ndarray | None = None) -> np.ndarray:
        centering_gain, orientation_gain, insertion_gain = np.clip(
            np.ones(3) if gains is None else np.asarray(gains, dtype=np.float64),
            0.5,
            1.5,
        )
        phase, _ = self.phase(step)
        if phase == "left_skill":
            return _pack(self.home_right, self.left_skill[step])
        if phase == "right_skill":
            index = step - len(self.left_skill)
            return _pack(self.right_skill[index], self.left_hold)
        self._advance_precision_phase()
        phase = self._fine_phase
        self._fine_step += 1
        if phase == "align":
            return self._servo_orient(
                max_angle=0.003 * orientation_gain,
                centering_step=0.0015 * centering_gain,
            )
        if phase == "orient":
            return self._servo_orient(
                max_angle=0.003 * orientation_gain,
                centering_step=0.0005 * centering_gain,
            )
        return self._servo_insert(
            clearance=0.0,
            step_size=0.001 * insertion_gain,
            max_angle=0.001 * orientation_gain,
        )


def evaluate(
    template_dir: Path,
    episodes: int,
    seed: int,
    max_steps: int,
    randomize_dynamics: bool,
    output: Path | None,
) -> None:
    results = []
    config = TaskConfig()
    for episode in range(episodes):
        episode_seed = seed + episode
        env = config.get_environment(
            policy_mode=True,
            render_mode="rgb_array",
            image_obs=False,
            randomize=False,
            randomize_dynamics=randomize_dynamics,
            seed=episode_seed,
        )
        raw = env.unwrapped
        raw.hz = 0
        raw._prime_rgb_array_renderer = lambda: None
        detector = AssemblyContactDetector(raw)
        try:
            env.reset(seed=episode_seed)
            raw._success_started = False
            raw._success_counter = 0
            detector.reset_reference(raw)
            initial_z = np.array(
                [
                    raw._data.xpos[raw._peg_body_id, 2],
                    raw._data.xpos[raw._socket_body_id, 2],
                ]
            )
            controller = CausalAssemblyController(raw, template_dir)
            controller.reset()
            tray_contact = peg_contact = dual_grasp = False
            first_tray_contact = first_peg_contact = None
            max_tray_contacts = max_peg_contacts = 0
            contact_phases: set[str] = set()
            max_lift = np.zeros(2)
            best_tip = float("inf")
            success = False
            final_step = max_steps
            for step in range(max_steps):
                _, _, terminated, truncated, info = env.step(controller.action(step))
                contact = detector.compute(raw)
                tray_contact |= contact.tray_contact
                peg_contact |= contact.peg_contact
                dual_grasp |= contact.tray_contact and contact.peg_contact
                if contact.tray_contact and first_tray_contact is None:
                    first_tray_contact = step
                if contact.peg_contact and first_peg_contact is None:
                    first_peg_contact = step
                max_tray_contacts = max(max_tray_contacts, contact.tray_contact_count)
                max_peg_contacts = max(max_peg_contacts, contact.peg_contact_count)
                if contact.tray_contact or contact.peg_contact:
                    contact_phases.add(controller.phase(step)[0])
                current_z = np.array(
                    [
                        raw._data.xpos[raw._peg_body_id, 2],
                        raw._data.xpos[raw._socket_body_id, 2],
                    ]
                )
                max_lift = np.maximum(max_lift, current_z - initial_z)
                best_tip = min(
                    best_tip,
                    float(
                        np.linalg.norm(
                            raw._data.site_xpos[raw._peg_tip_site_id]
                            - raw._data.site_xpos[raw._socket_site_id]
                        )
                    ),
                )
                if info.get("succeed", False):
                    success = True
                    final_step = step + 1
                    break
                if terminated or truncated:
                    final_step = step + 1
                    break
            results.append(
                {
                    "episode": episode,
                    "seed": episode_seed,
                    "success": success,
                    "steps": final_step,
                    "tray_contact": tray_contact,
                    "peg_contact": peg_contact,
                    "dual_grasp": dual_grasp,
                    "first_tray_contact_step": first_tray_contact,
                    "first_peg_contact_step": first_peg_contact,
                    "max_tray_contacts": max_tray_contacts,
                    "max_peg_contacts": max_peg_contacts,
                    "contact_phases": sorted(contact_phases),
                    "peg_max_lift_m": float(max_lift[0]),
                    "socket_max_lift_m": float(max_lift[1]),
                    "best_tip_distance_m": best_tip,
                }
            )
        finally:
            env.close()
    payload = {
        "episodes": episodes,
        "successes": sum(item["success"] for item in results),
        "dual_grasps": sum(item["dual_grasp"] for item in results),
        "both_lifted": sum(
            item["peg_max_lift_m"] > 0.05
            and item["socket_max_lift_m"] > 0.05
            for item in results
        ),
        "results": results,
        "future_demo_actions": False,
        "native_initial_reset": True,
    }
    rendered = json.dumps(payload, indent=2) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Causal full-task DexJoCo baseline")
    commands = root.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-templates")
    build.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    build.add_argument("--output", type=Path, default=DEFAULT_TEMPLATES)
    build.add_argument("--episode", type=int, default=1)
    build.add_argument("--train-episodes", type=int, default=80)
    build.add_argument("--lift-steps", type=int, default=60)
    build.set_defaults(
        handler=lambda args: build_templates(
            args.manifest,
            args.output,
            args.episode,
            args.train_episodes,
            args.lift_steps,
        )
    )
    run = commands.add_parser("eval")
    run.add_argument("--templates", type=Path, default=DEFAULT_TEMPLATES)
    run.add_argument("--episodes", type=int, default=5)
    run.add_argument("--seed", type=int, default=80_000)
    run.add_argument("--max-steps", type=int, default=1500)
    run.add_argument("--randomize-dynamics", action="store_true")
    run.add_argument("--output", type=Path)
    run.set_defaults(
        handler=lambda args: evaluate(
            args.templates,
            args.episodes,
            args.seed,
            args.max_steps,
            args.randomize_dynamics,
            args.output,
        )
    )
    return root


if __name__ == "__main__":
    args = parser().parse_args()
    args.handler(args)
