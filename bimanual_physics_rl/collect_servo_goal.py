from __future__ import annotations

import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import torch

from bimanual_physics_rl.causal import CausalAssemblyController, DEFAULT_TEMPLATES
from bimanual_physics_rl.keypose import (
    KEY_STEPS,
    _apply_peg_visual_head,
    _component_features,
    _trajectory,
)
from bimanual_physics_rl.student import (
    StudentNet,
    _tensor_batch,
    absolute_to_action,
    action_to_absolute,
    public_observation,
    state_to_absolute,
)


ROOT = Path("/mnt/hdd/dexjoco/outputs/bimanual_physics_rl/student_rgbp_v1")
CHECKPOINT = ROOT / "model_keypose_rgbhead.pt"
OUTPUT = ROOT / "servo_rgb_geometry_rollouts_v3"


def visual_features(images: np.ndarray) -> np.ndarray:
    values = []
    for low, high in (
        ((15, 80, 80), (45, 255, 255)),
        ((85, 45, 30), (125, 255, 180)),
    ):
        values.extend(
            _component_features(
                cv2.inRange(cv2.cvtColor(image, cv2.COLOR_RGB2HSV), low, high)
            )
            for image in images
        )
    return np.concatenate(values)


def collect(seeds: list[int]) -> list[dict]:
    from dexjoco.sim.mujoco_gym_env import GymRenderingSpec
    from dexjoco.tasks.bimanual_assembly.config import TaskConfig

    torch.set_num_threads(1)
    device = torch.device("cpu")
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = StudentNet(action_horizon=len(KEY_STEPS), action_mode="absolute")
    model.load_state_dict(payload["model"])
    model.eval()
    stats = tuple(
        torch.tensor(payload[key], device=device)
        for key in ("state_mean", "state_std", "action_mean", "action_std")
    )
    env = TaskConfig().get_environment(
        policy_mode=True,
        render_mode="rgb_array",
        image_obs=True,
        randomize=False,
        randomize_dynamics=False,
        seed=seeds[0],
        render_spec=GymRenderingSpec(128, 128),
    )
    env.unwrapped.hz = 0
    results = []
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            teacher = CausalAssemblyController(env.unwrapped, DEFAULT_TEMPLATES)
            teacher.reset()
            images, state = public_observation(obs)
            initial_images = images.copy()
            initial_visual = visual_features(images)
            initial = state_to_absolute(state)
            tensors = _tensor_batch(
                images[None], state[None], initial[None], None, device, *stats
            )
            with torch.inference_mode():
                targets = (
                    model(*tensors[:3])[0].clamp(-5.0, 5.0) * stats[3] + stats[2]
                ).numpy()
            _apply_peg_visual_head(targets, images, payload["peg_visual_head"])
            hold = absolute_to_action(targets[-1])
            grasp_target = action_to_absolute(hold)
            for command in _trajectory(initial, targets):
                obs, _, terminated, truncated, info = env.step(
                    absolute_to_action(command)
                )
            for _ in range(30):
                obs, _, terminated, truncated, info = env.step(hold)
            current_images, current_visual, states, geometry = [], [], [], []
            success = bool(info.get("succeed", False))
            for step in range(984):
                if step % 2 == 0:
                    images, state = public_observation(obs)
                    current_images.append(images)
                    current_visual.append(visual_features(images))
                    states.append(state)
                    data = env.unwrapped._data
                    socket_rotation = data.xmat[
                        env.unwrapped._socket_body_id
                    ].reshape(3, 3)
                    tip = data.site_xpos[env.unwrapped._peg_tip_site_id]
                    tip_local = socket_rotation.T @ (
                        tip - data.xpos[env.unwrapped._socket_body_id]
                    )
                    peg_axis = data.xmat[env.unwrapped._peg_body_id].reshape(3, 3)[:, 2]
                    socket_axis = socket_rotation[:, 2]
                    cross = np.cross(peg_axis, socket_axis)
                    angle = np.arccos(np.clip(peg_axis @ socket_axis, -1.0, 1.0))
                    orientation = (
                        np.zeros(3)
                        if np.linalg.norm(cross) < 1e-8
                        else cross / np.linalg.norm(cross) * angle
                    )
                    centering = socket_rotation @ np.array(
                        [-tip_local[0], -tip_local[1], 0.0]
                    )
                    axial = socket_rotation @ np.array(
                        [0.0, 0.0, teacher.target_tip_z - tip_local[2]]
                    )
                    geometry.append(np.r_[centering, orientation, axial])
                obs, _, terminated, truncated, info = env.step(
                    teacher.action(486 + step)
                )
                success = bool(info.get("succeed", False))
                if success or terminated or truncated:
                    break
            path = OUTPUT / f"episode_seed{seed:06d}.npz"
            np.savez_compressed(
                path,
                initial_visual=initial_visual.astype(np.float32),
                initial_images=initial_images,
                images=np.asarray(current_images, dtype=np.uint8),
                current_visual=np.asarray(current_visual, dtype=np.float32),
                state=np.asarray(states, dtype=np.float32),
                geometry=np.asarray(geometry, dtype=np.float32),
                grasp_target=grasp_target.astype(np.float32),
                success=np.asarray(success),
            )
            results.append(
                {"seed": seed, "file": str(path), "samples": len(states), "success": success}
            )
    finally:
        env.close()
    return results


if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=True)
    groups = [list(range(160000 + worker, 160048, 4)) for worker in range(4)]
    results = []
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(collect, group) for group in groups]
        for future in as_completed(futures):
            results.extend(future.result())
            print(f"collected={len(results)}/48", flush=True)
    results.sort(key=lambda value: value["seed"])
    (OUTPUT / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "teacher_is_privileged": True,
                "privileged_training_labels": [
                    "centering_world",
                    "orientation_world",
                    "axial_world",
                ],
                "policy_inputs": [
                    "initial_rgb_components",
                    "current_rgb_components",
                    "state46",
                    "own_grasp_target",
                ],
                "forbidden_policy_inputs": [
                    "object_pose",
                    "contact_truth",
                    "demo_action",
                    "phase_index",
                ],
                "episodes": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"success={sum(item['success'] for item in results)}/{len(results)}",
        flush=True,
    )
