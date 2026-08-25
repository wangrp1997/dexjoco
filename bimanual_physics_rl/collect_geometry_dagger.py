from __future__ import annotations

import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from bimanual_physics_rl.keypose import KEY_STEPS, _apply_peg_visual_head, _trajectory
from bimanual_physics_rl.student import (
    StudentNet,
    _tensor_batch,
    absolute_to_action,
    action_to_absolute,
    public_observation,
    state_to_absolute,
)
from eval_rgb_geometry import OFFSETS, ROOT, limit, rotate_about
from train_rgb_geometry import GeometryNet


OUTPUT = Path(ROOT) / "servo_rgb_geometry_dagger_v5"


def geometry_label(raw):
    data = raw._data
    socket_rotation = data.xmat[raw._socket_body_id].reshape(3, 3)
    tip = data.site_xpos[raw._peg_tip_site_id]
    tip_local = socket_rotation.T @ (tip - data.xpos[raw._socket_body_id])
    peg_axis = data.xmat[raw._peg_body_id].reshape(3, 3)[:, 2]
    socket_axis = socket_rotation[:, 2]
    cross = np.cross(peg_axis, socket_axis)
    angle = np.arccos(np.clip(peg_axis @ socket_axis, -1.0, 1.0))
    orientation = (
        np.zeros(3)
        if np.linalg.norm(cross) < 1e-8
        else cross / np.linalg.norm(cross) * angle
    )
    centering = socket_rotation @ np.array([-tip_local[0], -tip_local[1], 0.0])
    peg_bottom_z = (
        raw._model.geom_pos[raw._peg_geom_id, 2]
        - raw._model.geom_size[raw._peg_geom_id, 1]
    )
    socket_bottom_top_z = (
        raw._model.geom_pos[raw._socket_bottom_geom_id, 2]
        + raw._model.geom_size[raw._socket_bottom_geom_id, 2]
    )
    target_tip_z = (
        socket_bottom_top_z
        - peg_bottom_z
        + raw._model.site_pos[raw._peg_tip_site_id, 2]
    )
    axial = socket_rotation @ np.array([0.0, 0.0, target_tip_z - tip_local[2]])
    return np.r_[centering, orientation, axial].astype(np.float32)


def collect(seeds):
    from dexjoco.sim.mujoco_gym_env import GymRenderingSpec
    from dexjoco.tasks.bimanual_assembly.config import TaskConfig

    torch.set_num_threads(1)
    device = torch.device("cuda")
    geometry_payload = torch.load(
        f"{ROOT}/visual_geometry_rgb_dagger_v4.pt",
        map_location="cpu",
        weights_only=False,
    )
    geometry = GeometryNet().to(device)
    geometry.load_state_dict(geometry_payload["model"])
    geometry.eval()
    geometry_stats = tuple(
        torch.tensor(geometry_payload[key], device=device)
        for key in (
            "state_mean",
            "state_std",
            "grasp_mean",
            "grasp_std",
            "geometry_mean",
            "geometry_std",
        )
    )
    keypose_payload = torch.load(
        f"{ROOT}/model_keypose_rgbhead.pt", map_location="cpu", weights_only=False
    )
    keypose = StudentNet(action_horizon=len(KEY_STEPS), action_mode="absolute").to(
        device
    )
    keypose.load_state_dict(keypose_payload["model"])
    keypose.eval()
    keypose_stats = tuple(
        torch.tensor(keypose_payload[key], device=device)
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
    raw = env.unwrapped
    raw.hz = 0
    results = []
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            images, state = public_observation(obs)
            initial_images = images.copy()
            initial = state_to_absolute(state)
            tensors = _tensor_batch(
                images[None], state[None], initial[None], None, device, *keypose_stats
            )
            with torch.inference_mode():
                targets = (
                    keypose(*tensors[:3])[0].clamp(-5, 5) * keypose_stats[3]
                    + keypose_stats[2]
                ).cpu().numpy()
            _apply_peg_visual_head(
                targets, images, keypose_payload["peg_visual_head"]
            )
            command = absolute_to_action(targets[-1])
            grasp = action_to_absolute(command)
            for target in _trajectory(initial, targets):
                obs, _, terminated, truncated, info = env.step(
                    absolute_to_action(target)
                )
            for _ in range(30):
                obs, _, terminated, truncated, info = env.step(command)
            saved_images, states, labels = [], [], []
            smooth = None
            for step in range(500):
                images, state = public_observation(obs)
                saved_images.append(images)
                states.append(state)
                labels.append(geometry_label(raw))
                pixels = torch.from_numpy(
                    np.ascontiguousarray(
                        np.concatenate([initial_images, images])[None]
                    )
                ).to(device=device, dtype=torch.float32).permute(0, 1, 4, 2, 3).div_(255)
                sm, ss, gm, gs, ym, ys = geometry_stats
                state_tensor = ((torch.tensor(state, device=device) - sm) / ss)[None]
                grasp_tensor = ((torch.tensor(grasp, device=device) - gm) / gs)[None]
                with torch.inference_mode():
                    prediction = (
                        geometry(pixels, state_tensor, grasp_tensor)[0] * ys + ym
                    ).cpu().numpy()
                smooth = prediction if smooth is None else 0.2 * prediction + 0.8 * smooth
                center, orientation = smooth[:3], smooth[3:6]
                position_step = 0.0015 if np.linalg.norm(center) > 0.03 else 0.0002
                translation = 0.5 * limit(center, position_step)
                command[:3] += translation
                command[7:10] -= translation
                correction = Rotation.from_rotvec(
                    0.5 * limit(orientation, 0.001)
                )
                right = Rotation.from_quat(command[[4, 5, 6, 3]])
                left = Rotation.from_quat(command[[11, 12, 13, 10]])
                rotate_about(command, 0, correction * right, OFFSETS[0])
                rotate_about(command, 7, correction.inv() * left, OFFSETS[1])
                obs, _, terminated, truncated, info = env.step(command)
                if terminated or truncated:
                    break
            path = OUTPUT / f"episode_seed{seed:06d}.npz"
            np.savez_compressed(
                path,
                initial_images=initial_images,
                images=np.asarray(saved_images, dtype=np.uint8),
                state=np.asarray(states, dtype=np.float32),
                geometry=np.asarray(labels, dtype=np.float32),
                grasp_target=grasp.astype(np.float32),
                success=np.asarray(bool(info.get("succeed", False))),
            )
            results.append({"seed": seed, "file": str(path), "samples": len(states)})
    finally:
        env.close()
    return results


if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=True)
    groups = [list(range(180000 + worker, 180016, 2)) for worker in range(2)]
    results = []
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(collect, group) for group in groups]
        for future in as_completed(futures):
            results.extend(future.result())
            print(f"collected={len(results)}/16", flush=True)
    results.sort(key=lambda item: item["seed"])
    (OUTPUT / "manifest.json").write_text(
        json.dumps(
            {
                "version": 5,
                "rollout_policy": "rgb_geometry_dagger_v4",
                "privileged_training_labels": True,
                "policy_inputs": [
                    "initial_rgb",
                    "current_rgb",
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
