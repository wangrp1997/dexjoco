from __future__ import annotations

import argparse
import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from dexjoco.sim.mujoco_gym_env import GymRenderingSpec
from dexjoco.tasks.bimanual_assembly.config import TaskConfig
from interaction_retarget.sim.contact import AssemblyContactDetector
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
from train_rgb_geometry import GeometryNet


ROOT = "/mnt/hdd/dexjoco/outputs/bimanual_physics_rl/student_rgbp_v1"
OFFSETS = (np.array([0.0038, -0.0665, 0.1892]), np.array([0.0148, 0.1115, 0.1506]))


def limit(value, maximum):
    norm = np.linalg.norm(value)
    return value if norm <= maximum else value * maximum / norm


def rotate_about(command, offset, rotation, pivot_offset):
    old = Rotation.from_quat(command[[offset + 4, offset + 5, offset + 6, offset + 3]])
    pivot = command[offset : offset + 3] + old.apply(pivot_offset)
    command[offset : offset + 3] = pivot - rotation.apply(pivot_offset)
    command[offset + 3 : offset + 7] = rotation.as_quat()[[3, 0, 1, 2]]


def pixel_score(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    peg = _component_features(cv2.inRange(hsv, (15, 80, 80), (45, 255, 255)))
    socket = _component_features(cv2.inRange(hsv, (85, 45, 30), (125, 255, 180)))
    return np.linalg.norm((peg[1:3] - socket[1:3]) * 128 - np.array([2.94, -17.24]))


def main(seed, max_steps):
    device = torch.device("cuda")
    geometry_payload = torch.load(
        f"{ROOT}/visual_geometry_rgb_dagger_v5.pt",
        map_location="cpu",
        weights_only=False,
    )
    geometry = GeometryNet().to(device)
    geometry.load_state_dict(geometry_payload["model"])
    geometry.eval()
    geometry_stats = tuple(
        torch.tensor(geometry_payload[key], device=device)
        for key in ("state_mean", "state_std", "grasp_mean", "grasp_std", "geometry_mean", "geometry_std")
    )
    keypose_payload = torch.load(f"{ROOT}/model_keypose_rgbhead.pt", map_location="cpu", weights_only=False)
    keypose = StudentNet(action_horizon=len(KEY_STEPS), action_mode="absolute").to(device)
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
        seed=seed,
        render_spec=GymRenderingSpec(128, 128),
    )
    raw = env.unwrapped
    raw.hz = 0
    detector = AssemblyContactDetector(raw)
    try:
        obs, _ = env.reset(seed=seed)
        detector.reset_reference(raw)
        images, state = public_observation(obs)
        initial_images = images.copy()
        initial = state_to_absolute(state)
        tensors = _tensor_batch(images[None], state[None], initial[None], None, device, *keypose_stats)
        with torch.inference_mode():
            targets = (keypose(*tensors[:3])[0].clamp(-5, 5) * keypose_stats[3] + keypose_stats[2]).cpu().numpy()
        _apply_peg_visual_head(targets, images, keypose_payload["peg_visual_head"])
        command = absolute_to_action(targets[-1])
        grasp = action_to_absolute(command)
        for target in _trajectory(initial, targets):
            obs, _, terminated, truncated, info = env.step(absolute_to_action(target))
        for _ in range(30):
            obs, _, terminated, truncated, info = env.step(command)
        smooth = None
        success = bool(info.get("succeed", False))
        phase = "align"
        stable = 0
        for step in range(max_steps):
            images, state = public_observation(obs)
            pixels = torch.from_numpy(
                np.ascontiguousarray(np.concatenate([initial_images, images])[None])
            ).to(device=device, dtype=torch.float32).permute(0, 1, 4, 2, 3).div_(255)
            sm, ss, gm, gs, ym, ys = geometry_stats
            state_tensor = ((torch.tensor(state, device=device) - sm) / ss)[None]
            grasp_tensor = ((torch.tensor(grasp, device=device) - gm) / gs)[None]
            with torch.inference_mode():
                prediction = (geometry(pixels, state_tensor, grasp_tensor)[0] * ys + ym).cpu().numpy()
            smooth = prediction if smooth is None else 0.2 * prediction + 0.8 * smooth
            center, orientation, axial = smooth[:3], smooth[3:6], smooth[6:9]
            stable = stable + 1 if np.linalg.norm(center) < 0.012 else 0
            if phase == "align" and stable >= 5:
                phase = "insert"
            if phase == "align":
                position_step = 0.0015 if np.linalg.norm(center) > 0.03 else 0.0002
                translation = 0.5 * limit(center, position_step)
            else:
                translation = 0.5 * (
                    limit(center, 0.0001) + limit(axial, 0.0005)
                )
            command[:3] += translation
            command[7:10] -= translation
            rotation_step = 0.001 if phase == "align" else 0.0005
            correction = Rotation.from_rotvec(
                0.5 * limit(orientation, rotation_step)
            )
            right = Rotation.from_quat(command[[4, 5, 6, 3]])
            left = Rotation.from_quat(command[[11, 12, 13, 10]])
            rotate_about(command, 0, correction * right, OFFSETS[0])
            rotate_about(command, 7, correction.inv() * left, OFFSETS[1])
            obs, _, terminated, truncated, info = env.step(command)
            success = bool(info.get("succeed", False))
            if (step + 1) % 50 == 0:
                socket_rotation = raw._data.xmat[raw._socket_body_id].reshape(3, 3)
                local = socket_rotation.T @ (
                    raw._data.site_xpos[raw._peg_tip_site_id] - raw._data.xpos[raw._socket_body_id]
                )
                print(
                    {
                        "step": step + 1,
                        "phase": phase,
                        "pixel": round(float(pixel_score(images[0])), 2),
                        "pred_center_mm": round(float(np.linalg.norm(center) * 1000), 1),
                        "pred_axial_mm": round(float(np.linalg.norm(axial) * 1000), 1),
                        "tip_xy_mm_diagnostic": round(float(np.linalg.norm(local[:2]) * 1000), 1),
                    },
                    flush=True,
                )
            if success or terminated or truncated:
                break
        contact = detector.compute(raw)
        print(
            {
                "seed": seed,
                "success": success,
                "precision_steps": step + 1,
                "peg_contact_diagnostic": bool(contact.peg_contact),
                "tray_contact_diagnostic": bool(contact.tray_contact),
            }
        )
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=130000)
    parser.add_argument("--max-steps", type=int, default=500)
    args = parser.parse_args()
    main(args.seed, args.max_steps)
