import json
import math
from pathlib import Path

import cv2
import numpy as np

from dexjoco.sim.mujoco_gym_env import GymRenderingSpec
from dexjoco.tasks.bimanual_assembly.config import TaskConfig


SEED = 210000
CAMERAS = ("ego", "wrist_left", "wrist_right")
COLORS = {
    "yellow": ((15, 80, 80), (45, 255, 255)),
    "blue": ((85, 50, 40), (130, 255, 255)),
}
CLOSE_HAND = np.asarray(
    [0.0, 1.1, 1.1, 1.0] * 3 + [0.8, 0.55, 0.65, 0.65], dtype=np.float32
)


def visual_feature(image, color):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, *COLORS[color])
    count, _, stats, centers = cv2.connectedComponentsWithStats(mask)
    if count < 2:
        raise RuntimeError(f"{color} target is not visible")
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[index, cv2.CC_STAT_AREA])
    if area < 8:
        raise RuntimeError(f"{color} target component is too small: {area}")
    height, width = mask.shape
    return np.asarray(
        [centers[index, 0] / width, centers[index, 1] / height, math.log(area / (height * width))],
        dtype=np.float64,
    )


def step_to(env, obs, target, steps):
    start = obs["state"][:46].astype(np.float32)
    reward = 0.0
    info = {"succeed": False}
    for index in range(steps):
        ratio = (index + 1) / steps
        action = (1.0 - ratio) * start + ratio * target
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    return obs, reward, info


def arm_slice(side):
    return slice(0, 3) if side == "right" else slice(7, 10)


def calibrate(env, obs, target, side, camera, color, epsilon=0.025):
    base = target.copy()
    columns = []
    for axis in range(3):
        plus = base.copy()
        plus[arm_slice(side).start + axis] += epsilon
        obs, _, _ = step_to(env, obs, plus, 12)
        upper = visual_feature(obs[camera], color)
        minus = base.copy()
        minus[arm_slice(side).start + axis] -= epsilon
        obs, _, _ = step_to(env, obs, minus, 24)
        lower = visual_feature(obs[camera], color)
        columns.append((upper - lower) / (2.0 * epsilon))
        obs, _, _ = step_to(env, obs, base, 12)
    return obs, np.stack(columns, axis=1)


def save_frame(obs, name):
    mosaic = np.concatenate([obs[camera] for camera in CAMERAS], axis=1)
    cv2.imwrite(str(Path("/tmp") / f"ibvs_{name}.png"), cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR))


env = TaskConfig().get_environment(
    policy_mode=True,
    render_mode="rgb_array",
    image_obs=True,
    randomize=False,
    seed=SEED,
    render_spec=GymRenderingSpec(128, 128),
)
env.unwrapped.hz = 0
history = []
try:
    obs, _ = env.reset(seed=SEED)
    target = obs["state"][:46].astype(np.float32).copy()
    save_frame(obs, "start")
    specs = {
        "right": ("wrist_right", "yellow", np.asarray([0.58, 0.82, math.log(0.20)])),
        "left": ("wrist_left", "blue", np.asarray([0.65, 0.82, math.log(0.25)])),
    }
    jacobians = {}
    for side, (camera, color, _) in specs.items():
        obs, jacobians[side] = calibrate(env, obs, target, side, camera, color)

    for iteration in range(16):
        old_features = {}
        deltas = {}
        record = {"iteration": iteration}
        for side, (camera, color, desired) in specs.items():
            current = visual_feature(obs[camera], color)
            error = desired - current
            jacobian = jacobians[side]
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + 0.08 * np.eye(3), error
            )
            delta = np.clip(delta, -0.04, 0.04)
            norm = np.linalg.norm(delta)
            if norm > 0.05:
                delta *= 0.05 / norm
            old_features[side] = current
            deltas[side] = delta
            record[side] = {
                "feature": current.tolist(),
                "error_norm": float(np.linalg.norm(error)),
                "delta": delta.tolist(),
            }
            target[arm_slice(side)] += delta.astype(np.float32)
        target[[0, 7]] = np.clip(target[[0, 7]], -0.75, 0.20)
        target[[1, 8]] = np.clip(target[[1, 8]], -0.75, 0.75)
        target[[2, 9]] = np.clip(target[[2, 9]], 0.78, 1.60)
        obs, reward, info = step_to(env, obs, target, 20)
        history.append(record)
        for side, (camera, color, _) in specs.items():
            new_feature = visual_feature(obs[camera], color)
            denominator = float(deltas[side] @ deltas[side]) + 1e-6
            jacobians[side] += np.outer(
                new_feature - old_features[side] - jacobians[side] @ deltas[side],
                deltas[side],
            ) / denominator
        if info["succeed"]:
            break

    save_frame(obs, "aligned")
    target[14:30] = CLOSE_HAND
    target[30:46] = CLOSE_HAND
    obs, reward, info = step_to(env, obs, target, 100)
    save_frame(obs, "closed")
    before_lift = {
        side: visual_feature(obs[camera], color).tolist()
        for side, (camera, color, _) in specs.items()
    }
    target[[2, 9]] += 0.20
    obs, reward, info = step_to(env, obs, target, 100)
    save_frame(obs, "lifted")
    after_lift = {
        side: visual_feature(obs[camera], color).tolist()
        for side, (camera, color, _) in specs.items()
    }
    output = {
        "policy_observation": [*CAMERAS, "state[:46]"],
        "training_feedback": ["public RGB feature change", "native reward", "termination"],
        "forbidden_sources_used": [],
        "demo_replay": False,
        "seed": SEED,
        "native_success": bool(info["succeed"]),
        "native_reward": float(reward),
        "before_lift": before_lift,
        "after_lift": after_lift,
        "history": history,
    }
finally:
    env.close()
Path("/tmp/ibvs_result.json").write_text(json.dumps(output, indent=2) + "\n")
print(json.dumps(output, indent=2))
