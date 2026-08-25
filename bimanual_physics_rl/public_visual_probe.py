import json

import cv2
import numpy as np

from dexjoco.sim.mujoco_gym_env import GymRenderingSpec
from dexjoco.tasks.bimanual_assembly.config import TaskConfig


CAMERAS = ("ego", "wrist_left", "wrist_right")
COLORS = {
    "yellow": ((15, 80, 80), (45, 255, 255)),
    "blue": ((85, 50, 40), (130, 255, 255)),
}


def component(image, bounds):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, *bounds)
    count, _, stats, centers = cv2.connectedComponentsWithStats(mask)
    if count < 2:
        return None
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[index, cv2.CC_STAT_AREA])
    if area < 8:
        return None
    height, width = mask.shape
    return [
        float(centers[index, 0] / width),
        float(centers[index, 1] / height),
        float(area / (height * width)),
    ]


def features(obs):
    return {
        camera: {color: component(obs[camera], bounds) for color, bounds in COLORS.items()}
        for camera in CAMERAS
    }


def rollout(env, seed, arm=None, axis=None, delta=0.0):
    obs, _ = env.reset(seed=seed)
    action = obs["state"][:46].astype(np.float32).copy()
    if arm is not None:
        action[(0 if arm == "right" else 7) + axis] += delta
    for _ in range(20):
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    return {
        "arm": arm,
        "axis": axis,
        "delta": delta,
        "tcp": obs["state"][:14].tolist(),
        "features": features(obs),
        "native_reward": reward,
        "native_success": bool(info["succeed"]),
    }


env = TaskConfig().get_environment(
    policy_mode=True,
    render_mode="rgb_array",
    image_obs=True,
    randomize=False,
    seed=210000,
    render_spec=GymRenderingSpec(128, 128),
)
env.unwrapped.hz = 0
try:
    results = [rollout(env, 210000)]
    for arm in ("right", "left"):
        for axis in range(3):
            for delta in (-0.02, 0.02):
                results.append(rollout(env, 210000, arm, axis, delta))
finally:
    env.close()
print(json.dumps(results, indent=2))
