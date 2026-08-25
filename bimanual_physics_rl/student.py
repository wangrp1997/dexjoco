from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


CAMERAS = ("ego", "wrist_left", "wrist_right")
PUBLIC_STATE_DIM = 46
RESIDUAL_ACTION_DIM = 44


def public_observation(obs: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return only deployable RGB and robot proprioception."""
    state = np.asarray(obs["state"], dtype=np.float32)
    if state.ndim != 1 or state.size < PUBLIC_STATE_DIM:
        raise ValueError(f"Expected at least {PUBLIC_STATE_DIM} state values, got {state.shape}")
    images = np.stack([np.asarray(obs[key], dtype=np.uint8) for key in CAMERAS])
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"Expected three HWC RGB images, got {images.shape}")
    return images, state[:PUBLIC_STATE_DIM].copy()


def action_to_residual(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Encode an absolute 46-D target as a robot-relative 44-D command."""
    state = np.asarray(state, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    if state.shape != (PUBLIC_STATE_DIM,) or action.shape != (46,):
        raise ValueError(f"Expected state/action (46,)/(46,), got {state.shape}/{action.shape}")

    def arm(state_pose, target_pose, state_hand, target_hand):
        current = Rotation.from_quat(state_pose[[4, 5, 6, 3]])
        target = Rotation.from_quat(target_pose[[4, 5, 6, 3]])
        return np.concatenate(
            [target_pose[:3] - state_pose[:3], (target * current.inv()).as_rotvec(), target_hand - state_hand]
        )

    right = arm(state[:7], action[:7], state[14:30], action[14:30])
    left = arm(state[7:14], action[7:14], state[30:46], action[30:46])
    return np.concatenate([right, left]).astype(np.float32)


def residual_to_action(state: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """Decode a robot-relative command into DexJoCo's absolute 46-D action."""
    state = np.asarray(state, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)
    if state.shape != (PUBLIC_STATE_DIM,) or residual.shape != (RESIDUAL_ACTION_DIM,):
        raise ValueError(
            f"Expected state/residual (46,)/(44,), got {state.shape}/{residual.shape}"
        )

    def arm(state_pose, command, state_hand):
        current = Rotation.from_quat(state_pose[[4, 5, 6, 3]])
        target = Rotation.from_rotvec(command[3:6]) * current
        quat = target.as_quat()[[3, 0, 1, 2]]
        pose = np.concatenate([state_pose[:3] + command[:3], quat])
        return pose, state_hand + command[6:22]

    right_pose, right_hand = arm(state[:7], residual[:22], state[14:30])
    left_pose, left_hand = arm(state[7:14], residual[22:], state[30:46])
    return np.concatenate([right_pose, left_pose, right_hand, left_hand]).astype(np.float32)


def action_to_absolute(action: np.ndarray) -> np.ndarray:
    """Convert DexJoCo's quaternion action to a continuous 44-D policy action."""
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (46,):
        raise ValueError(f"Expected action (46,), got {action.shape}")

    def arm(pose, hand):
        rotation = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_rotvec()
        return np.concatenate([pose[:3], rotation, hand])

    return np.concatenate(
        [arm(action[:7], action[14:30]), arm(action[7:14], action[30:46])]
    ).astype(np.float32)


def absolute_to_action(action: np.ndarray) -> np.ndarray:
    """Convert a 44-D position/rotvec/hand command to DexJoCo's action."""
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (RESIDUAL_ACTION_DIM,):
        raise ValueError(f"Expected absolute action (44,), got {action.shape}")

    def arm(command):
        quat = Rotation.from_rotvec(command[3:6]).as_quat()[[3, 0, 1, 2]]
        return np.concatenate([command[:3], quat]), command[6:22]

    right_pose, right_hand = arm(action[:22])
    left_pose, left_hand = arm(action[22:])
    return np.concatenate([right_pose, left_pose, right_hand, left_hand]).astype(np.float32)


def state_to_absolute(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (PUBLIC_STATE_DIM,):
        raise ValueError(f"Expected state (46,), got {state.shape}")
    return action_to_absolute(
        np.concatenate([state[:14], state[14:30], state[30:46]])
    )


class StudentNet(nn.Module):
    def __init__(
        self,
        pretrained: bool = False,
        uses_previous_action: bool = True,
        action_horizon: int = 1,
        action_mode: str = "residual",
    ) -> None:
        super().__init__()
        if action_horizon < 1:
            raise ValueError("action_horizon must be positive")
        if action_mode not in ("residual", "absolute"):
            raise ValueError(f"Unknown action mode: {action_mode}")
        self.uses_previous_action = uses_previous_action
        self.action_horizon = action_horizon
        self.action_mode = action_mode
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        backbone.fc = nn.Identity()
        self.vision = backbone
        self.proprio = nn.Sequential(
            nn.Linear(
                PUBLIC_STATE_DIM + (RESIDUAL_ACTION_DIM if uses_previous_action else 0),
                128,
            ),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(3 * 512 + 128, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, action_horizon * RESIDUAL_ACTION_DIM),
        )
        self.register_buffer(
            "image_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        )

    def forward(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        previous_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, views, channels, height, width = images.shape
        if views != len(CAMERAS) or channels != 3:
            raise ValueError(f"Expected Bx3x3xHxW images, got {images.shape}")
        images = images.reshape(batch * views, channels, height, width)
        features = self.vision((images - self.image_mean) / self.image_std)
        features = features.reshape(batch, views * features.shape[-1])
        if self.uses_previous_action:
            if previous_action is None:
                raise ValueError("Previous policy action is required")
            state = torch.cat([state, previous_action], dim=-1)
        output = self.head(torch.cat([features, self.proprio(state)], dim=-1))
        return output.reshape(batch, self.action_horizon, RESIDUAL_ACTION_DIM)


class RecurrentStudentNet(nn.Module):
    def __init__(self, pretrained: bool = False, action_mode: str = "residual") -> None:
        super().__init__()
        self.uses_previous_action = True
        self.action_horizon = 1
        self.action_mode = action_mode
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        backbone.fc = nn.Identity()
        self.vision = backbone
        self.proprio = nn.Sequential(
            nn.Linear(PUBLIC_STATE_DIM + RESIDUAL_ACTION_DIM, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
        )
        self.fusion = nn.Sequential(nn.Linear(3 * 512 + 128, 256), nn.SiLU())
        self.gru = nn.GRU(256, 256, batch_first=True)
        self.head = nn.Linear(256, RESIDUAL_ACTION_DIM)
        self.register_buffer(
            "image_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        )

    def forward(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        previous_action: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, views, channels, height, width = images.shape
        images = images.reshape(batch * steps * views, channels, height, width)
        features = self.vision((images - self.image_mean) / self.image_std)
        features = features.reshape(batch, steps, views * features.shape[-1])
        proprio = self.proprio(torch.cat([state, previous_action], dim=-1))
        output, hidden = self.gru(self.fusion(torch.cat([features, proprio], dim=-1)), hidden)
        return self.head(output), hidden


def _tensor_batch(
    images: np.ndarray,
    state: np.ndarray,
    previous_action: np.ndarray,
    target: np.ndarray | None,
    device: torch.device,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
):
    image_tensor = (
        torch.from_numpy(np.ascontiguousarray(images))
        .to(device=device, dtype=torch.float32, non_blocking=True)
        .permute(0, 1, 4, 2, 3)
        .div_(255.0)
    )
    state_tensor = torch.from_numpy(np.ascontiguousarray(state)).to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    state_tensor = (state_tensor - state_mean) / state_std
    previous_tensor = torch.from_numpy(np.ascontiguousarray(previous_action)).to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    previous_tensor = (previous_tensor - action_mean) / action_std
    if target is None:
        return image_tensor, state_tensor, previous_tensor, None
    target_tensor = torch.from_numpy(np.ascontiguousarray(target)).to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    return (
        image_tensor,
        state_tensor,
        previous_tensor,
        (target_tensor - action_mean) / action_std,
    )


class StudentPolicy:
    def __init__(self, checkpoint: Path, device: str = "cpu") -> None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("public_state_dim") != PUBLIC_STATE_DIM or tuple(
            payload.get("cameras", ())
        ) != CAMERAS:
            raise ValueError("Checkpoint does not match the deployable observation contract")
        self.device = torch.device(device)
        self.uses_previous_action = bool(payload.get("uses_previous_action", False))
        self.action_horizon = int(payload.get("action_horizon", 1))
        self.action_mode = payload.get("action_mode", "residual")
        self.recurrent = bool(payload.get("recurrent", False))
        if self.recurrent:
            self.model = RecurrentStudentNet(action_mode=self.action_mode).to(self.device)
        else:
            self.model = StudentNet(
                uses_previous_action=self.uses_previous_action,
                action_horizon=self.action_horizon,
                action_mode=self.action_mode,
            ).to(self.device)
        self.model.load_state_dict(payload["model"])
        self.model.eval()
        self.state_mean = torch.tensor(payload["state_mean"], device=self.device)
        self.state_std = torch.tensor(payload["state_std"], device=self.device)
        self.action_mean = torch.tensor(payload["action_mean"], device=self.device)
        self.action_std = torch.tensor(payload["action_std"], device=self.device)
        self.reset()

    def reset(self) -> None:
        self.previous_action = (
            None
            if self.action_mode == "absolute"
            else np.zeros(RESIDUAL_ACTION_DIM, dtype=np.float32)
        )
        self.action_queue: list[np.ndarray] = []
        self.hidden = None

    def set_previous_action(self, residual: np.ndarray) -> None:
        self.previous_action = np.asarray(residual, dtype=np.float32).copy()

    @torch.inference_mode()
    def predict(self, obs: dict) -> np.ndarray:
        images, state = public_observation(obs)
        if self.previous_action is None:
            self.previous_action = state_to_absolute(state)
        if self.recurrent:
            image_tensor, state_tensor, previous_tensor, _ = _tensor_batch(
                images[None],
                state[None],
                self.previous_action[None],
                None,
                self.device,
                self.state_mean,
                self.state_std,
                self.action_mean,
                self.action_std,
            )
            normalized, self.hidden = self.model(
                image_tensor[:, None],
                state_tensor[:, None],
                previous_tensor[:, None],
                self.hidden,
            )
            command = normalized[0, 0].clamp(-5.0, 5.0)
            self.set_previous_action(
                (command * self.action_std + self.action_mean).float().cpu().numpy()
            )
        elif not self.action_queue:
            image_tensor, state_tensor, previous_tensor, _ = _tensor_batch(
                images[None],
                state[None],
                self.previous_action[None],
                None,
                self.device,
                self.state_mean,
                self.state_std,
                self.action_mean,
                self.action_std,
            )
            normalized = self.model(image_tensor, state_tensor, previous_tensor)[0].clamp(
                -5.0, 5.0
            )
            residuals = normalized * self.action_std + self.action_mean
            self.action_queue.extend(residuals.float().cpu().numpy())
        if not self.recurrent:
            self.set_previous_action(self.action_queue.pop(0))
        if self.action_mode == "absolute":
            return absolute_to_action(self.previous_action)
        return residual_to_action(state, self.previous_action)


def _collect_episode(spec: dict) -> dict:
    from dexjoco.sim.mujoco_gym_env import GymRenderingSpec
    from dexjoco.tasks.bimanual_assembly.config import TaskConfig

    from .causal import CausalAssemblyController

    seed = int(spec["seed"])
    output = Path(spec["output"]) / f"episode_seed{seed:06d}.npz"
    if output.exists() and not spec["overwrite"]:
        return {"seed": seed, "file": str(output), "skipped": True}
    random.seed(seed)
    np.random.seed(seed)
    env = TaskConfig().get_environment(
        policy_mode=True,
        render_mode="rgb_array",
        image_obs=True,
        randomize=spec["randomize"],
        randomize_dynamics=spec["randomize_dynamics"],
        seed=seed,
        render_spec=GymRenderingSpec(spec["image_size"], spec["image_size"]),
    )
    env.unwrapped.hz = 0
    student = StudentPolicy(Path(spec["student"]), spec["device"]) if spec["student"] else None
    images, states, previous_actions, previous_absolute_actions, targets, absolute_targets = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    previous_action = np.zeros(RESIDUAL_ACTION_DIM, dtype=np.float32)
    previous_absolute_action = None
    success = False
    try:
        obs, _ = env.reset(seed=seed)
        teacher = CausalAssemblyController(env.unwrapped, Path(spec["templates"]))
        teacher.reset()
        for step in range(spec["max_steps"]):
            public_images, state = public_observation(obs)
            teacher_action = teacher.action(step)
            teacher_residual = action_to_residual(state, teacher_action)
            if previous_absolute_action is None:
                previous_absolute_action = state_to_absolute(state)
            if step % spec["stride"] == 0:
                images.append(public_images)
                states.append(state)
                previous_actions.append(previous_action)
                previous_absolute_actions.append(previous_absolute_action)
                targets.append(teacher_residual)
                absolute_targets.append(action_to_absolute(teacher_action))

            if student is None or spec["beta"] >= 1.0:
                executed = teacher_action
            else:
                student_residual = action_to_residual(state, student.predict(obs))
                mixed = spec["beta"] * teacher_residual + (1.0 - spec["beta"]) * student_residual
                executed = residual_to_action(state, mixed)
                student.set_previous_action(
                    action_to_absolute(executed)
                    if student.action_mode == "absolute"
                    else mixed
                )
            previous_action = action_to_residual(state, executed)
            previous_absolute_action = action_to_absolute(executed)
            obs, _, terminated, truncated, info = env.step(executed)
            if info.get("succeed", False):
                success = True
                break
            if terminated or truncated:
                break
    finally:
        env.close()

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        images=np.stack(images).astype(np.uint8),
        state=np.stack(states).astype(np.float32),
        previous_action=np.stack(previous_actions).astype(np.float32),
        previous_absolute_action=np.stack(previous_absolute_actions).astype(np.float32),
        target=np.stack(targets).astype(np.float32),
        absolute_target=np.stack(absolute_targets).astype(np.float32),
    )
    return {
        "seed": seed,
        "file": str(output),
        "samples": len(states),
        "steps": step + 1,
        "success": success,
        "skipped": False,
    }


def collect(args) -> None:
    if not 0.0 <= args.beta <= 1.0:
        raise ValueError("beta must be in [0, 1]")
    if args.student is None and args.beta != 1.0:
        raise ValueError("beta below 1 requires --student")
    args.output.mkdir(parents=True, exist_ok=True)
    common = {
        "output": str(args.output),
        "templates": str(args.templates),
        "student": str(args.student) if args.student else None,
        "device": args.device,
        "beta": args.beta,
        "image_size": args.image_size,
        "max_steps": args.max_steps,
        "stride": args.stride,
        "randomize": args.randomize,
        "randomize_dynamics": args.randomize_dynamics,
        "overwrite": args.overwrite,
    }
    specs = [{**common, "seed": args.seed + index} for index in range(args.episodes)]
    results = []
    if args.workers == 1:
        for spec in specs:
            result = _collect_episode(spec)
            results.append(result)
            print(json.dumps(result))
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
            futures = [pool.submit(_collect_episode, spec) for spec in specs]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(json.dumps(result))
    results.sort(key=lambda item: item["seed"])
    manifest = {
        "version": 1,
        "observation": {"cameras": CAMERAS, "state": "tcp_pose+gripper_pose", "dim": 46},
        "forbidden_policy_inputs": [
            "peg_pose",
            "socket_pose",
            "table_height",
            "contact_truth",
            "demo_action",
            "phase_index",
        ],
        "teacher_is_privileged": True,
        "student": str(args.student) if args.student else None,
        "beta": args.beta,
        "episodes": results,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    completed = [item for item in results if not item.get("skipped")]
    successes = sum(bool(item.get("success")) for item in completed)
    print(f"collected={len(completed)} success={successes}/{len(completed)}")


def _episode_files(paths: list[Path]) -> list[Path]:
    files = sorted(file for path in paths for file in path.glob("episode_seed*.npz"))
    if len(files) < 2:
        raise ValueError("At least two collected episodes are required")
    return files


def _normalization(
    files: list[Path], key: str, floor: float, start_step: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    count = 0
    total = total_sq = None
    for path in files:
        with np.load(path, allow_pickle=False) as episode:
            offset = max(
                0,
                start_step
                - (int(episode["first_step"]) if "first_step" in episode.files else 0),
            )
            values = episode[key][offset:].astype(np.float64)
        count += len(values)
        summed = values.sum(axis=0)
        summed_sq = np.square(values).sum(axis=0)
        total = summed if total is None else total + summed
        total_sq = summed_sq if total_sq is None else total_sq + summed_sq
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - np.square(mean), 0.0))
    return mean.astype(np.float32), np.maximum(std, floor).astype(np.float32)


def _episode_actions(episode, action_mode: str) -> np.ndarray:
    if action_mode == "residual":
        return episode["target"]
    if "absolute_target" in episode.files:
        return episode["absolute_target"]
    return np.stack(
        [
            action_to_absolute(residual_to_action(state, residual))
            for state, residual in zip(episode["state"], episode["target"])
        ]
    )


def _episode_previous_actions(
    episode, action_mode: str, targets: np.ndarray
) -> np.ndarray:
    key = "previous_action" if action_mode == "residual" else "previous_absolute_action"
    if key in episode.files:
        return episode[key]
    initial = (
        np.zeros_like(targets[:1])
        if action_mode == "residual"
        else state_to_absolute(episode["state"][0])[None]
    )
    return np.concatenate([initial, targets[:-1]])


def _action_normalization(
    files: list[Path], action_mode: str, floor: float, start_step: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    count = 0
    total = total_sq = None
    for path in files:
        with np.load(path, allow_pickle=False) as episode:
            offset = max(
                0,
                start_step
                - (int(episode["first_step"]) if "first_step" in episode.files else 0),
            )
            values = _episode_actions(episode, action_mode)[offset:].astype(np.float64)
        count += len(values)
        summed = values.sum(axis=0)
        summed_sq = np.square(values).sum(axis=0)
        total = summed if total is None else total + summed
        total_sq = summed_sq if total_sq is None else total_sq + summed_sq
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - np.square(mean), 0.0))
    return mean.astype(np.float32), np.maximum(std, floor).astype(np.float32)


def _loss(
    prediction: torch.Tensor, target: torch.Tensor, hand_weight: float = 0.25
) -> torch.Tensor:
    pose = torch.tensor(
        list(range(0, 6)) + list(range(22, 28)), device=prediction.device
    )
    hand = torch.tensor(
        list(range(6, 22)) + list(range(28, 44)), device=prediction.device
    )
    loss = F.smooth_l1_loss(prediction[..., pose], target[..., pose], beta=0.2)
    if hand_weight:
        loss = loss + hand_weight * F.smooth_l1_loss(
            prediction[..., hand], target[..., hand], beta=0.2
        )
    return loss


def _run_files(
    model,
    files,
    batch_size,
    device,
    stats,
    optimizer=None,
    start_step=0,
) -> float:
    training = optimizer is not None
    model.train(training)
    losses = []
    files = list(files)
    if training:
        random.shuffle(files)
    state_mean, state_std, action_mean, action_std = stats
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    for path in files:
        with np.load(path, allow_pickle=False) as episode:
            images = episode["images"]
            states = episode["state"]
            targets = _episode_actions(episode, model.action_mode)
            previous_actions = _episode_previous_actions(
                episode, model.action_mode, targets
            )
            offset = max(
                0,
                start_step
                - (int(episode["first_step"]) if "first_step" in episode.files else 0),
            )
            images = images[offset:]
            states = states[offset:]
            targets = targets[offset:]
            previous_actions = previous_actions[offset:]
        indices = np.random.permutation(len(states)) if training else np.arange(len(states))
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            future = np.minimum(
                batch[:, None] + np.arange(model.action_horizon)[None, :],
                len(targets) - 1,
            )
            image_tensor, state_tensor, previous_tensor, target_tensor = _tensor_batch(
                images[batch],
                states[batch],
                previous_actions[batch],
                targets[future],
                device,
                state_mean,
                state_std,
                action_mean,
                action_std,
            )
            with torch.set_grad_enabled(training), amp:
                loss = _loss(
                    model(image_tensor, state_tensor, previous_tensor),
                    target_tensor,
                    model.hand_loss_weight,
                )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            losses.append(float(loss.detach()))
    return float(np.mean(losses))


def _run_sequence_files(
    model,
    files,
    sequence_length,
    device,
    stats,
    optimizer=None,
    start_step=0,
) -> float:
    training = optimizer is not None
    model.train(training)
    model.vision.eval()
    files = list(files)
    if training:
        random.shuffle(files)
    losses = []
    state_mean, state_std, action_mean, action_std = stats
    amp = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    for path in files:
        with np.load(path, allow_pickle=False) as episode:
            images = episode["images"]
            states = episode["state"]
            targets = _episode_actions(episode, model.action_mode)
            previous_actions = _episode_previous_actions(
                episode, model.action_mode, targets
            )
            offset = max(
                0,
                start_step
                - (int(episode["first_step"]) if "first_step" in episode.files else 0),
            )
            images = images[offset:]
            states = states[offset:]
            targets = targets[offset:]
            previous_actions = previous_actions[offset:]
        hidden = None
        for start in range(0, len(states), sequence_length):
            end = min(start + sequence_length, len(states))
            image_tensor, state_tensor, previous_tensor, target_tensor = _tensor_batch(
                images[start:end],
                states[start:end],
                previous_actions[start:end],
                targets[start:end],
                device,
                state_mean,
                state_std,
                action_mean,
                action_std,
            )
            with torch.set_grad_enabled(training), amp:
                prediction, hidden = model(
                    image_tensor[None],
                    state_tensor[None],
                    previous_tensor[None],
                    hidden,
                )
                loss = _loss(prediction[0], target_tensor, model.hand_loss_weight)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            hidden = hidden.detach()
            losses.append(float(loss.detach()))
    return float(np.mean(losses))


def train(args) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    files = _episode_files(args.data)
    validation = files[::5]
    training = [file for file in files if file not in set(validation)]
    previous = (
        torch.load(args.resume, map_location="cpu", weights_only=False)
        if args.resume
        else None
    )
    if previous:
        state_mean = np.asarray(previous["state_mean"], dtype=np.float32)
        state_std = np.asarray(previous["state_std"], dtype=np.float32)
        action_mean = np.asarray(previous["action_mean"], dtype=np.float32)
        action_std = np.asarray(previous["action_std"], dtype=np.float32)
    else:
        state_mean, state_std = _normalization(
            training, "state", 1e-4, args.start_step
        )
        action_mean, action_std = _action_normalization(
            training, args.action_mode, 1e-4, args.start_step
        )
    if args.recurrent and args.action_horizon != 1:
        raise ValueError("recurrent students require --action-horizon 1")
    device = torch.device(args.device)
    if args.recurrent:
        model = RecurrentStudentNet(
            pretrained=not args.no_pretrained,
            action_mode=args.action_mode,
        ).to(device)
    else:
        model = StudentNet(
            pretrained=not args.no_pretrained,
            action_horizon=args.action_horizon,
            action_mode=args.action_mode,
        ).to(device)
    model.hand_loss_weight = args.hand_loss_weight
    if previous:
        if not previous.get("uses_previous_action", False):
            raise ValueError("--resume must be a history-enabled student checkpoint")
        if int(previous.get("action_horizon", 1)) != args.action_horizon:
            raise ValueError("--resume action horizon does not match --action-horizon")
        if previous.get("action_mode", "residual") != args.action_mode:
            raise ValueError("--resume action mode does not match --action-mode")
        if bool(previous.get("recurrent", False)) != args.recurrent:
            raise ValueError("--resume recurrent mode does not match --recurrent")
        if int(previous.get("start_step", 0)) != args.start_step:
            raise ValueError("--resume start step does not match --start-step")
        model.load_state_dict(previous["model"])
    elif args.vision_init:
        source = torch.load(args.vision_init, map_location="cpu", weights_only=False)
        vision = {
            key: value for key, value in source["model"].items() if key.startswith("vision.")
        }
        model.load_state_dict(vision, strict=False)
    if args.recurrent:
        # ponytail: keep the already-distilled visual encoder fixed until GRU rollout works.
        for parameter in model.vision.parameters():
            parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    stats = tuple(
        torch.tensor(value, device=device)
        for value in (state_mean, state_std, action_mean, action_std)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start_epoch = int(previous.get("epoch", 0)) if previous else 0
    same_checkpoint = previous and args.output.resolve() == args.resume.resolve()
    best = float(previous["validation_loss"]) if same_checkpoint else float("inf")
    for epoch in range(1, args.epochs + 1):
        run = _run_sequence_files if args.recurrent else _run_files
        size = args.sequence_length if args.recurrent else args.batch_size
        train_loss = run(
            model, training, size, device, stats, optimizer, args.start_step
        )
        with torch.inference_mode():
            val_loss = run(
                model, validation, size, device, stats, None, args.start_step
            )
        completed_epoch = start_epoch + epoch
        print(f"epoch={completed_epoch} train={train_loss:.6f} val={val_loss:.6f}", flush=True)
        if val_loss < best:
            best = val_loss
            torch.save(
                {
                    "version": 3,
                    "model": model.state_dict(),
                    "state_mean": state_mean.tolist(),
                    "state_std": state_std.tolist(),
                    "action_mean": action_mean.tolist(),
                    "action_std": action_std.tolist(),
                    "public_state_dim": PUBLIC_STATE_DIM,
                    "cameras": CAMERAS,
                    "uses_previous_action": True,
                    "action_horizon": args.action_horizon,
                    "action_mode": args.action_mode,
                    "recurrent": args.recurrent,
                    "start_step": args.start_step,
                    "hand_loss_weight": args.hand_loss_weight,
                    "epoch": completed_epoch,
                    "validation_loss": val_loss,
                },
                args.output,
            )
    print(args.output)


def evaluate(args) -> None:
    from dexjoco.sim.mujoco_gym_env import GymRenderingSpec
    from dexjoco.tasks.bimanual_assembly.config import TaskConfig

    policy = StudentPolicy(args.checkpoint, args.device)
    results = []
    for episode in range(args.episodes):
        seed = args.seed + episode
        env = TaskConfig().get_environment(
            policy_mode=True,
            render_mode="rgb_array",
            image_obs=True,
            randomize=args.randomize,
            randomize_dynamics=args.randomize_dynamics,
            seed=seed,
            render_spec=GymRenderingSpec(args.image_size, args.image_size),
        )
        env.unwrapped.hz = 0
        success = False
        try:
            policy.reset()
            obs, _ = env.reset(seed=seed)
            for step in range(args.max_steps):
                obs, _, terminated, truncated, info = env.step(policy.predict(obs))
                if info.get("succeed", False):
                    success = True
                    break
                if terminated or truncated:
                    break
        finally:
            env.close()
        result = {"episode": episode, "seed": seed, "success": success, "steps": step + 1}
        results.append(result)
        print(json.dumps(result))
    payload = {
        "policy": "rgb_proprio_student",
        "privileged_policy_input": False,
        "demo_replay": False,
        "success_criterion": "native info.succeed",
        "episodes": results,
        "successes": sum(item["success"] for item in results),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"success={payload['successes']}/{len(results)}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="RGB-proprio distillation for bimanual assembly")
    commands = root.add_subparsers(dest="command", required=True)

    command = commands.add_parser("collect")
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--templates", type=Path, required=True)
    command.add_argument("--student", type=Path)
    command.add_argument("--beta", type=float, default=1.0)
    command.add_argument("--episodes", type=int, default=32)
    command.add_argument("--seed", type=int, default=90_000)
    command.add_argument("--workers", type=int, default=4)
    command.add_argument("--device", default="cpu")
    command.add_argument("--image-size", type=int, default=128)
    command.add_argument("--max-steps", type=int, default=1500)
    command.add_argument("--stride", type=int, default=1)
    command.add_argument("--randomize", action="store_true")
    command.add_argument("--randomize-dynamics", action="store_true")
    command.add_argument("--overwrite", action="store_true")
    command.set_defaults(handler=collect)

    command = commands.add_parser("train")
    command.add_argument("--data", type=Path, action="append", required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--epochs", type=int, default=12)
    command.add_argument("--batch-size", type=int, default=128)
    command.add_argument("--learning-rate", type=float, default=1e-4)
    command.add_argument("--device", default="cuda")
    command.add_argument("--seed", type=int, default=0)
    command.add_argument("--resume", type=Path)
    command.add_argument("--vision-init", type=Path)
    command.add_argument("--action-horizon", type=int, default=16)
    command.add_argument(
        "--action-mode", choices=("absolute", "residual"), default="absolute"
    )
    command.add_argument("--recurrent", action="store_true")
    command.add_argument("--sequence-length", type=int, default=64)
    command.add_argument("--start-step", type=int, default=0)
    command.add_argument("--hand-loss-weight", type=float, default=0.25)
    command.add_argument("--no-pretrained", action="store_true")
    command.set_defaults(handler=train)

    command = commands.add_parser("eval")
    command.add_argument("--checkpoint", type=Path, required=True)
    command.add_argument("--episodes", type=int, default=20)
    command.add_argument("--seed", type=int, default=95_000)
    command.add_argument("--device", default="cuda")
    command.add_argument("--image-size", type=int, default=128)
    command.add_argument("--max-steps", type=int, default=1500)
    command.add_argument("--randomize", action="store_true")
    command.add_argument("--randomize-dynamics", action="store_true")
    command.add_argument("--output", type=Path)
    command.set_defaults(handler=evaluate)
    return root


if __name__ == "__main__":
    args = parser().parse_args()
    args.handler(args)
