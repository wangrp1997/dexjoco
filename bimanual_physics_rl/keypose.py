from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .student import (
    CAMERAS,
    PUBLIC_STATE_DIM,
    StudentNet,
    _loss,
    _tensor_batch,
    absolute_to_action,
    action_to_absolute,
    public_observation,
    state_to_absolute,
)


KEY_STEPS = (0, 40, 78, 110, 150, 181, 182, 250, 330, 393, 430, 485)


def _component_features(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if count < 2:
        return np.zeros(20, dtype=np.float32)
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    moments = cv2.moments((labels == component).astype(np.uint8))
    hu = cv2.HuMoments(moments).ravel()
    height, width = mask.shape
    x, y, box_width, box_height, area = stats[component]
    return np.asarray(
        [
            1.0,
            centroids[component, 0] / width,
            centroids[component, 1] / height,
            area / (height * width),
            x / width,
            y / height,
            box_width / width,
            box_height / height,
            *(
                moments[name]
                for name in (
                    "nu20",
                    "nu11",
                    "nu02",
                    "nu30",
                    "nu21",
                    "nu12",
                    "nu03",
                )
            ),
            *(np.sign(hu[:5]) * np.log1p(np.abs(hu[:5]))),
        ],
        dtype=np.float32,
    )


def _peg_visual_features(images: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            _component_features(
                cv2.inRange(
                    cv2.cvtColor(image, cv2.COLOR_RGB2HSV),
                    (15, 80, 80),
                    (45, 255, 255),
                )
            )
            for image in images
        ]
    )


def _fit_peg_visual_head(images: np.ndarray, targets: np.ndarray) -> dict:
    features = np.stack([_peg_visual_features(value) for value in images])
    rotations = Rotation.from_rotvec(targets[:, 6:, 3:6].reshape(-1, 3))
    outputs = np.concatenate(
        [
            targets[:, 6:, :3].reshape(len(targets), -1),
            rotations.as_matrix().reshape(len(targets), -1),
        ],
        axis=1,
    )
    feature_mean = features.mean(axis=0)
    feature_std = np.maximum(features.std(axis=0), 1e-6)
    output_mean = outputs.mean(axis=0)
    output_std = np.maximum(outputs.std(axis=0), 1e-6)
    normalized_features = (features - feature_mean) / feature_std
    normalized_outputs = (outputs - output_mean) / output_std
    ridge = normalized_features.T @ normalized_features
    ridge.flat[:: ridge.shape[0] + 1] += 10.0
    weights = np.linalg.solve(
        ridge, normalized_features.T @ normalized_outputs
    ).astype(np.float32)
    return {
        "feature_mean": feature_mean.astype(np.float32),
        "feature_std": feature_std.astype(np.float32),
        "output_mean": output_mean.astype(np.float32),
        "output_std": output_std.astype(np.float32),
        "weights": weights,
    }


def _apply_peg_visual_head(
    targets: np.ndarray, images: np.ndarray, head: dict
) -> None:
    features = _peg_visual_features(images)
    output = (
        (features - head["feature_mean"])
        / head["feature_std"]
        @ head["weights"]
    ) * head["output_std"] + head["output_mean"]
    targets[6:, :3] = output[:18].reshape(6, 3)
    matrices = output[18:].reshape(6, 3, 3)
    for index, matrix in enumerate(matrices):
        left, _, right = np.linalg.svd(matrix)
        rotation = left @ right
        if np.linalg.det(rotation) < 0:
            left[:, -1] *= -1
            rotation = left @ right
        targets[index + 6, 3:6] = Rotation.from_matrix(rotation).as_rotvec()


def _collect_group(spec: dict) -> list[dict]:
    from dexjoco.sim.mujoco_gym_env import GymRenderingSpec
    from dexjoco.tasks.bimanual_assembly.config import TaskConfig

    from .causal import CausalAssemblyController

    env = TaskConfig().get_environment(
        policy_mode=True,
        render_mode="rgb_array",
        image_obs=True,
        randomize=False,
        randomize_dynamics=False,
        seed=spec["seeds"][0],
        render_spec=GymRenderingSpec(spec["image_size"], spec["image_size"]),
    )
    env.unwrapped.hz = 0
    results = []
    try:
        for seed in spec["seeds"]:
            output = Path(spec["output"]) / f"keypose_seed{seed:06d}.npz"
            if output.exists() and not spec["overwrite"]:
                results.append({"seed": seed, "file": str(output), "skipped": True})
                continue
            random.seed(seed)
            np.random.seed(seed)
            obs, _ = env.reset(seed=seed)
            images, state = public_observation(obs)
            teacher = CausalAssemblyController(env.unwrapped, Path(spec["templates"]))
            teacher.reset()
            targets = np.stack(
                [action_to_absolute(teacher.action(step)) for step in KEY_STEPS]
            )
            np.savez_compressed(
                output,
                images=images,
                state=state,
                target=targets.astype(np.float32),
            )
            results.append({"seed": seed, "file": str(output), "skipped": False})
    finally:
        env.close()
    return results


def collect(args) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    groups = [
        list(range(args.seed + worker, args.seed + args.episodes, args.workers))
        for worker in range(args.workers)
    ]
    specs = [
        {
            "seeds": group,
            "output": str(args.output),
            "templates": str(args.templates),
            "image_size": args.image_size,
            "overwrite": args.overwrite,
        }
        for group in groups
        if group
    ]
    results = []
    if len(specs) == 1:
        results = _collect_group(specs[0])
    else:
        with ProcessPoolExecutor(
            max_workers=len(specs), mp_context=mp.get_context("spawn")
        ) as pool:
            futures = [pool.submit(_collect_group, spec) for spec in specs]
            for future in as_completed(futures):
                batch = future.result()
                results.extend(batch)
                print(f"collected={len(results)}/{args.episodes}", flush=True)
    results.sort(key=lambda item: item["seed"])
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "teacher_is_privileged": True,
                "policy_inputs": [*CAMERAS, "tcp_pose", "gripper_pose"],
                "forbidden_policy_inputs": [
                    "object_pose",
                    "contact_truth",
                    "demo_action",
                    "phase_index",
                ],
                "key_steps": KEY_STEPS,
                "episodes": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(args.output)


def _load_dataset(paths: list[Path]):
    files = sorted(file for path in paths for file in path.glob("keypose_seed*.npz"))
    if len(files) < 10:
        raise ValueError("At least 10 keypose samples are required")
    images, states, targets, seeds = [], [], [], []
    for path in files:
        with np.load(path, allow_pickle=False) as sample:
            images.append(sample["images"])
            states.append(sample["state"])
            targets.append(sample["target"])
        seeds.append(int(path.stem.removeprefix("keypose_seed")))
    return (
        np.stack(images),
        np.stack(states),
        np.stack(targets),
        np.asarray(seeds),
    )


def _mean_std(values: np.ndarray, floor: float = 1e-4):
    return values.mean(axis=0).astype(np.float32), np.maximum(
        values.std(axis=0), floor
    ).astype(np.float32)


def train(args) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    images, states, targets, seeds = _load_dataset(args.data)
    validation = seeds % 5 == 0
    training = ~validation
    peg_visual_head = _fit_peg_visual_head(images[training], targets[training])
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
        state_mean, state_std = _mean_std(states[training])
        action_mean, action_std = _mean_std(targets[training].reshape(-1, 44))
    device = torch.device(args.device)
    model = StudentNet(action_horizon=len(KEY_STEPS), action_mode="absolute").to(device)
    if previous:
        model.load_state_dict(previous["model"])
    elif args.vision_init:
        source = torch.load(args.vision_init, map_location="cpu", weights_only=False)
        model.load_state_dict(
            {
                key: value
                for key, value in source["model"].items()
                if key.startswith("vision.")
            },
            strict=False,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    stats = tuple(
        torch.tensor(value, device=device)
        for value in (state_mean, state_std, action_mean, action_std)
    )
    start_epoch = int(previous.get("epoch", 0)) if previous else 0
    same_checkpoint = previous and args.output.resolve() == args.resume.resolve()
    best = float(previous["validation_loss"]) if same_checkpoint else float("inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.permutation(np.flatnonzero(training))
        losses = []
        for start in range(0, len(order), args.batch_size):
            batch = order[start : start + args.batch_size]
            previous_actions = np.stack([state_to_absolute(state) for state in states[batch]])
            image_tensor, state_tensor, previous_tensor, target_tensor = _tensor_batch(
                images[batch],
                states[batch],
                previous_actions,
                targets[batch],
                device,
                *stats,
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = _loss(
                    model(image_tensor, state_tensor, previous_tensor), target_tensor
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            batch = np.flatnonzero(validation)
            previous_actions = np.stack([state_to_absolute(state) for state in states[batch]])
            image_tensor, state_tensor, previous_tensor, target_tensor = _tensor_batch(
                images[batch],
                states[batch],
                previous_actions,
                targets[batch],
                device,
                *stats,
            )
            val_loss = float(
                _loss(model(image_tensor, state_tensor, previous_tensor), target_tensor)
            )
        completed_epoch = start_epoch + epoch
        print(
            f"epoch={completed_epoch} train={np.mean(losses):.6f} val={val_loss:.6f}",
            flush=True,
        )
        if val_loss < best:
            best = val_loss
            torch.save(
                {
                    "version": 1,
                    "model": model.state_dict(),
                    "state_mean": state_mean.tolist(),
                    "state_std": state_std.tolist(),
                    "action_mean": action_mean.tolist(),
                    "action_std": action_std.tolist(),
                    "public_state_dim": PUBLIC_STATE_DIM,
                    "cameras": CAMERAS,
                    "uses_previous_action": True,
                    "action_horizon": len(KEY_STEPS),
                    "action_mode": "absolute",
                    "key_steps": KEY_STEPS,
                    "peg_visual_head": peg_visual_head,
                    "epoch": completed_epoch,
                    "validation_loss": val_loss,
                },
                args.output,
            )
    print(args.output)


def _interpolate(a: np.ndarray, b: np.ndarray, ratio: float) -> np.ndarray:
    output = (1.0 - ratio) * a + ratio * b
    for offset in (0, 22):
        first = Rotation.from_rotvec(a[offset + 3 : offset + 6])
        second = Rotation.from_rotvec(b[offset + 3 : offset + 6])
        relative = (first.inv() * second).as_rotvec()
        output[offset + 3 : offset + 6] = (
            first * Rotation.from_rotvec(ratio * relative)
        ).as_rotvec()
    return output


def _trajectory(initial: np.ndarray, targets: np.ndarray) -> list[np.ndarray]:
    actions = []
    previous_step = -1
    previous = initial
    for step, target in zip(KEY_STEPS, targets):
        duration = step - previous_step
        actions.extend(
            _interpolate(previous, target, (index + 1) / duration)
            for index in range(duration)
        )
        previous_step = step
        previous = target
    return actions


def evaluate(args) -> None:
    from dexjoco.sim.mujoco_gym_env import GymRenderingSpec
    from dexjoco.tasks.bimanual_assembly.config import TaskConfig
    from interaction_retarget.sim.contact import AssemblyContactDetector

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = torch.device(args.device)
    model = StudentNet(action_horizon=len(KEY_STEPS), action_mode="absolute").to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    stats = tuple(
        torch.tensor(payload[key], device=device)
        for key in ("state_mean", "state_std", "action_mean", "action_std")
    )
    results = []
    for episode in range(args.episodes):
        seed = args.seed + episode
        env = TaskConfig().get_environment(
            policy_mode=True,
            render_mode="rgb_array",
            image_obs=True,
            randomize=False,
            randomize_dynamics=args.randomize_dynamics,
            seed=seed,
            render_spec=GymRenderingSpec(args.image_size, args.image_size),
        )
        raw = env.unwrapped
        raw.hz = 0
        detector = AssemblyContactDetector(raw)
        try:
            obs, _ = env.reset(seed=seed)
            detector.reset_reference(raw)
            initial_z = raw._data.xpos[[raw._peg_body_id, raw._socket_body_id], 2].copy()
            images, state = public_observation(obs)
            previous = state_to_absolute(state)
            image_tensor, state_tensor, previous_tensor, _ = _tensor_batch(
                images[None], state[None], previous[None], None, device, *stats
            )
            with torch.inference_mode():
                normalized = model(image_tensor, state_tensor, previous_tensor)[0].clamp(
                    -5.0, 5.0
                )
                targets = (normalized * stats[3] + stats[2]).float().cpu().numpy()
            if "peg_visual_head" in payload:
                _apply_peg_visual_head(targets, images, payload["peg_visual_head"])
            for command in _trajectory(previous, targets):
                obs, _, terminated, truncated, _ = env.step(absolute_to_action(command))
                if terminated or truncated:
                    break
            for _ in range(args.hold_steps):
                obs, _, terminated, truncated, _ = env.step(
                    absolute_to_action(targets[-1])
                )
                if terminated or truncated:
                    break
            contact = detector.compute(raw)
            lift = raw._data.xpos[[raw._peg_body_id, raw._socket_body_id], 2] - initial_z
            result = {
                "episode": episode,
                "seed": seed,
                "dual_grasp": bool(contact.peg_contact and contact.tray_contact),
                "peg_contact": bool(contact.peg_contact),
                "tray_contact": bool(contact.tray_contact),
                "peg_lift_m": float(lift[0]),
                "socket_lift_m": float(lift[1]),
            }
        finally:
            env.close()
        results.append(result)
        print(json.dumps(result), flush=True)
    output = {
        "policy_inputs_privileged": False,
        "demo_replay": False,
        "diagnostics_use_privileged_state": True,
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n")
    dual = sum(item["dual_grasp"] for item in results)
    print(f"dual_grasp={dual}/{len(results)}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Visual keypose distillation for grasping")
    commands = root.add_subparsers(dest="command", required=True)

    command = commands.add_parser("collect")
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--templates", type=Path, required=True)
    command.add_argument("--episodes", type=int, default=512)
    command.add_argument("--seed", type=int, default=120_000)
    command.add_argument("--workers", type=int, default=4)
    command.add_argument("--image-size", type=int, default=128)
    command.add_argument("--overwrite", action="store_true")
    command.set_defaults(handler=collect)

    command = commands.add_parser("train")
    command.add_argument("--data", type=Path, action="append", required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--vision-init", type=Path)
    command.add_argument("--resume", type=Path)
    command.add_argument("--epochs", type=int, default=25)
    command.add_argument("--batch-size", type=int, default=64)
    command.add_argument("--learning-rate", type=float, default=1e-4)
    command.add_argument("--device", default="cuda")
    command.add_argument("--seed", type=int, default=0)
    command.set_defaults(handler=train)

    command = commands.add_parser("eval")
    command.add_argument("--checkpoint", type=Path, required=True)
    command.add_argument("--episodes", type=int, default=20)
    command.add_argument("--seed", type=int, default=130_000)
    command.add_argument("--device", default="cuda")
    command.add_argument("--image-size", type=int, default=128)
    command.add_argument("--hold-steps", type=int, default=30)
    command.add_argument("--randomize-dynamics", action="store_true")
    command.add_argument("--output", type=Path)
    command.set_defaults(handler=evaluate)
    return root


if __name__ == "__main__":
    args = parser().parse_args()
    args.handler(args)
