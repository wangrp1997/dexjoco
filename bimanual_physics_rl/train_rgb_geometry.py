from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18


ROOT = Path("/mnt/hdd/dexjoco/outputs/bimanual_physics_rl/student_rgbp_v1")
DATA = ROOT / "servo_rgb_geometry_rollouts_v3"
DAGGER = ROOT / "servo_rgb_geometry_dagger_v4"
DAGGER_V5 = ROOT / "servo_rgb_geometry_dagger_v5"
OUTPUT = ROOT / "visual_geometry_rgb_dagger_v5.pt"


class GeometryNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision = resnet18(weights=None)
        self.vision.fc = nn.Identity()
        self.context = nn.Sequential(
            nn.Linear(90, 128), nn.LayerNorm(128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU()
        )
        self.head = nn.Sequential(
            nn.Linear(6 * 512 + 128, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, 9),
        )
        self.register_buffer(
            "image_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        )

    def forward(self, images, state, grasp):
        batch, views, channels, height, width = images.shape
        pixels = images.reshape(batch * views, channels, height, width)
        visual = self.vision((pixels - self.image_mean) / self.image_std)
        visual = visual.reshape(batch, views * visual.shape[-1])
        return self.head(torch.cat([visual, self.context(torch.cat([state, grasp], -1))], -1))


def moments(files, key):
    values = []
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            if key == "grasp_target":
                values.append(data[key][None])
            else:
                values.append(data[key])
    values = np.concatenate(values).astype(np.float32)
    return values.mean(0), np.maximum(values.std(0), 1e-5)


def batch_tensors(data, indices, device, stats):
    initial = np.broadcast_to(data["initial_images"], (len(indices), 3, 128, 128, 3))
    images = np.concatenate([initial, data["images"][indices]], axis=1)
    images = (
        torch.from_numpy(np.ascontiguousarray(images))
        .to(device=device, dtype=torch.float32)
        .permute(0, 1, 4, 2, 3)
        .div_(255.0)
    )
    state = torch.from_numpy(np.ascontiguousarray(data["state"][indices])).to(
        device=device, dtype=torch.float32
    )
    grasp = torch.from_numpy(
        np.broadcast_to(data["grasp_target"], (len(indices), 44)).copy()
    ).to(device=device, dtype=torch.float32)
    target = torch.from_numpy(np.ascontiguousarray(data["geometry"][indices])).to(
        device=device, dtype=torch.float32
    )
    sm, ss, gm, gs, ym, ys = stats
    return images, (state - sm) / ss, (grasp - gm) / gs, (target - ym) / ys


def run(model, files, device, stats, optimizer=None):
    training = optimizer is not None
    model.train(training)
    files = list(files)
    if training:
        random.shuffle(files)
    losses = []
    errors = []
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    for path in files:
        with np.load(path, allow_pickle=False) as source:
            data = {key: source[key] for key in source.files}
        order = np.random.permutation(len(data["state"])) if training else np.arange(len(data["state"]))
        for start in range(0, len(order), 64):
            batch = order[start : start + 64]
            images, state, grasp, target = batch_tensors(data, batch, device, stats)
            with torch.set_grad_enabled(training), amp:
                prediction = model(images, state, grasp)
                loss = F.smooth_l1_loss(prediction, target, beta=0.2)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            losses.append(float(loss.detach()))
            if not training:
                errors.append(((prediction.float() - target) * stats[-1]).cpu().numpy())
    if training:
        return float(np.mean(losses)), None
    error = np.concatenate(errors)
    metrics = (
        np.linalg.norm(error[:, :3], axis=1).mean() * 1000,
        np.degrees(np.linalg.norm(error[:, 3:6], axis=1).mean()),
        np.linalg.norm(error[:, 6:9], axis=1).mean() * 1000,
    )
    return float(np.mean(losses)), metrics


if __name__ == "__main__":
    random.seed(33)
    np.random.seed(33)
    torch.manual_seed(33)
    base_files = []
    for path in sorted(DATA.glob("episode_seed*.npz")):
        with np.load(path, allow_pickle=False) as data:
            if bool(data["success"]):
                base_files.append(path)
    dagger_files = sorted(DAGGER.glob("episode_seed*.npz")) + sorted(
        DAGGER_V5.glob("episode_seed*.npz")
    )
    files = base_files + dagger_files
    validation = [path for path in files if int(path.stem[12:]) % 5 == 0]
    training = [path for path in files if path not in set(validation)]
    previous = torch.load(
        ROOT / "visual_geometry_rgb_dagger_v4.pt",
        map_location="cpu",
        weights_only=False,
    )
    values = [
        np.asarray(previous[key], dtype=np.float32)
        for key in (
            "state_mean",
            "state_std",
            "grasp_mean",
            "grasp_std",
            "geometry_mean",
            "geometry_std",
        )
    ]
    device = torch.device("cuda")
    stats = tuple(torch.tensor(value, device=device) for value in values)
    model = GeometryNet().to(device)
    model.load_state_dict(previous["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
    best = float("inf")
    for epoch in range(1, 7):
        train_loss, _ = run(model, training, device, stats, optimizer)
        with torch.inference_mode():
            val_loss, metrics = run(model, validation, device, stats)
        print(
            f"epoch={epoch} train={train_loss:.4f} val={val_loss:.4f} "
            f"center_mm={metrics[0]:.2f} orient_deg={metrics[1]:.2f} axial_mm={metrics[2]:.2f}",
            flush=True,
        )
        if val_loss < best:
            best = val_loss
            torch.save(
                {
                    "version": 3,
                    "model": model.state_dict(),
                    "state_mean": values[0].tolist(),
                    "state_std": values[1].tolist(),
                    "grasp_mean": values[2].tolist(),
                    "grasp_std": values[3].tolist(),
                    "geometry_mean": values[4].tolist(),
                    "geometry_std": values[5].tolist(),
                    "validation_loss": val_loss,
                    "epoch": epoch,
                    "policy_inputs": ["initial_rgb", "current_rgb", "state46", "own_grasp_target"],
                    "privileged_policy_input": False,
                    "privileged_training_labels": True,
                    "demo_replay": False,
                },
                OUTPUT,
            )
    print(OUTPUT)
