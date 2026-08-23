"""Apply one P2B sensor profile to all P2A shards and audit the result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from retrieval_cerebellum.real_sensor_model import SensorDegrader, SensorModelConfig
from retrieval_cerebellum.sensor_observation import CerebellumSensorObservation


DEFAULT_DATASET = Path("/mnt/ssd/datasets/dexjoco_lerobot_datasets/bimanual_assembly")
DEFAULT_PROFILE = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "sensor_profiles"
    / "sim_stress_v1.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--estimation-dir", type=Path, default=None)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _array(table, name: str, shape: tuple[int, ...]) -> np.ndarray:
    values = np.asarray(table[name].to_pylist(), dtype=np.float32)
    expected = (table.num_rows, *shape)
    if values.shape != expected:
        raise ValueError(f"{name} expected shape {expected}, got {values.shape}")
    return values


def _stats(values: np.ndarray) -> dict[str, float | bool]:
    return {
        "finite": bool(np.isfinite(values).all()),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "zero_fraction": float(np.mean(values == 0.0)),
    }


def main() -> None:
    import pyarrow.parquet as parquet

    args = parse_args()
    estimation_dir = args.estimation_dir or args.dataset_root / "retrieval_cerebellum_estimation"
    profile = SensorModelConfig.from_json(args.profile)
    output = args.output or estimation_dir / f"sensor_model_{profile.name}_audit.json"
    paths = sorted((estimation_dir / "episodes").glob("episode_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No estimation shards under {estimation_dir}")

    degraded_state: list[np.ndarray] = []
    degraded_arm: list[np.ndarray] = []
    degraded_fingertip: list[np.ndarray] = []
    degraded_wrist: list[np.ndarray] = []
    contacts: list[np.ndarray] = []
    valid_counts = {
        "proprio": 0,
        "arm_torque": 0,
        "fingertip": 0,
        "wrist_wrench": 0,
    }
    valid_totals = {
        "proprio": 0,
        "arm_torque": 0,
        "fingertip": 0,
        "wrist_wrench": 0,
    }
    input_frames = 0
    output_frames = 0
    for path in paths:
        table = parquet.read_table(
            path,
            columns=[
                "dataset_timestamp_s",
                "sensor_state46",
                "sensor_previous_action44",
                "sensor_arm_joint_torque",
                "sensor_fingertip_force_world",
                "sensor_wrist_wrench_world",
            ],
        )
        state = _array(table, "sensor_state46", (46,))
        action = _array(table, "sensor_previous_action44", (44,))
        arm = _array(table, "sensor_arm_joint_torque", (14,)).reshape(-1, 2, 7)
        fingertip = _array(
            table,
            "sensor_fingertip_force_world",
            (24,),
        ).reshape(-1, 2, 4, 3)
        wrist = _array(table, "sensor_wrist_wrench_world", (12,)).reshape(-1, 2, 6)
        timestamp = np.asarray(table["dataset_timestamp_s"].to_numpy(), dtype=np.float64)
        episode_index = int(path.stem.rsplit("_", 1)[-1])
        degrader = SensorDegrader(
            profile,
            random_seed=profile.random_seed + episode_index,
        )
        input_frames += table.num_rows
        for row in range(table.num_rows):
            degraded = degrader.transform(
                CerebellumSensorObservation(
                    timestamp_s=float(timestamp[row]),
                    state46=state[row],
                    arm_joint_torque=arm[row],
                    fingertip_force_world=fingertip[row],
                    wrist_wrench_world=wrist[row],
                    images={},
                    previous_action44=action[row],
                )
            )
            if degraded is None:
                continue
            output_frames += 1
            degraded_state.append(degraded.state46)
            degraded_arm.append(degraded.arm_joint_torque.reshape(-1))
            degraded_fingertip.append(degraded.fingertip_force_magnitude.reshape(-1))
            degraded_wrist.append(degraded.wrist_wrench_local.reshape(-1))
            contacts.append(degraded.fingertip_contact.reshape(-1))
            valid_counts["proprio"] += int(degraded.proprio_valid)
            valid_totals["proprio"] += 1
            valid_counts["arm_torque"] += int(np.count_nonzero(degraded.arm_torque_valid))
            valid_totals["arm_torque"] += degraded.arm_torque_valid.size
            valid_counts["fingertip"] += int(np.count_nonzero(degraded.fingertip_valid))
            valid_totals["fingertip"] += degraded.fingertip_valid.size
            valid_counts["wrist_wrench"] += int(
                np.count_nonzero(degraded.wrist_wrench_valid)
            )
            valid_totals["wrist_wrench"] += degraded.wrist_wrench_valid.size

    arrays = {
        "state46": np.asarray(degraded_state),
        "arm_joint_torque": np.asarray(degraded_arm),
        "fingertip_force_magnitude": np.asarray(degraded_fingertip),
        "wrist_wrench_local": np.asarray(degraded_wrist),
    }
    contact_values = np.asarray(contacts, dtype=bool)
    audit = {
        "profile": profile.to_dict(),
        "deployment_ready": bool(profile.hardware_verified),
        "input_episodes": len(paths),
        "input_frames": input_frames,
        "output_frames": output_frames,
        "dropped_warmup_frames": input_frames - output_frames,
        "output_schema": {
            "state46": 46,
            "arm_joint_torque": 14,
            "fingertip_force_magnitude": 8,
            "fingertip_contact": 8,
            "wrist_wrench_local": 12,
        },
        "channel_stats": {name: _stats(values) for name, values in arrays.items()},
        "fingertip_contact_rate": float(np.mean(contact_values)),
        "valid_rates": {
            name: float(valid_counts[name] / valid_totals[name]) for name in valid_counts
        },
        "known_hardware_gaps": [
            "target arm driver joint-torque interface and timestamp semantics",
            "target wrist F/T sensor model, mounting transform and calibration",
            "target fingertip tactile model, taxel layout and force reconstruction",
            "camera hardware timestamps and cross-device synchronization",
            "measured noise, bias drift, saturation, dropout and end-to-end latency",
        ],
        "interpretation": (
            "This profile is a robustness perturbation benchmark, not a measured "
            "real-robot sensor specification."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
