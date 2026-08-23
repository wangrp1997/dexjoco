"""Run the sensor-belief-retrieval-deformation prototype on held-out episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from retrieval_cerebellum.belief_estimation import RidgeObservationModel
from retrieval_cerebellum.finger_kinematics import AllegroFingertipKinematics
from retrieval_cerebellum.learning_data import state46_to_action44
from retrieval_cerebellum.real_sensor_model import SensorModelConfig
from retrieval_cerebellum.scripts.train_belief_baseline import _episode_data
from retrieval_cerebellum.skill_prototype import (
    RetrievalAugmentedSkillPrototype,
    SuccessfulSkillMemory,
)
from retrieval_cerebellum.visual_initialization import EpisodeVisualFeatureStore


DEFAULT_DATASET = Path("/mnt/ssd/datasets/dexjoco_lerobot_datasets/bimanual_assembly")
PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--learning-dir", type=Path, default=None)
    parser.add_argument("--estimation-dir", type=Path, default=None)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "outputs/retrieval_cerebellum/"
            "belief_baseline_kin_visual_pca4_a1000/observation_model.npz"
        ),
    )
    parser.add_argument(
        "--visual-cache",
        type=Path,
        default=Path(
            "outputs/retrieval_cerebellum/visual_initialization/clip_vit_b16_pca4.npz"
        ),
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=PACKAGE_ROOT / "configs" / "sensor_profiles" / "sim_stress_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/retrieval_cerebellum/skill_prototype"),
    )
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--step-quantile", type=float, default=0.95)
    return parser.parse_args()


def _first_sensor_state(path: Path) -> np.ndarray:
    import pyarrow.parquet as parquet

    table = parquet.read_table(path, columns=["sensor_state46"])
    return np.asarray(table["sensor_state46"][0].as_py(), dtype=np.float32)


def _position_jump(action: np.ndarray, current: np.ndarray) -> float:
    return float(
        max(
            np.linalg.norm(action[:3] - current[:3]),
            np.linalg.norm(action[22:25] - current[22:25]),
        )
    )


def main() -> None:
    args = parse_args()
    learning_dir = args.learning_dir or args.dataset_root / "retrieval_cerebellum_learning"
    estimation_dir = args.estimation_dir or args.dataset_root / "retrieval_cerebellum_estimation"
    memory = SuccessfulSkillMemory.load(
        learning_dir,
        step_quantile=args.step_quantile,
    )
    prototype = RetrievalAugmentedSkillPrototype(memory)
    model = RidgeObservationModel.load(args.model)
    profile = SensorModelConfig.from_json(args.profile)
    visual_store = EpisodeVisualFeatureStore.load(args.visual_cache)
    kinematics = AllegroFingertipKinematics()

    records = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted((estimation_dir / "episodes").glob("episode_*.parquet")):
        split, features, target = _episode_data(
            path,
            profile,
            history_size=model.history_size,
            kinematics=kinematics,
            visual_store=visual_store,
        )
        if split == "train":
            continue
        episode_index = int(path.stem.rsplit("_", 1)[-1])
        prediction = model.predict(features[:1])[0]
        state46 = _first_sensor_state(path)
        plan = prototype.plan(
            prediction,
            state46,
            family_id="round_8mm",
            horizon=args.horizon,
        )
        current_action = np.asarray(
            memory.trajectories[plan.source_episode_index].proprio_action44[0]
        )
        actual_current = state46_to_action44(state46)
        plan_path = args.output_dir / f"episode_{episode_index:06d}_plan.npz"
        np.savez_compressed(
            plan_path,
            predicted_belief18=prediction,
            teacher_belief18=target[0],
            source_episode_index=np.asarray(plan.source_episode_index),
            raw_actions44=plan.raw_actions44,
            adapted_actions44=plan.adapted_actions44,
        )
        records.append(
            {
                "episode_index": episode_index,
                "split": split,
                "source_episode_index": plan.source_episode_index,
                "retrieval_distance": plan.retrieval_distance,
                "horizon": len(plan.adapted_actions44),
                "direct_replay_initial_position_jump_m": _position_jump(
                    plan.raw_actions44[0], actual_current
                ),
                "adapted_initial_position_jump_m": _position_jump(
                    plan.adapted_actions44[0], actual_current
                ),
                "source_initial_position_gap_m": _position_jump(
                    current_action, actual_current
                ),
                "belief_position_error_m": float(
                    np.mean(
                        [
                            np.linalg.norm(prediction[offset : offset + 3] - target[0, offset : offset + 3])
                            for offset in (0, 6, 12)
                        ]
                    )
                ),
                "plan_path": str(plan_path),
            }
        )
        print(
            f"episode={episode_index} split={split} source={plan.source_episode_index} "
            f"distance={plan.retrieval_distance:.3f}",
            flush=True,
        )

    if not records:
        raise ValueError("no validation or test episodes were evaluated")
    summary = {
        "stage": "DexContactRAM core prototype",
        "pipeline": "sensor belief -> successful skill retrieval -> data-driven projection",
        "num_gallery_episodes": len(memory.trajectories),
        "num_held_out_episodes": len(records),
        "horizon": args.horizon,
        "step_quantile": args.step_quantile,
        "data_driven_limits": {
            "right_position_step_m": memory.limits.right_position_step_m,
            "left_position_step_m": memory.limits.left_position_step_m,
            "right_rotation_step_rad": memory.limits.right_rotation_step_rad,
            "left_rotation_step_rad": memory.limits.left_rotation_step_rad,
        },
        "mean_direct_replay_initial_position_jump_m": float(
            np.mean([item["direct_replay_initial_position_jump_m"] for item in records])
        ),
        "mean_adapted_initial_position_jump_m": float(
            np.mean([item["adapted_initial_position_jump_m"] for item in records])
        ),
        "records": records,
        "limitations": [
            "offline shadow prototype only; actions are not executed",
            "retrieval uses the current P3 estimated belief",
            "projection enforces learned motion limits but not yet full robot/contact constraints",
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
