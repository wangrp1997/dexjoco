"""Build the successful contact-manifold retrieval graph."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from retrieval_cerebellum.contact_manifold import (
    ContactManifoldConfig,
    SuccessContactManifold,
    contact_signature_matrix,
    manifold_feature_matrix,
)
from retrieval_cerebellum.demo_segments import PostGraspSegment, load_state_action_segment
from retrieval_cerebellum.learning_data import episode_split, table_columns


DEFAULT_DATASET = Path("/mnt/ssd/datasets/dexjoco_lerobot_datasets/bimanual_assembly")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--geometry-dir", type=Path, default=None)
    parser.add_argument(
        "--segments-manifest",
        type=Path,
        default=Path("outputs/retrieval_cerebellum/post_grasp_demo_audit/segments.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/retrieval_cerebellum/contact_manifold/train_graph.npz"),
    )
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--retrieval-neighbors", type=int, default=6)
    parser.add_argument("--cross-edge-max-distance", type=float, default=2.5)
    return parser.parse_args()


def _load_segments(path: Path) -> dict[int, PostGraspSegment]:
    result: dict[int, PostGraspSegment] = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            segment = PostGraspSegment(**json.loads(line))
            if segment.eligible:
                result[segment.episode_index] = segment
    return result


def main() -> None:
    args = parse_args()
    if args.stride <= 0:
        raise ValueError("stride must be positive")
    geometry_dir = args.geometry_dir or args.dataset_root / "retrieval_cerebellum_geometry"
    segments = _load_segments(args.segments_manifest)

    features_parts: list[np.ndarray] = []
    signature_parts: list[np.ndarray] = []
    episode_parts: list[np.ndarray] = []
    frame_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []
    terminal_parts: list[np.ndarray] = []
    included: list[int] = []
    skipped_splits = Counter()
    missing_geometry: list[int] = []
    for episode_index, segment in sorted(segments.items()):
        geometry_path = geometry_dir / "episodes" / f"episode_{episode_index:06d}.parquet"
        if not geometry_path.is_file():
            missing_geometry.append(episode_index)
            continue
        geometry = table_columns(geometry_path)
        family_ids = set(geometry["family_id"].tolist())
        if len(family_ids) != 1:
            raise ValueError(f"episode {episode_index} has family ids {family_ids}")
        family_id = str(next(iter(family_ids)))
        split = episode_split(episode_index, family_id, seed=args.split_seed)
        if split != "train":
            skipped_splits[split] += 1
            continue

        state_action = load_state_action_segment(args.dataset_root, segment)
        if not np.array_equal(state_action["index"], geometry["index"]):
            raise ValueError(f"episode {episode_index} state/geometry index mismatch")
        features = manifold_feature_matrix(geometry)
        signatures = contact_signature_matrix(geometry)
        terminal = np.asarray(geometry["insert_ok"], dtype=bool)
        selected = np.arange(0, len(features), args.stride, dtype=np.int64)
        required = np.flatnonzero(terminal)
        selected = np.unique(np.concatenate([selected, required, [len(features) - 1]]))

        features_parts.append(features[selected])
        signature_parts.append(signatures[selected])
        episode_parts.append(np.full(len(selected), episode_index, dtype=np.int64))
        frame_parts.append(state_action["frame_index"][selected])
        action_parts.append(state_action["action"][selected])
        terminal_parts.append(terminal[selected])
        included.append(episode_index)

    if not included:
        raise FileNotFoundError("no completed training geometry episodes are available")
    manifold = SuccessContactManifold.fit(
        features=np.concatenate(features_parts),
        contact_signatures=np.concatenate(signature_parts),
        episode_indices=np.concatenate(episode_parts),
        frame_indices=np.concatenate(frame_parts),
        actions44=np.concatenate(action_parts),
        terminal=np.concatenate(terminal_parts),
        config=ContactManifoldConfig(
            retrieval_neighbors=args.retrieval_neighbors,
            cross_edge_max_distance=args.cross_edge_max_distance,
        ),
    )
    manifold.save(args.output)
    summary = {
        "dataset_root": str(args.dataset_root),
        "geometry_dir": str(geometry_dir),
        "output": str(args.output),
        "stride": args.stride,
        "train_episodes": included,
        "num_train_episodes": len(included),
        "num_nodes": int(len(manifold.features)),
        "num_terminal_nodes": int(manifold.terminal.sum()),
        "num_reachable_nodes": int(np.isfinite(manifold.cost_to_goal).sum()),
        "temporal_edges": manifold.temporal_edge_count,
        "retrieval_edges": manifold.retrieval_edge_count,
        "missing_geometry_episodes": missing_geometry,
        "skipped_splits": dict(skipped_splits),
        "feature_dim": int(manifold.features.shape[1]),
        "action_prior_dim": int(manifold.actions44.shape[1]),
        "role": "retrieved priors for graph/contact optimization; never direct replay",
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
