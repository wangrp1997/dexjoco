"""Retrieval graph over successful dexterous-assembly contact states."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.spatial import cKDTree


MANIFOLD_VECTOR_COLUMNS = (
    "peg_tip_in_hole_position",
    "peg_in_hole_rotvec",
    "peg_in_right_palm_position",
    "peg_in_right_palm_rotvec",
    "tray_in_left_palm_position",
    "tray_in_left_palm_rotvec",
)
MANIFOLD_SCALAR_COLUMNS = (
    "lateral_error_m",
    "axis_error_rad",
    "approach_height_m",
    "insertion_depth_m",
    "peg_contact_count",
    "tray_contact_count",
)
MANIFOLD_FEATURE_DIM = 3 * len(MANIFOLD_VECTOR_COLUMNS) + len(
    MANIFOLD_SCALAR_COLUMNS
)
CONTACT_SIGNATURE_DIM = 5


def manifold_feature_matrix(columns: Mapping[str, np.ndarray]) -> np.ndarray:
    """Build the continuous object-centric state used by graph retrieval."""
    parts: list[np.ndarray] = []
    row_count: int | None = None
    for name in MANIFOLD_VECTOR_COLUMNS:
        values = np.asarray(columns[name], dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError(f"{name} must have shape (T, 3), got {values.shape}")
        row_count = values.shape[0] if row_count is None else row_count
        if values.shape[0] != row_count:
            raise ValueError("manifold columns must have equal row counts")
        parts.append(values)
    for name in MANIFOLD_SCALAR_COLUMNS:
        values = np.asarray(columns[name], dtype=np.float32).reshape(-1, 1)
        row_count = values.shape[0] if row_count is None else row_count
        if values.shape[0] != row_count:
            raise ValueError("manifold columns must have equal row counts")
        parts.append(values)
    features = np.concatenate(parts, axis=1)
    if features.shape[1] != MANIFOLD_FEATURE_DIM:
        raise RuntimeError(f"unexpected manifold feature shape {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError("manifold features contain non-finite values")
    return features


def contact_signature_matrix(columns: Mapping[str, np.ndarray]) -> np.ndarray:
    """Encode discrete grasp and contact modes without distance thresholds."""
    signature = np.column_stack(
        [
            np.asarray(columns["peg_ok"], dtype=bool),
            np.asarray(columns["tray_ok"], dtype=bool),
            np.asarray(columns["insert_ok"], dtype=bool),
            np.asarray(columns["peg_contact_count"], dtype=np.int32) > 0,
            np.asarray(columns["tray_contact_count"], dtype=np.int32) > 0,
        ]
    ).astype(np.int8)
    if signature.ndim != 2 or signature.shape[1] != CONTACT_SIGNATURE_DIM:
        raise RuntimeError(f"unexpected contact signature shape {signature.shape}")
    return signature


@dataclass(frozen=True)
class ContactManifoldConfig:
    retrieval_neighbors: int = 6
    retrieval_search_neighbors: int = 48
    cross_edge_max_distance: float = 2.5
    cross_edge_penalty: float = 0.25
    temporal_step_penalty: float = 0.02
    attachment_neighbors: int = 24
    attachment_distance_weight: float = 2.0

    def __post_init__(self) -> None:
        if self.retrieval_neighbors < 0:
            raise ValueError("retrieval_neighbors must be non-negative")
        if self.retrieval_search_neighbors <= 0:
            raise ValueError("retrieval_search_neighbors must be positive")
        if self.cross_edge_max_distance <= 0:
            raise ValueError("cross_edge_max_distance must be positive")
        if self.cross_edge_penalty < 0 or self.temporal_step_penalty < 0:
            raise ValueError("edge penalties must be non-negative")
        if self.attachment_neighbors <= 0:
            raise ValueError("attachment_neighbors must be positive")


@dataclass(frozen=True)
class ContactManifoldPlan:
    node_indices: np.ndarray
    episode_indices: np.ndarray
    frame_indices: np.ndarray
    action_priors44: np.ndarray
    bridge_target: np.ndarray
    attachment_distance: float
    graph_cost_to_goal: float
    reached_terminal: bool


class SuccessContactManifold:
    """Graph-optimized memory of successful contact-state evolution.

    Temporal edges preserve demonstrated progress. Cross-episode edges connect
    geometrically similar states with the same discrete contact signature, so
    planning can switch between locally compatible successful demonstrations.
    """

    def __init__(
        self,
        *,
        features: np.ndarray,
        normalized_features: np.ndarray,
        contact_signatures: np.ndarray,
        episode_indices: np.ndarray,
        frame_indices: np.ndarray,
        actions44: np.ndarray,
        terminal: np.ndarray,
        feature_mean: np.ndarray,
        feature_scale: np.ndarray,
        cost_to_goal: np.ndarray,
        next_node: np.ndarray,
        temporal_edge_count: int,
        retrieval_edge_count: int,
        config: ContactManifoldConfig,
    ) -> None:
        self.features = np.asarray(features, dtype=np.float32)
        self.normalized_features = np.asarray(normalized_features, dtype=np.float32)
        self.contact_signatures = np.asarray(contact_signatures, dtype=np.int8)
        self.episode_indices = np.asarray(episode_indices, dtype=np.int64)
        self.frame_indices = np.asarray(frame_indices, dtype=np.int64)
        self.actions44 = np.asarray(actions44, dtype=np.float32)
        self.terminal = np.asarray(terminal, dtype=bool)
        self.feature_mean = np.asarray(feature_mean, dtype=np.float32)
        self.feature_scale = np.asarray(feature_scale, dtype=np.float32)
        self.cost_to_goal = np.asarray(cost_to_goal, dtype=np.float64)
        self.next_node = np.asarray(next_node, dtype=np.int64)
        self.temporal_edge_count = int(temporal_edge_count)
        self.retrieval_edge_count = int(retrieval_edge_count)
        self.config = config
        self._tree = cKDTree(self.normalized_features)

    @classmethod
    def fit(
        cls,
        *,
        features: np.ndarray,
        contact_signatures: np.ndarray,
        episode_indices: np.ndarray,
        frame_indices: np.ndarray,
        actions44: np.ndarray,
        terminal: np.ndarray,
        config: ContactManifoldConfig | None = None,
    ) -> "SuccessContactManifold":
        cfg = config or ContactManifoldConfig()
        features = np.asarray(features, dtype=np.float32)
        signatures = np.asarray(contact_signatures, dtype=np.int8)
        episodes = np.asarray(episode_indices, dtype=np.int64).reshape(-1)
        frames = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
        actions = np.asarray(actions44, dtype=np.float32)
        terminal_mask = np.asarray(terminal, dtype=bool).reshape(-1)
        node_count = features.shape[0]
        if features.ndim != 2 or features.shape[1] != MANIFOLD_FEATURE_DIM:
            raise ValueError(
                f"features must have shape (N, {MANIFOLD_FEATURE_DIM}), got {features.shape}"
            )
        if signatures.shape != (node_count, CONTACT_SIGNATURE_DIM):
            raise ValueError(
                f"contact_signatures must have shape (N, {CONTACT_SIGNATURE_DIM})"
            )
        if actions.shape != (node_count, 44):
            raise ValueError(f"actions44 must have shape (N, 44), got {actions.shape}")
        if any(values.shape[0] != node_count for values in (episodes, frames, terminal_mask)):
            raise ValueError("all manifold arrays must have equal row counts")
        if node_count == 0 or not terminal_mask.any():
            raise ValueError("manifold requires nodes and at least one terminal")
        if not np.isfinite(features).all() or not np.isfinite(actions).all():
            raise ValueError("manifold arrays contain non-finite values")

        mean = features.mean(axis=0, dtype=np.float64)
        scale = features.std(axis=0, dtype=np.float64)
        scale = np.where(scale > 1e-6, scale, 1.0)
        normalized = ((features - mean) / scale).astype(np.float32)
        tree = cKDTree(normalized)

        edges: list[tuple[int, int, float]] = []
        temporal_edge_count = 0
        for source in range(node_count - 1):
            target = source + 1
            if episodes[source] != episodes[target] or frames[target] <= frames[source]:
                continue
            distance = float(np.linalg.norm(normalized[target] - normalized[source]))
            edges.append((source, target, cfg.temporal_step_penalty + distance))
            temporal_edge_count += 1

        retrieval_edge_count = 0
        if cfg.retrieval_neighbors > 0 and node_count > 1:
            search_k = min(node_count, cfg.retrieval_search_neighbors + 1)
            distances, neighbors = tree.query(normalized, k=search_k)
            if search_k == 1:
                distances = distances[:, None]
                neighbors = neighbors[:, None]
            for source in range(node_count):
                accepted = 0
                for distance, target in zip(distances[source], neighbors[source], strict=True):
                    target = int(target)
                    if target == source or episodes[target] == episodes[source]:
                        continue
                    if distance > cfg.cross_edge_max_distance:
                        continue
                    if not np.array_equal(signatures[target], signatures[source]):
                        continue
                    edges.append((source, target, cfg.cross_edge_penalty + float(distance)))
                    retrieval_edge_count += 1
                    accepted += 1
                    if accepted >= cfg.retrieval_neighbors:
                        break

        reverse_edges: list[list[tuple[int, float]]] = [[] for _ in range(node_count)]
        for source, target, cost in edges:
            reverse_edges[target].append((source, cost))
        cost_to_goal = np.full(node_count, np.inf, dtype=np.float64)
        next_node = np.full(node_count, -1, dtype=np.int64)
        queue: list[tuple[float, int]] = []
        for node in np.flatnonzero(terminal_mask):
            cost_to_goal[node] = 0.0
            heapq.heappush(queue, (0.0, int(node)))
        while queue:
            current_cost, target = heapq.heappop(queue)
            if current_cost != cost_to_goal[target]:
                continue
            for source, edge_cost in reverse_edges[target]:
                candidate = current_cost + edge_cost
                if candidate >= cost_to_goal[source]:
                    continue
                cost_to_goal[source] = candidate
                next_node[source] = target
                heapq.heappush(queue, (candidate, source))

        return cls(
            features=features,
            normalized_features=normalized,
            contact_signatures=signatures,
            episode_indices=episodes,
            frame_indices=frames,
            actions44=actions,
            terminal=terminal_mask,
            feature_mean=mean,
            feature_scale=scale,
            cost_to_goal=cost_to_goal,
            next_node=next_node,
            temporal_edge_count=temporal_edge_count,
            retrieval_edge_count=retrieval_edge_count,
            config=cfg,
        )

    def plan(
        self,
        feature: np.ndarray,
        contact_signature: np.ndarray,
        *,
        max_nodes: int = 128,
    ) -> ContactManifoldPlan:
        """Attach any observed state to the manifold and graph-optimize to insertion."""
        if max_nodes <= 0:
            raise ValueError("max_nodes must be positive")
        feature = np.asarray(feature, dtype=np.float32).reshape(-1)
        signature = np.asarray(contact_signature, dtype=np.int8).reshape(-1)
        if feature.shape != (MANIFOLD_FEATURE_DIM,):
            raise ValueError(f"feature must have shape ({MANIFOLD_FEATURE_DIM},)")
        if signature.shape != (CONTACT_SIGNATURE_DIM,):
            raise ValueError(f"contact_signature must have shape ({CONTACT_SIGNATURE_DIM},)")
        normalized = (feature - self.feature_mean) / self.feature_scale
        neighbor_count = min(len(self.features), self.config.attachment_neighbors)
        distances, neighbors = self._tree.query(normalized, k=neighbor_count)
        distances = np.atleast_1d(distances)
        neighbors = np.atleast_1d(neighbors).astype(np.int64)
        compatible = [
            (float(distance), int(node))
            for distance, node in zip(distances, neighbors, strict=True)
            if np.array_equal(self.contact_signatures[node], signature)
            and np.isfinite(self.cost_to_goal[node])
        ]
        candidates = compatible or [
            (float(distance), int(node))
            for distance, node in zip(distances, neighbors, strict=True)
            if np.isfinite(self.cost_to_goal[node])
        ]
        if not candidates:
            raise RuntimeError("no manifold node has a path to insertion")
        attachment_distance, start = min(
            candidates,
            key=lambda item: (
                self.config.attachment_distance_weight * item[0]
                + self.cost_to_goal[item[1]],
                item[0],
                item[1],
            ),
        )

        path = [start]
        visited = {start}
        while len(path) < max_nodes and not self.terminal[path[-1]]:
            next_index = int(self.next_node[path[-1]])
            if next_index < 0 or next_index in visited:
                break
            path.append(next_index)
            visited.add(next_index)
        nodes = np.asarray(path, dtype=np.int64)
        return ContactManifoldPlan(
            node_indices=nodes,
            episode_indices=self.episode_indices[nodes].copy(),
            frame_indices=self.frame_indices[nodes].copy(),
            action_priors44=self.actions44[nodes].copy(),
            bridge_target=self.features[start].copy(),
            attachment_distance=attachment_distance,
            graph_cost_to_goal=float(self.cost_to_goal[start]),
            reached_terminal=bool(self.terminal[nodes[-1]]),
        )

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            features=self.features,
            normalized_features=self.normalized_features,
            contact_signatures=self.contact_signatures,
            episode_indices=self.episode_indices,
            frame_indices=self.frame_indices,
            actions44=self.actions44,
            terminal=self.terminal,
            feature_mean=self.feature_mean,
            feature_scale=self.feature_scale,
            cost_to_goal=self.cost_to_goal,
            next_node=self.next_node,
            temporal_edge_count=np.asarray(self.temporal_edge_count),
            retrieval_edge_count=np.asarray(self.retrieval_edge_count),
            config=np.asarray(
                [
                    self.config.retrieval_neighbors,
                    self.config.retrieval_search_neighbors,
                    self.config.cross_edge_max_distance,
                    self.config.cross_edge_penalty,
                    self.config.temporal_step_penalty,
                    self.config.attachment_neighbors,
                    self.config.attachment_distance_weight,
                ],
                dtype=np.float64,
            ),
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "SuccessContactManifold":
        with np.load(path) as data:
            values = data["config"]
            config = ContactManifoldConfig(
                retrieval_neighbors=int(values[0]),
                retrieval_search_neighbors=int(values[1]),
                cross_edge_max_distance=float(values[2]),
                cross_edge_penalty=float(values[3]),
                temporal_step_penalty=float(values[4]),
                attachment_neighbors=int(values[5]),
                attachment_distance_weight=float(values[6]),
            )
            return cls(
                features=data["features"],
                normalized_features=data["normalized_features"],
                contact_signatures=data["contact_signatures"],
                episode_indices=data["episode_indices"],
                frame_indices=data["frame_indices"],
                actions44=data["actions44"],
                terminal=data["terminal"],
                feature_mean=data["feature_mean"],
                feature_scale=data["feature_scale"],
                cost_to_goal=data["cost_to_goal"],
                next_node=data["next_node"],
                temporal_edge_count=int(data["temporal_edge_count"]),
                retrieval_edge_count=int(data["retrieval_edge_count"]),
                config=config,
            )
