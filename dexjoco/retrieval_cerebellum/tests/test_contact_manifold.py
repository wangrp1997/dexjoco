import numpy as np

from retrieval_cerebellum.contact_manifold import (
    CONTACT_SIGNATURE_DIM,
    MANIFOLD_FEATURE_DIM,
    ContactManifoldConfig,
    SuccessContactManifold,
    contact_signature_matrix,
    manifold_feature_matrix,
)


def _graph() -> SuccessContactManifold:
    progress = np.array([4.0, 3.0, 2.0, 0.0], dtype=np.float32)
    episode_one = np.zeros((4, MANIFOLD_FEATURE_DIM), dtype=np.float32)
    episode_two = np.zeros((4, MANIFOLD_FEATURE_DIM), dtype=np.float32)
    episode_one[:, 0] = progress
    episode_two[:, 0] = progress + 0.05
    features = np.concatenate([episode_one, episode_two])
    signatures = np.ones((8, CONTACT_SIGNATURE_DIM), dtype=np.int8)
    signatures[:, 2] = 0
    signatures[[3, 7], 2] = 1
    return SuccessContactManifold.fit(
        features=features,
        contact_signatures=signatures,
        episode_indices=np.repeat([1, 2], 4),
        frame_indices=np.tile(np.arange(4), 2),
        actions44=np.zeros((8, 44), dtype=np.float32),
        terminal=np.array([False, False, False, True] * 2),
        config=ContactManifoldConfig(
            retrieval_neighbors=2,
            retrieval_search_neighbors=8,
            cross_edge_max_distance=2.0,
        ),
    )


def test_manifold_plan_reaches_success_instead_of_rejecting_far_query():
    graph = _graph()
    query = np.zeros(MANIFOLD_FEATURE_DIM, dtype=np.float32)
    query[0] = 5.0
    signature = np.ones(CONTACT_SIGNATURE_DIM, dtype=np.int8)
    signature[2] = 0

    plan = graph.plan(query, signature)

    assert plan.reached_terminal
    assert plan.attachment_distance > 0
    assert graph.terminal[plan.node_indices[-1]]
    assert len(plan.node_indices) >= 2


def test_manifold_adds_cross_episode_retrieval_edges():
    graph = _graph()

    assert graph.temporal_edge_count == 6
    assert graph.retrieval_edge_count > 0
    assert np.isfinite(graph.cost_to_goal).all()


def test_manifold_round_trip(tmp_path):
    graph = _graph()
    path = graph.save(tmp_path / "graph.npz")

    restored = SuccessContactManifold.load(path)

    np.testing.assert_allclose(restored.features, graph.features)
    np.testing.assert_array_equal(restored.next_node, graph.next_node)
    assert restored.retrieval_edge_count == graph.retrieval_edge_count


def test_feature_and_contact_encoders():
    length = 3
    columns = {
        name: np.zeros((length, 3), dtype=np.float32)
        for name in (
            "peg_tip_in_hole_position",
            "peg_in_hole_rotvec",
            "peg_in_right_palm_position",
            "peg_in_right_palm_rotvec",
            "tray_in_left_palm_position",
            "tray_in_left_palm_rotvec",
        )
    }
    for name in (
        "lateral_error_m",
        "axis_error_rad",
        "approach_height_m",
        "insertion_depth_m",
        "peg_contact_count",
        "tray_contact_count",
    ):
        columns[name] = np.zeros(length, dtype=np.float32)
    columns.update(
        peg_ok=np.ones(length, dtype=bool),
        tray_ok=np.ones(length, dtype=bool),
        insert_ok=np.array([False, False, True]),
    )

    features = manifold_feature_matrix(columns)
    signatures = contact_signature_matrix(columns)

    assert features.shape == (length, MANIFOLD_FEATURE_DIM)
    assert signatures.shape == (length, CONTACT_SIGNATURE_DIM)
    assert signatures[-1, 2] == 1
