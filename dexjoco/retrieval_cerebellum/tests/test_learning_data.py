import numpy as np
from retrieval_cerebellum.learning_data import (
    GEOMETRY_FEATURE_DIM,
    RETRIEVAL_DESCRIPTOR_DIM,
    RetrievalEntry,
    RetrievalIndex,
    episode_split,
    geometry_feature_matrix,
    retrieval_descriptor,
    state46_to_action44,
)


def _state46() -> np.ndarray:
    state = np.zeros(46, dtype=np.float32)
    state[:3] = [0.1, 0.2, 0.3]
    state[3:7] = [1.0, 0.0, 0.0, 0.0]
    state[7:10] = [-0.1, -0.2, -0.3]
    state[10:14] = [1.0, 0.0, 0.0, 0.0]
    state[14:30] = np.arange(16) / 10
    state[30:46] = np.arange(16) / 20
    return state


def _geometry(length: int = 2) -> dict[str, np.ndarray]:
    vector_names = (
        "peg_tip_in_hole_position",
        "peg_in_hole_rotvec",
        "peg_in_right_palm_position",
        "peg_in_right_palm_rotvec",
        "tray_in_left_palm_position",
        "tray_in_left_palm_rotvec",
    )
    scalar_names = (
        "lateral_error_m",
        "axis_error_rad",
        "approach_height_m",
        "insertion_depth_m",
        "target_depth_m",
        "nominal_peg_size_m",
        "peg_ok",
        "tray_ok",
        "insert_ok",
        "peg_contact_count",
        "tray_contact_count",
    )
    columns = {
        name: np.full((length, 3), index + 1, dtype=np.float32)
        for index, name in enumerate(vector_names)
    }
    columns.update(
        {
            name: np.full(length, index + 0.5, dtype=np.float32)
            for index, name in enumerate(scalar_names)
        }
    )
    return columns


def test_state_conversion_matches_policy_layout():
    converted = state46_to_action44(_state46())

    np.testing.assert_allclose(converted[:3], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(converted[3:6], np.zeros(3))
    np.testing.assert_allclose(converted[6:22], np.arange(16) / 10)
    np.testing.assert_allclose(converted[22:25], [-0.1, -0.2, -0.3])
    np.testing.assert_allclose(converted[28:44], np.arange(16) / 20)


def test_geometry_features_and_retrieval_descriptor_are_object_centric():
    geometry = _geometry()

    features = geometry_feature_matrix(geometry)
    descriptor = retrieval_descriptor(geometry)

    assert features.shape == (2, GEOMETRY_FEATURE_DIM)
    assert descriptor.shape == (RETRIEVAL_DESCRIPTOR_DIM,)
    np.testing.assert_allclose(descriptor[:3], geometry["peg_in_right_palm_position"][0])


def test_retrieval_index_uses_train_gallery_family_and_exclusion():
    zero = np.zeros(RETRIEVAL_DESCRIPTOR_DIM, dtype=np.float32)
    entries = [
        RetrievalEntry(1, "round", "train", zero),
        RetrievalEntry(2, "round", "train", zero + 0.1),
        RetrievalEntry(3, "round", "validation", zero + 0.01),
        RetrievalEntry(4, "square", "train", zero),
    ]
    index = RetrievalIndex(entries)

    matches = index.query(zero, family_id="round", exclude_episode=1, top_k=4)

    assert [entry.episode_index for entry, _ in matches] == [2]


def test_retrieval_normalization_uses_training_entries_only():
    zero = np.zeros(RETRIEVAL_DESCRIPTOR_DIM, dtype=np.float32)
    index = RetrievalIndex(
        [
            RetrievalEntry(1, "round", "train", zero),
            RetrievalEntry(2, "round", "train", zero + 2.0),
            RetrievalEntry(3, "round", "test", zero + 1000.0),
        ]
    )

    np.testing.assert_allclose(index.mean, np.ones(RETRIEVAL_DESCRIPTOR_DIM))


def test_episode_split_is_deterministic():
    assert episode_split(12, "round", seed=7) == episode_split(12, "round", seed=7)
