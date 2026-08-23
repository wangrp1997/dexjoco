import numpy as np

from retrieval_cerebellum.handoff_selection import select_stable_handoff_row


def test_selects_latest_preentry_row_with_stable_attachment_history():
    features = np.zeros((6, 29), dtype=np.float64)
    features[:, 20] = [0.06, 0.05, 0.04, 0.03, 0.01, -0.001]
    features[:, 24:26] = 1.0
    features[:, 6] = [0.0, 0.001, 0.002, 0.003, 0.013, 0.014]

    selection = select_stable_handoff_row(
        features,
        maximum_depth_advance_m=0.02,
        right_attachment_translation_step_m=0.002,
        left_attachment_translation_step_m=0.002,
        right_attachment_rotation_step_rad=0.02,
        left_attachment_rotation_step_rad=0.02,
    )

    assert selection.row == 3
    assert selection.entry_row == 5


def test_rejects_handoff_after_either_grasp_becomes_invalid():
    features = np.zeros((7, 29), dtype=np.float64)
    features[:, 20] = [0.07, 0.06, 0.05, 0.04, 0.03, 0.01, -0.001]
    features[:, 24:26] = 1.0
    features[4:, 24] = 0.0

    selection = select_stable_handoff_row(
        features,
        maximum_depth_advance_m=0.02,
        right_attachment_translation_step_m=0.002,
        left_attachment_translation_step_m=0.002,
        right_attachment_rotation_step_rad=0.02,
        left_attachment_rotation_step_rad=0.02,
    )

    assert selection.row == 3
