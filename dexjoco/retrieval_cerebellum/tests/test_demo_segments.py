import numpy as np

from retrieval_cerebellum.demo_segments import (
    DemoSegmentationConfig,
    EpisodeLabels,
    segment_post_grasp_episode,
)


def _labels(peg_ok, tray_ok, insert_ok) -> EpisodeLabels:
    length = len(peg_ok)
    return EpisodeLabels(
        episode_index=7,
        frame_index=np.arange(length),
        global_index=np.arange(100, 100 + length),
        peg_ok=np.asarray(peg_ok),
        tray_ok=np.asarray(tray_ok),
        insert_ok=np.asarray(insert_ok),
    )


def test_segment_starts_after_confirmed_bimanual_grasp_and_ends_at_insert():
    peg = [0, 0] + [1] * 12
    tray = [0, 0] + [1] * 12
    insert = [0] * 11 + [1, 1, 1]

    segment = segment_post_grasp_episode(
        _labels(peg, tray, insert),
        DemoSegmentationConfig(grasp_confirm_frames=3, min_segment_frames=5),
    )

    assert segment.eligible
    assert segment.start_frame == 2
    assert segment.end_frame == 11
    assert segment.insert_frame == 11
    assert segment.start_index == 102
    assert segment.end_index == 111
    assert segment.num_frames == 10


def test_segment_ignores_short_contact_flicker():
    peg = [0, 1, 1, 0, 0] + [1] * 10
    tray = [0, 1, 1, 0, 0] + [1] * 10
    insert = [0] * 13 + [1, 1]

    segment = segment_post_grasp_episode(
        _labels(peg, tray, insert),
        DemoSegmentationConfig(grasp_confirm_frames=3, min_segment_frames=5),
    )

    assert segment.eligible
    assert segment.start_frame == 5


def test_segment_keeps_insert_demo_with_intermediate_grasp_label_gap():
    peg = [0, 0] + [1] * 7 + [0] * 5 + [1] * 4
    tray = [0, 0] + [1] * 16
    insert = [0] * 16 + [1, 1]

    segment = segment_post_grasp_episode(
        _labels(peg, tray, insert),
        DemoSegmentationConfig(
            grasp_confirm_frames=3,
            grasp_loss_confirm_frames=3,
            min_segment_frames=4,
        ),
    )

    assert segment.eligible
    assert not segment.grasp_retained_to_insert
    assert segment.rejection_reason is None
    assert segment.insert_frame == 16


def test_segment_rejects_episode_without_bimanual_grasp():
    segment = segment_post_grasp_episode(
        _labels([1] * 10, [0] * 10, [0] * 9 + [1]),
        DemoSegmentationConfig(grasp_confirm_frames=3),
    )

    assert not segment.eligible
    assert segment.rejection_reason == "both_grasp_not_confirmed"


def test_segment_rejects_missing_insert_contact():
    segment = segment_post_grasp_episode(
        _labels([1] * 20, [1] * 20, [0] * 20),
        DemoSegmentationConfig(grasp_confirm_frames=3, min_segment_frames=4),
    )

    assert not segment.eligible
    assert segment.grasp_retained_to_insert
    assert segment.rejection_reason == "insert_not_observed"
