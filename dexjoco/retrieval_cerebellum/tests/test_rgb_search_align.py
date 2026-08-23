import cv2
import numpy as np

from retrieval_cerebellum.rgb_search_align import (
    RGBSearchAlignEstimator,
    damped_visual_command,
)


def test_rgb_estimator_detects_peg_tip_hole_and_axis() -> None:
    image = np.full((240, 320, 3), 220, dtype=np.uint8)
    cv2.rectangle(image, (80, 120), (180, 220), (20, 70, 130), -1)
    cv2.line(image, (250, 60), (210, 190), (230, 190, 40), 18)

    feature = RGBSearchAlignEstimator().estimate(image)

    assert feature is not None
    assert feature.confidence > 0.2
    assert feature.peg_tip_uv[1] > 170
    assert 110 < feature.hole_uv[0] < 150
    assert abs(feature.peg_axis_angle_rad) > 0.1


def test_damped_visual_command_reduces_linearized_error() -> None:
    jacobian = np.zeros((3, 6))
    jacobian[:3, :3] = np.eye(3) * 10.0
    error = np.asarray([20.0, -10.0, 0.2])
    command = damped_visual_command(jacobian, error, damping=1.0)
    after = error + jacobian @ command
    assert np.linalg.norm(after) < np.linalg.norm(error)
