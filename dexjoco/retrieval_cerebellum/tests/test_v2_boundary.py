import pytest

from retrieval_cerebellum.v2_boundary import assert_deployable_payload


def test_deployable_boundary_accepts_real_sensor_fields():
    assert_deployable_payload(
        {
            "images": {"wrist_right": [[0]]},
            "state46": [0.0] * 46,
            "wrist_wrench_world": [[0.0] * 6, [0.0] * 6],
        }
    )


@pytest.mark.parametrize(
    "field",
    [
        "teacher_state5",
        "oracle_belief",
        "privileged_geometry",
        "geometry_features",
        "peg_pose",
        "socket_pose",
    ],
)
def test_deployable_boundary_rejects_privileged_fields(field):
    with pytest.raises(ValueError, match="forbidden non-deployable field"):
        assert_deployable_payload({"sensor": {field: [0.0]}})
