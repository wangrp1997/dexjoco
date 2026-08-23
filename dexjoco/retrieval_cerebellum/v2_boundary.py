"""Strict deployable-input boundary for the V2 precision controller."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


_FORBIDDEN_KEY_PARTS = (
    "teacher",
    "oracle",
    "privileged",
    "ground_truth",
    "truth_state",
    "geometry_features",
    "object_pose",
    "peg_pose",
    "socket_pose",
    "tray_pose",
)


def assert_deployable_payload(payload: object, *, path: str = "payload") -> None:
    """Reject mappings that could expose simulator or teacher state to V2."""
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            normalized = str(key).strip().lower()
            if any(part in normalized for part in _FORBIDDEN_KEY_PARTS):
                raise ValueError(f"forbidden non-deployable field at {path}.{key}")
            assert_deployable_payload(value, path=f"{path}.{key}")
        return
    if isinstance(payload, Sequence) and not isinstance(
        payload,
        (str, bytes, bytearray),
    ):
        for index, value in enumerate(payload):
            assert_deployable_payload(value, path=f"{path}[{index}]")
