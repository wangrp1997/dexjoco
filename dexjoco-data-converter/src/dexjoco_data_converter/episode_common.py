import numpy as np


def find_first_non_static_frame(action: np.ndarray) -> int:
    """Find the first frame where an episode starts changing.

    Args:
        action: Per-step action array ordered by time.

    Returns:
        Index of the first action whose value differs from the following action.

    Raises:
        ValueError: If all adjacent actions are identical.
    """
    for i in range(len(action) - 1):
        if not np.array_equal(action[i], action[i + 1]):
            return i
    raise ValueError("All actions in episode are identical (entirely static)")


def hand_action_mean(action: np.ndarray) -> float:
    """Mean abs hand command for DexJoCo action44 / action_rotvec layout."""
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    if a.shape[0] < 44:
        raise ValueError(f"expected action dim >= 44, got {a.shape[0]}")
    right = float(np.abs(a[6:22]).mean())
    left = float(np.abs(a[28:44]).mean())
    return max(right, left)


def should_skip_init_hold_hand_frame(
    action: np.ndarray,
    *,
    hold_hand_max: float = 0.05,
    next_hand_min: float = 0.3,
) -> bool:
    """True if action[0] is an open-hand hold and action[1] is a grasp command.

    Absolute wrist rotvecs are often nonzero even at rest, so checking the full
    44-D vector for exact zeros misses this init residue. Hand dims alone are the
    reliable signal for DexJoCo bimanual_assembly.
    """
    action = np.asarray(action)
    if action.ndim != 2 or action.shape[0] < 2:
        return False
    h0 = hand_action_mean(action[0])
    h1 = hand_action_mean(action[1])
    return h0 <= hold_hand_max and h1 >= next_hand_min
