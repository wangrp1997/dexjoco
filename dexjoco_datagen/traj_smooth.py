"""Cartesian trajectory helpers: min-jerk + OSQP smoothing (no cuRobo)."""

from __future__ import annotations

import numpy as np

try:
    import osqp
    import scipy.sparse as sp

    _HAS_OSQP = True
except Exception:  # noqa: BLE001
    _HAS_OSQP = False


def smoothstep(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def min_jerk(t: float) -> float:
    """Classic 5th-order min-jerk scalar schedule."""
    t = float(np.clip(t, 0.0, 1.0))
    return 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5


def densify_line(
    a: np.ndarray,
    b: np.ndarray,
    *,
    step_m: float = 0.004,
    n_min: int = 16,
    n_max: int = 120,
) -> np.ndarray:
    """Dense linear waypoints from a→b (inclusive of b)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    dist = float(np.linalg.norm(b - a))
    n = int(np.ceil(dist / max(step_m, 1e-4)))
    n = int(np.clip(n, n_min, n_max))
    ts = np.linspace(0.0, 1.0, n + 1, dtype=np.float64)[1:]
    return np.stack([(1.0 - t) * a + t * b for t in ts], axis=0)


def osqp_smooth_path(
    waypoints: np.ndarray,
    *,
    w_ref: float = 8.0,
    w_acc: float = 40.0,
    w_jerk: float = 8.0,
) -> np.ndarray:
    """Smooth an (N, D) path with OSQP; fall back to min-jerk retime if OSQP missing.

    Cost: w_ref||x-ref||^2 + w_acc||Δ²x||^2 + w_jerk||Δ³x||^2
    Hard: x[0]=ref[0], x[-1]=ref[-1]
    """
    ref = np.asarray(waypoints, dtype=np.float64)
    if ref.ndim != 2 or ref.shape[0] < 3:
        return ref.copy()
    if not _HAS_OSQP:
        return _min_jerk_retime(ref)

    n, d = ref.shape
    # Solve each dimension independently (shared sparsity).
    out = np.zeros_like(ref)
    for dim in range(d):
        out[:, dim] = _osqp_smooth_1d(
            ref[:, dim],
            w_ref=w_ref,
            w_acc=w_acc,
            w_jerk=w_jerk,
        )
    return out


def _min_jerk_retime(ref: np.ndarray) -> np.ndarray:
    n = ref.shape[0]
    a, b = ref[0], ref[-1]
    out = np.zeros_like(ref)
    for i in range(n):
        t = min_jerk((i + 1) / float(n))
        out[i] = (1.0 - t) * a + t * b
    return out


def _osqp_smooth_1d(
    ref: np.ndarray,
    *,
    w_ref: float,
    w_acc: float,
    w_jerk: float,
) -> np.ndarray:
    n = int(ref.shape[0])
    # Variables: x_0..x_{n-1}
    # P = w_ref I + w_acc A2'A2 + w_jerk A3'A3
    # q = -w_ref ref
    # Aeq: x0=ref0, x_{n-1}=ref_{n-1}

    # Acc rows: x_{i+1}-2x_i+x_{i-1} for i=1..n-2 → (n-2, n)
    # Jerk: x_{i+2}-3x_{i+1}+3x_i-x_{i-1} for i=1..n-3 → (n-3, n)
    rows_p: list[int] = []
    cols_p: list[int] = []
    data_p: list[float] = []

    def _add_gram(row_coeffs: list[tuple[int, float]], weight: float) -> None:
        # Add weight * c c^T into P (upper triangle only for OSQP).
        for i, ci in row_coeffs:
            for j, cj in row_coeffs:
                if j < i:
                    continue
                rows_p.append(i)
                cols_p.append(j)
                data_p.append(weight * ci * cj)

    for i in range(n):
        rows_p.append(i)
        cols_p.append(i)
        data_p.append(float(w_ref))

    for i in range(1, n - 1):
        _add_gram([(i - 1, 1.0), (i, -2.0), (i + 1, 1.0)], w_acc)

    for i in range(1, n - 2):
        _add_gram(
            [(i - 1, -1.0), (i, 3.0), (i + 1, -3.0), (i + 2, 1.0)],
            w_jerk,
        )

    P = sp.coo_matrix((data_p, (rows_p, cols_p)), shape=(n, n)).tocsc()
    P = (P + P.T) * 0.5  # symmetrize numerical noise
    q = (-w_ref * ref).astype(np.float64)

    # Equality constraints on endpoints.
    A = sp.csc_matrix(
        ([1.0, 1.0], ([0, 1], [0, n - 1])),
        shape=(2, n),
    )
    l = np.array([ref[0], ref[-1]], dtype=np.float64)
    u = l.copy()

    solver = osqp.OSQP()
    solver.setup(P=P, q=q, A=A, l=l, u=u, verbose=False, polish=True)
    res = solver.solve()
    if res.x is None or not np.all(np.isfinite(res.x)):
        return _min_jerk_retime(ref.reshape(-1, 1)).reshape(-1)
    return np.asarray(res.x, dtype=np.float64)


def cbf_push_away(
    movable_xy: np.ndarray,
    obstacle_xy: np.ndarray,
    *,
    d_safe: float,
    gain: float = 1.0,
    max_push: float = 0.12,
) -> np.ndarray:
    """Soft CBF-style XY push of movable away from obstacle when closer than d_safe."""
    m = np.asarray(movable_xy, dtype=np.float64).reshape(2)
    o = np.asarray(obstacle_xy, dtype=np.float64).reshape(2)
    delta = m - o
    dist = float(np.linalg.norm(delta))
    if dist >= d_safe:
        return m
    if dist < 1e-6:
        n = np.array([0.0, 1.0], dtype=np.float64)
    else:
        n = delta / dist
    push = float(np.clip(gain * (d_safe - dist), 0.0, max_push))
    return m + n * push
