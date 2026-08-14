"""External FullEpisodeEnv snapshot adapter (does not modify reach_insert_rl)."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

# Recorded for audit / optional vector restore; primary restore uses MjData deepcopy
# (same approach as InsertEnvSnapshot) because constraint buffers (efc_*) change size.
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
STATE_SPEC_NAME = "mjSTATE_INTEGRATION"

_EXPLICIT_DATA_FIELDS = (
    "qpos",
    "qvel",
    "act",
    "ctrl",
    "mocap_pos",
    "mocap_quat",
    "userdata",
    "time",
)

# Fixed-size fields safe to re-apply after deepcopy (defense in depth).
_FIXED_REAPPLY = ("qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat")


def _require_array(name: str, value: Any) -> np.ndarray:
    if value is None:
        raise RuntimeError(f"snapshot missing required MuJoCo field: {name}")
    return np.array(np.asarray(value), copy=True)


@dataclass(frozen=True)
class FullEpisodeSnapshot:
    """Immutable capture of MuJoCo dynamics + FullEpisodeEnv Python state."""

    state_spec_name: str
    state_spec: int
    state: np.ndarray
    model_nq: int
    model_nv: int
    model_nu: int
    model_na: int
    model_nmocap: int
    # Full MjData deep copy — required for exact matched intervention.
    sim_data: Any
    # Explicit dynamical fields (copies; never views).
    qpos: np.ndarray
    qvel: np.ndarray
    act: np.ndarray | None
    ctrl: np.ndarray
    mocap_pos: np.ndarray
    mocap_quat: np.ndarray
    userdata: np.ndarray | None
    time: float
    # Raw / gym wrapper bookkeeping.
    raw_env_step: int
    wrapper_elapsed_steps: int | None
    # FullEpisodeEnv Python state.
    t: int
    peg_lost: int
    hold44: np.ndarray
    force_baseline: np.ndarray | None
    prev_tip: float
    prev_lat: float
    prev_along: float
    tray_ok_seen: bool
    peg_ok_seen: bool
    done: bool
    episode_index: int
    zarr_path: str
    peg_lift_end: int
    # Labeler resting heights (both env labeler and force_labeler's nested labeler).
    labeler_peg_rest_z: float | None
    labeler_tray_rest_z: float | None
    force_labeler_peg_rest_z: float | None
    force_labeler_tray_rest_z: float | None

    @staticmethod
    def _model_schema(model: mujoco.MjModel) -> dict[str, int]:
        return {
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": int(model.nu),
            "na": int(model.na),
            "nmocap": int(model.nmocap),
        }

    @classmethod
    def capture(cls, env) -> "FullEpisodeSnapshot":
        if env._hold44 is None:
            raise RuntimeError("snapshot requires reset() first (_hold44 is None)")
        if env._spec is None:
            raise RuntimeError("snapshot requires reset() first (_spec is None)")
        if bool(env._done):
            raise RuntimeError("refusing to capture a done episode; reset or restore first")

        raw = env._raw
        model = raw._model
        data = raw._data
        schema = cls._model_schema(model)

        state = np.empty(mujoco.mj_stateSize(model, STATE_SPEC), dtype=np.float64)
        mujoco.mj_getState(model, data, state, STATE_SPEC)

        explicit: dict[str, Any] = {}
        for name in _EXPLICIT_DATA_FIELDS:
            if not hasattr(data, name):
                if name in {"act", "userdata"}:
                    explicit[name] = None
                    continue
                raise RuntimeError(f"MuJoCo data missing required field: {name}")
            value = getattr(data, name)
            if name == "time":
                explicit[name] = float(value)
            elif name in {"act", "userdata"}:
                arr = np.asarray(value)
                explicit[name] = None if arr.size == 0 else np.array(arr, copy=True)
            else:
                explicit[name] = _require_array(name, value)

        elapsed = None
        if hasattr(env._env, "_elapsed_steps") and env._env._elapsed_steps is not None:
            elapsed = int(env._env._elapsed_steps)

        labeler = env._labeler
        force_nested = None
        if env._force_labeler is not None:
            force_nested = getattr(env._force_labeler, "_contact_labeler", None)

        return cls(
            state_spec_name=STATE_SPEC_NAME,
            state_spec=int(STATE_SPEC),
            state=np.array(state, copy=True),
            model_nq=schema["nq"],
            model_nv=schema["nv"],
            model_nu=schema["nu"],
            model_na=schema["na"],
            model_nmocap=schema["nmocap"],
            sim_data=deepcopy(data),
            qpos=explicit["qpos"],
            qvel=explicit["qvel"],
            act=explicit["act"],
            ctrl=explicit["ctrl"],
            mocap_pos=explicit["mocap_pos"],
            mocap_quat=explicit["mocap_quat"],
            userdata=explicit["userdata"],
            time=float(explicit["time"]),
            raw_env_step=int(getattr(raw, "env_step", 0)),
            wrapper_elapsed_steps=elapsed,
            t=int(env._t),
            peg_lost=int(env._peg_lost),
            hold44=np.asarray(env._hold44, dtype=np.float64).copy(),
            force_baseline=(
                None
                if env._force_baseline is None
                else np.asarray(env._force_baseline, dtype=np.float64).copy()
            ),
            prev_tip=float(env._prev_tip),
            prev_lat=float(env._prev_lat),
            prev_along=float(env._prev_along),
            tray_ok_seen=bool(env._tray_ok_seen),
            peg_ok_seen=bool(env._peg_ok_seen),
            done=bool(env._done),
            episode_index=int(env._spec.episode_index),
            zarr_path=str(env._spec.zarr_path),
            peg_lift_end=int(env._spec.peg_lift_end),
            labeler_peg_rest_z=_optional_float(getattr(labeler, "_peg_rest_z", None)),
            labeler_tray_rest_z=_optional_float(getattr(labeler, "_tray_rest_z", None)),
            force_labeler_peg_rest_z=_optional_float(
                None if force_nested is None else getattr(force_nested, "_peg_rest_z", None)
            ),
            force_labeler_tray_rest_z=_optional_float(
                None if force_nested is None else getattr(force_nested, "_tray_rest_z", None)
            ),
        )

    def _assert_schema(self, model: mujoco.MjModel) -> None:
        got = self._model_schema(model)
        expected = {
            "nq": self.model_nq,
            "nv": self.model_nv,
            "nu": self.model_nu,
            "na": self.model_na,
            "nmocap": self.model_nmocap,
        }
        if got != expected:
            raise RuntimeError(f"snapshot model schema mismatch: expected {expected}, got {got}")

    def restore(self, env) -> np.ndarray:
        """Restore into an already-constructed FullEpisodeEnv without reset()."""
        if env._spec is None or int(env._spec.episode_index) != int(self.episode_index):
            raise RuntimeError(
                "restore requires the same episode already loaded via reset(); "
                f"env episode={None if env._spec is None else env._spec.episode_index}, "
                f"snapshot episode={self.episode_index}"
            )
        if str(env._spec.zarr_path) != str(self.zarr_path):
            raise RuntimeError(
                f"zarr_path mismatch: env={env._spec.zarr_path} snapshot={self.zarr_path}"
            )

        raw = env._raw
        model = raw._model
        self._assert_schema(model)

        if int(self.state_spec) != int(STATE_SPEC):
            raise RuntimeError(
                f"unsupported state_spec {self.state_spec_name}={self.state_spec}; "
                f"adapter expects {STATE_SPEC_NAME}={int(STATE_SPEC)}"
            )

        if self.sim_data is None:
            raise RuntimeError("snapshot missing sim_data deepcopy; cannot restore exactly")

        # Primary path: replace MjData (includes variable-size constraint buffers).
        raw._data = deepcopy(self.sim_data)
        data = raw._data

        # Defense in depth: re-apply fixed-size integration fields from explicit copies.
        for name in _FIXED_REAPPLY:
            src = getattr(self, name)
            dst = getattr(data, name)
            if dst.shape != src.shape:
                raise RuntimeError(f"{name} shape mismatch after deepcopy: {dst.shape} vs {src.shape}")
            np.copyto(dst, src)
        data.time = float(self.time)
        if self.act is not None and getattr(data, "act", None) is not None and data.act.size:
            if data.act.shape != self.act.shape:
                raise RuntimeError(f"act shape mismatch: {data.act.shape} vs {self.act.shape}")
            np.copyto(data.act, self.act)
        if (
            self.userdata is not None
            and getattr(data, "userdata", None) is not None
            and data.userdata.size
        ):
            if data.userdata.shape != self.userdata.shape:
                raise RuntimeError(
                    f"userdata shape mismatch: {data.userdata.shape} vs {self.userdata.shape}"
                )
            np.copyto(data.userdata, self.userdata)

        if hasattr(raw, "env_step"):
            raw.env_step = int(self.raw_env_step)
        if self.wrapper_elapsed_steps is not None and hasattr(env._env, "_elapsed_steps"):
            env._env._elapsed_steps = int(self.wrapper_elapsed_steps)

        env._t = int(self.t)
        env._peg_lost = int(self.peg_lost)
        env._hold44 = np.asarray(self.hold44, dtype=np.float64).copy()
        env._force_baseline = (
            None
            if self.force_baseline is None
            else np.asarray(self.force_baseline, dtype=np.float64).copy()
        )
        env._prev_tip = float(self.prev_tip)
        env._prev_lat = float(self.prev_lat)
        env._prev_along = float(self.prev_along)
        env._tray_ok_seen = bool(self.tray_ok_seen)
        env._peg_ok_seen = bool(self.peg_ok_seen)
        env._done = bool(self.done)
        # Gate smokes may continue past original capture; never leave a sticky done=True
        # if the snapshot itself was taken mid-episode.
        if not bool(self.done):
            env._done = False

        _set_rest(env._labeler, self.labeler_peg_rest_z, self.labeler_tray_rest_z)
        if env._force_labeler is not None:
            nested = getattr(env._force_labeler, "_contact_labeler", None)
            if nested is not None:
                _set_rest(nested, self.force_labeler_peg_rest_z, self.force_labeler_tray_rest_z)

        return env._obs()


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _set_rest(labeler: Any, peg_rest_z: float | None, tray_rest_z: float | None) -> None:
    if peg_rest_z is None or tray_rest_z is None:
        raise RuntimeError("snapshot missing labeler resting heights; cannot restore safely")
    labeler._peg_rest_z = float(peg_rest_z)
    labeler._tray_rest_z = float(tray_rest_z)
