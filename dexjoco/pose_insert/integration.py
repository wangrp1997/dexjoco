"""Plug PoseInsert rollout into DexJoCo eval loops."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from hybrid_insert.integration import get_raw_env

from .ckpt_utils import detect_action_dim
from .config import PoseInsertAdapterConfig
from .controller import PoseInsertController
from .controller_bimanual import DirectAction44Controller, DirectDualWristController, ZarrInsertOracleController
from .inference import PoseInsertPolicyRunner
from .inference_bimanual import BimanualAction44Runner, BimanualPoseInsertRunner

SUPPORTED_TASKS = frozenset({"bimanual_assembly"})


class EvalDirectAction44Insert:
    """Obs pose9, action dual-arm44, direct execution (wrist + hands)."""

    def __init__(
        self,
        *,
        task: str,
        enabled: bool,
        ckpt_path: Path | str,
        data_root: Path | str | None = None,
        config: PoseInsertAdapterConfig | None = None,
        num_action: int = 20,
        normalize: bool = True,
    ) -> None:
        self.task = task
        self.enabled = bool(enabled) and task in SUPPORTED_TASKS
        self.controller: DirectAction44Controller | None = None
        if self.enabled:
            runner = BimanualAction44Runner(
                ckpt_path,
                data_root=data_root,
                num_action=num_action,
                normalize=normalize,
            )
            self.controller = DirectAction44Controller(runner, config=config)

    def on_reset(self, gym_env: Any, *, peg_rest_z: float | None = None, tray_rest_z: float | None = None) -> None:
        if self.controller is None:
            return
        self.controller.reset(get_raw_env(gym_env), peg_rest_z=peg_rest_z, tray_rest_z=tray_rest_z)

    def observe(self, gym_env: Any, policy_action44: np.ndarray) -> None:
        if self.controller is None or self.controller.active:
            return
        self.controller.update_handoff(get_raw_env(gym_env), policy_action44)

    @property
    def active(self) -> bool:
        return bool(self.controller is not None and self.controller.active)

    @property
    def needs_policy_left(self) -> bool:
        return False

    def merge(self, gym_env: Any, policy_action44: np.ndarray) -> np.ndarray:
        action = np.asarray(policy_action44, dtype=np.float64).reshape(-1)
        if self.controller is None:
            return action.astype(np.float32)
        return self.controller.merge_dual_arm(get_raw_env(gym_env), action)

    def begin_pose_insert(self, gym_env: Any, policy_action44: np.ndarray) -> bool:
        if self.controller is None:
            return False
        return self.controller.begin_pose_insert(
            get_raw_env(gym_env), np.asarray(policy_action44, dtype=np.float64).reshape(-1)
        )

    def episode_summary(self) -> str:
        if self.controller is None:
            return "disabled"
        return self.controller.episode_summary()


class EvalDirectWrist12Insert:
    """Obs pose9, action dual wrist12, direct mocap execution."""

    def __init__(
        self,
        *,
        task: str,
        enabled: bool,
        ckpt_path: Path | str,
        data_root: Path | str | None = None,
        config: PoseInsertAdapterConfig | None = None,
        num_action: int = 20,
        normalize: bool = True,
    ) -> None:
        self.task = task
        self.enabled = bool(enabled) and task in SUPPORTED_TASKS
        self.controller: DirectDualWristController | None = None
        if self.enabled:
            runner = BimanualPoseInsertRunner(
                ckpt_path,
                data_root=data_root,
                num_action=num_action,
                normalize=normalize,
            )
            self.controller = DirectDualWristController(runner, config=config)

    def on_reset(self, gym_env: Any, *, peg_rest_z: float | None = None, tray_rest_z: float | None = None) -> None:
        if self.controller is None:
            return
        self.controller.reset(get_raw_env(gym_env), peg_rest_z=peg_rest_z, tray_rest_z=tray_rest_z)

    def observe(self, gym_env: Any, policy_action44: np.ndarray) -> None:
        if self.controller is None or self.controller.active:
            return
        self.controller.update_handoff(get_raw_env(gym_env), policy_action44)

    @property
    def active(self) -> bool:
        return bool(self.controller is not None and self.controller.active)

    @property
    def needs_policy_left(self) -> bool:
        return False

    def merge(self, gym_env: Any, policy_action44: np.ndarray) -> np.ndarray:
        action = np.asarray(policy_action44, dtype=np.float64).reshape(-1)
        if self.controller is None:
            return action.astype(np.float32)
        return self.controller.merge_dual_arm(get_raw_env(gym_env), action)

    def begin_pose_insert(self, gym_env: Any, policy_action44: np.ndarray) -> bool:
        if self.controller is None:
            return False
        return self.controller.begin_pose_insert(
            get_raw_env(gym_env), np.asarray(policy_action44, dtype=np.float64).reshape(-1)
        )

    def episode_summary(self) -> str:
        if self.controller is None:
            return "disabled"
        return self.controller.episode_summary()


class EvalZarrOracleInsert:
    """Demo insert segment open-loop from zarr (oracle baseline)."""

    def __init__(
        self,
        *,
        task: str,
        zarr_path: Path | str,
        insert_start_frame: int,
        config: PoseInsertAdapterConfig | None = None,
        insert_end_frame: int | None = None,
    ) -> None:
        self.task = task
        self.enabled = task in SUPPORTED_TASKS
        self.controller = ZarrInsertOracleController(
            zarr_path,
            int(insert_start_frame),
            config=config or PoseInsertAdapterConfig(),
            end_frame=insert_end_frame,
        )

    def on_reset(self, gym_env: Any, *, peg_rest_z: float | None = None, tray_rest_z: float | None = None) -> None:
        self.controller.reset(get_raw_env(gym_env), peg_rest_z=peg_rest_z, tray_rest_z=tray_rest_z)

    def observe(self, gym_env: Any, policy_action44: np.ndarray) -> None:
        return

    @property
    def active(self) -> bool:
        return self.controller.active

    @property
    def needs_policy_left(self) -> bool:
        return False

    def merge(self, gym_env: Any, policy_action44: np.ndarray) -> np.ndarray:
        return self.controller.merge_dual_arm(get_raw_env(gym_env), policy_action44)

    def begin_pose_insert(self, gym_env: Any, policy_action44: np.ndarray) -> bool:
        return self.controller.begin_pose_insert(
            get_raw_env(gym_env), np.asarray(policy_action44, dtype=np.float64).reshape(-1)
        )

    def episode_summary(self) -> str:
        return self.controller.episode_summary()


class EvalBimanualPoseInsert(EvalDirectWrist12Insert):
    """Alias for direct wrist12 insert."""


class EvalPoseInsert:
    """Legacy: 9D pose -> right wrist only, left frozen."""

    def __init__(
        self,
        *,
        task: str,
        enabled: bool,
        ckpt_path: Path | str,
        data_root: Path | str | None = None,
        config: PoseInsertAdapterConfig | None = None,
        num_action: int = 20,
        normalize: bool = True,
    ) -> None:
        self.task = task
        self.enabled = bool(enabled) and task in SUPPORTED_TASKS
        self.controller: PoseInsertController | None = None
        if self.enabled:
            runner = PoseInsertPolicyRunner(
                ckpt_path,
                data_root=data_root,
                num_action=num_action,
                normalize=normalize,
            )
            self.controller = PoseInsertController(runner, config=config)

    def on_reset(
        self,
        gym_env: Any,
        *,
        peg_rest_z: float | None = None,
        tray_rest_z: float | None = None,
    ) -> None:
        if not self.enabled or self.controller is None:
            return
        self.controller.reset(
            get_raw_env(gym_env),
            peg_rest_z=peg_rest_z,
            tray_rest_z=tray_rest_z,
        )

    def observe(self, gym_env: Any, policy_action44: np.ndarray) -> None:
        if not self.enabled or self.controller is None or self.controller.active:
            return
        self.controller.update_handoff(get_raw_env(gym_env), policy_action44)

    @property
    def active(self) -> bool:
        return bool(self.enabled and self.controller is not None and self.controller.active)

    @property
    def needs_policy_left(self) -> bool:
        return bool(
            self.enabled and self.controller is not None and self.controller.needs_policy_left
        )

    def merge(self, gym_env: Any, policy_action44: np.ndarray) -> np.ndarray:
        action = np.asarray(policy_action44, dtype=np.float64).reshape(-1)
        if not self.enabled or self.controller is None:
            return action.astype(np.float32)
        return self.controller.merge_right_arm(get_raw_env(gym_env), action)

    def begin_pose_insert(self, gym_env: Any, policy_action44: np.ndarray) -> bool:
        if not self.enabled or self.controller is None:
            return False
        return self.controller.begin_pose_insert(
            get_raw_env(gym_env),
            np.asarray(policy_action44, dtype=np.float64).reshape(-1),
        )

    def episode_summary(self) -> str:
        if not self.enabled or self.controller is None:
            return "disabled"
        return self.controller.episode_summary()


def make_eval_insert_runner(
    *,
    task: str,
    ckpt_path: Path | str | None = None,
    data_root: Path | str | None = None,
    config: PoseInsertAdapterConfig | None = None,
    insert_mode: str = "auto",
    zarr_path: Path | str | None = None,
    insert_start_frame: int | None = None,
    insert_end_frame: int | None = None,
) -> EvalDirectAction44Insert | EvalDirectWrist12Insert | EvalPoseInsert | EvalZarrOracleInsert:
    """Factory: auto picks wrist12 vs pose9 from ckpt; or zarr oracle."""
    if insert_mode == "zarr_oracle":
        if zarr_path is None or insert_start_frame is None:
            raise ValueError("zarr_oracle requires zarr_path and insert_start_frame")
        return EvalZarrOracleInsert(
            task=task,
            zarr_path=zarr_path,
            insert_start_frame=int(insert_start_frame),
            config=config,
            insert_end_frame=insert_end_frame,
        )
    if ckpt_path is None:
        raise ValueError("policy insert requires ckpt_path")
    dim = detect_action_dim(ckpt_path) if insert_mode in ("auto", "policy") else 9
    if insert_mode == "action44" or (insert_mode == "auto" and dim == 44):
        return EvalDirectAction44Insert(
            task=task,
            enabled=True,
            ckpt_path=ckpt_path,
            data_root=data_root,
            config=config,
        )
    if insert_mode == "wrist12" or (insert_mode == "auto" and dim == 12):
        return EvalDirectWrist12Insert(
            task=task,
            enabled=True,
            ckpt_path=ckpt_path,
            data_root=data_root,
            config=config,
        )
    return EvalPoseInsert(
        task=task,
        enabled=True,
        ckpt_path=ckpt_path,
        data_root=data_root,
        config=config or PoseInsertAdapterConfig(freeze_left_arm_at_handoff=True),
    )
