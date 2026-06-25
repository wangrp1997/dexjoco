"""ResiP-style milestone rewards for bimanual assembly via sim contacts."""

from __future__ import annotations

from dataclasses import dataclass, field

from dexjoco_openpi_client.dexjoco_openpi_env import _get_raw_env
from hybrid_insert.assembly_contacts import AssemblyContactLabeler


@dataclass
class MilestoneRewardConfig:
    """Per-stage bonuses, each awarded at most once per episode."""

    tray_reward: float = 0.33
    peg_reward: float = 0.33
    insert_reward: float = 0.34
    success_reward: float = 1.0


@dataclass
class MilestoneRewardInfo:
    step_reward: float = 0.0
    tray_ok: bool = False
    peg_ok: bool = False
    insert_ok: bool = False
    succeed: bool = False
    milestones_reached: dict[str, bool] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "step_reward": self.step_reward,
            "tray_ok": self.tray_ok,
            "peg_ok": self.peg_ok,
            "insert_ok": self.insert_ok,
            "succeed": self.succeed,
            "milestones_reached": dict(self.milestones_reached),
        }


@dataclass
class MilestoneAwardState:
    tray: bool = False
    peg: bool = False
    insert: bool = False
    success: bool = False


def milestone_reward_from_flags(
    *,
    tray_ok: bool,
    peg_ok: bool,
    insert_ok: bool,
    awarded: MilestoneAwardState,
    config: MilestoneRewardConfig | None = None,
    terminated: bool = False,
    succeed: bool = False,
) -> tuple[float, MilestoneAwardState]:
    """Award milestone bonuses at most once per episode (shared online/offline)."""
    cfg = config or MilestoneRewardConfig()
    reward = 0.0

    if tray_ok and not awarded.tray:
        reward += cfg.tray_reward
        awarded.tray = True
    if peg_ok and not awarded.peg:
        reward += cfg.peg_reward
        awarded.peg = True
    if insert_ok and not awarded.insert:
        reward += cfg.insert_reward
        awarded.insert = True
    if terminated and succeed and not awarded.success:
        reward += cfg.success_reward
        awarded.success = True

    return reward, awarded


class AssemblyMilestoneReward:
    """Dense milestone reward using ``AssemblyContactLabeler``."""

    def __init__(
        self,
        labeler: AssemblyContactLabeler,
        config: MilestoneRewardConfig | None = None,
    ) -> None:
        self.labeler = labeler
        self.config = config or MilestoneRewardConfig()
        self._awarded_tray = False
        self._awarded_peg = False
        self._awarded_insert = False
        self._awarded_success = False

    def reset(self, wrapped_env) -> None:
        raw_env = _get_raw_env(wrapped_env)
        self.labeler.reset_reference(raw_env)
        self._awarded_tray = False
        self._awarded_peg = False
        self._awarded_insert = False
        self._awarded_success = False

    def compute(
        self,
        wrapped_env,
        *,
        terminated: bool,
        succeed: bool,
    ) -> tuple[float, MilestoneRewardInfo]:
        outcome = self.labeler.compute(_get_raw_env(wrapped_env))
        reward, awarded = milestone_reward_from_flags(
            tray_ok=outcome.tray_ok,
            peg_ok=outcome.peg_ok,
            insert_ok=outcome.insert_ok,
            awarded=MilestoneAwardState(
                tray=self._awarded_tray,
                peg=self._awarded_peg,
                insert=self._awarded_insert,
                success=self._awarded_success,
            ),
            config=self.config,
            terminated=terminated,
            succeed=succeed,
        )
        self._awarded_tray = awarded.tray
        self._awarded_peg = awarded.peg
        self._awarded_insert = awarded.insert
        self._awarded_success = awarded.success

        info = MilestoneRewardInfo(
            step_reward=reward,
            tray_ok=outcome.tray_ok,
            peg_ok=outcome.peg_ok,
            insert_ok=outcome.insert_ok,
            succeed=succeed,
            milestones_reached={
                "tray": self._awarded_tray,
                "peg": self._awarded_peg,
                "insert": self._awarded_insert,
                "success": self._awarded_success,
            },
        )
        return reward, info

    @classmethod
    def for_bimanual_assembly(
        cls,
        wrapped_env,
        config: MilestoneRewardConfig | None = None,
    ) -> "AssemblyMilestoneReward":
        raw_env = _get_raw_env(wrapped_env)
        return cls(AssemblyContactLabeler(raw_env), config=config)
