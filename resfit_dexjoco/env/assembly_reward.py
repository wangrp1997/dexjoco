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
        reward = 0.0

        if outcome.tray_ok and not self._awarded_tray:
            reward += self.config.tray_reward
            self._awarded_tray = True
        if outcome.peg_ok and not self._awarded_peg:
            reward += self.config.peg_reward
            self._awarded_peg = True
        if outcome.insert_ok and not self._awarded_insert:
            reward += self.config.insert_reward
            self._awarded_insert = True
        if terminated and succeed and not self._awarded_success:
            reward += self.config.success_reward
            self._awarded_success = True

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
