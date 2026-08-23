"""Stateful rule baseline for VLA-to-cerebellum ownership transfer."""

from __future__ import annotations

from dataclasses import dataclass

from .control import CerebellumMode, CerebellumObservation, HandoffDecision


@dataclass(frozen=True)
class RuleHandoffConfig:
    """Thresholds for the first interpretable handoff baseline."""

    transport_lateral_m: float = 0.060
    transport_height_m: float = 0.100
    align_lateral_m: float = 0.008
    align_axis_rad: float = 0.14
    stable_slip_speed_mps: float = 0.040
    regrasp_slip_speed_mps: float = 0.080
    stable_rotation_slip_radps: float = 0.60
    regrasp_rotation_slip_radps: float = 1.50
    grasp_lost_confirm_frames: int = 8
    regrasp_slip_confirm_frames: int = 3

    def __post_init__(self) -> None:
        positive_fields = (
            "transport_lateral_m",
            "transport_height_m",
            "align_lateral_m",
            "align_axis_rad",
            "stable_slip_speed_mps",
            "regrasp_slip_speed_mps",
            "stable_rotation_slip_radps",
            "regrasp_rotation_slip_radps",
        )
        for name in positive_fields:
            value = float(getattr(self, name))
            if value <= 0.0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.regrasp_slip_speed_mps <= self.stable_slip_speed_mps:
            raise ValueError(
                "regrasp_slip_speed_mps must exceed stable_slip_speed_mps"
            )
        if self.regrasp_rotation_slip_radps <= self.stable_rotation_slip_radps:
            raise ValueError(
                "regrasp_rotation_slip_radps must exceed stable_rotation_slip_radps"
            )
        if self.grasp_lost_confirm_frames <= 0:
            raise ValueError("grasp_lost_confirm_frames must be positive")
        if self.regrasp_slip_confirm_frames <= 0:
            raise ValueError("regrasp_slip_confirm_frames must be positive")


class RuleBasedHandoffPolicy:
    """Interpretable P0 controller ownership policy with grasp-loss memory."""

    def __init__(self, config: RuleHandoffConfig | None = None) -> None:
        self.config = config or RuleHandoffConfig()
        self.reset()

    def reset(self) -> None:
        self._peg_grasp_seen = False
        self._tray_grasp_seen = False
        self._peg_lost_streak = 0
        self._tray_lost_streak = 0
        self._severe_slip_streak = 0

    def decide(self, observation: CerebellumObservation) -> HandoffDecision:
        if observation.insert_contact:
            return HandoffDecision(CerebellumMode.COMPLETE, "insert contact confirmed", 1.0)

        if observation.peg_grasped:
            self._peg_grasp_seen = True
            self._peg_lost_streak = 0
        elif self._peg_grasp_seen:
            self._peg_lost_streak += 1

        if observation.tray_grasped:
            self._tray_grasp_seen = True
            self._tray_lost_streak = 0
        elif self._tray_grasp_seen:
            self._tray_lost_streak += 1

        peg_lost = self._peg_lost_streak >= self.config.grasp_lost_confirm_frames
        tray_lost = self._tray_lost_streak >= self.config.grasp_lost_confirm_frames
        if peg_lost or tray_lost:
            lost = "peg" if peg_lost else "tray"
            return HandoffDecision(
                CerebellumMode.VLA_REGRASP,
                f"return control to VLA because {lost} was lost for multiple frames",
                0.98,
            )

        slip_speed = observation.slip_speed_mps
        rotation_slip = observation.rotation_slip_radps
        severe_slip = bool(
            slip_speed is not None
            and slip_speed >= self.config.regrasp_slip_speed_mps
            or rotation_slip is not None
            and rotation_slip >= self.config.regrasp_rotation_slip_radps
        )
        self._severe_slip_streak = self._severe_slip_streak + 1 if severe_slip else 0
        if self._severe_slip_streak >= self.config.regrasp_slip_confirm_frames:
            return HandoffDecision(
                CerebellumMode.VLA_REGRASP,
                "return control to VLA because slip requires regrasp",
                0.95,
            )

        both_grasped = observation.peg_grasped and observation.tray_grasped
        if not both_grasped:
            if (
                observation.peg_grasp_stable
                and not observation.tray_grasped
                and not self._tray_grasp_seen
            ):
                return HandoffDecision(
                    CerebellumMode.GRASP_ASSIST,
                    "hold stable right-hand peg grasp while assisting left-hand tray grasp",
                    0.94,
                )
            if (
                observation.tray_grasp_stable
                and not observation.peg_grasped
                and not self._peg_grasp_seen
            ):
                return HandoffDecision(
                    CerebellumMode.GRASP_ASSIST,
                    "hold stable left-hand tray grasp while assisting right-hand peg grasp",
                    0.94,
                )
            return HandoffDecision(
                CerebellumMode.VLA_GRASP,
                "VLA retains control until both objects are grasped and lifted",
                0.90,
            )

        grasp_unstable = not (
            observation.peg_grasp_stable and observation.tray_grasp_stable
        )
        slip_unstable = (
            slip_speed is not None and slip_speed > self.config.stable_slip_speed_mps
        )
        rotation_unstable = (
            rotation_slip is not None
            and rotation_slip > self.config.stable_rotation_slip_radps
        )
        if grasp_unstable or slip_unstable or rotation_unstable:
            return HandoffDecision(
                CerebellumMode.GRASP_STABILIZE,
                "bimanual grasp requires finger-level stabilization",
                0.92,
            )

        primitives = observation.primitives
        if (
            primitives.lateral_error_m > self.config.transport_lateral_m
            or primitives.approach_height_m > self.config.transport_height_m
        ):
            return HandoffDecision(
                CerebellumMode.TRANSPORT,
                "stable objects remain outside the pre-insert region",
                0.88,
            )

        if (
            primitives.lateral_error_m > self.config.align_lateral_m
            or primitives.axis_error_rad > self.config.align_axis_rad
        ):
            return HandoffDecision(
                CerebellumMode.ALIGN,
                "pre-insert geometry requires axis or lateral correction",
                0.94,
            )

        return HandoffDecision(
            CerebellumMode.INSERT,
            "stable grasp and pre-insert alignment confirmed",
            0.96,
        )
