"""Regrasp recovery hook for ForceVLA / pi0.5 eval loops."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.spatial.transform import Rotation as R

_REPO = Path(__file__).resolve().parents[2]
_PKG = _REPO / "dexjoco"
for p in (_REPO, _PKG):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from hybrid_insert.assembly_contacts import AssemblyContactLabeler

from skill_graph.adapters.assembly import AssemblySim
from skill_graph.adapters.control import read_arm23
from skill_graph.constants import ObjectName, Side
from skill_graph.paths import template_bank_dir
from skill_graph.skills.regrasp.execute import RegraspReport
from skill_graph.skills.regrasp.session import RegraspSession
from skill_graph.skills.templates.bank import load_bank

LIFT_READY_M = 0.06
DEFAULT_LOG_INTERVAL = 30
REGRASP_PROGRESS_INTERVAL = 60


def _raw_env(gym_env):
    current = gym_env
    while hasattr(current, "env"):
        current = current.env
    return current.unwrapped if hasattr(current, "unwrapped") else current


def _object_lift_m(raw, labeler: AssemblyContactLabeler, object_name: ObjectName) -> float | None:
    if object_name == "tray":
        body_id = labeler._tray_body_id  # noqa: SLF001
        rest_z = labeler._tray_rest_z  # noqa: SLF001
    else:
        body_id = labeler._peg_body_id  # noqa: SLF001
        rest_z = labeler._peg_rest_z  # noqa: SLF001
    if rest_z is None:
        return None
    return float(raw._data.xpos[body_id, 2]) - float(rest_z)


def _world_z(raw, body_id: int) -> float:
    return float(raw._data.xpos[int(body_id), 2])


def _tray_held(raw, labeler: AssemblyContactLabeler, *, min_lift_m: float) -> bool:
    dz = _object_lift_m(raw, labeler, "tray")
    return dz is not None and dz >= float(min_lift_m)


def _peg_lifted(raw, labeler: AssemblyContactLabeler, *, min_lift_m: float) -> bool:
    dz = _object_lift_m(raw, labeler, "peg")
    return dz is not None and dz >= float(min_lift_m)


def _insert_phase_ready(raw, labeler: AssemblyContactLabeler, *, lift_ready_m: float) -> bool:
    """Both objects lifted off table (各自 dz);不要求 peg_z > tray_z。"""
    return _tray_held(raw, labeler, min_lift_m=lift_ready_m) and _peg_lifted(
        raw, labeler, min_lift_m=lift_ready_m
    )


def _sync_openpi_env(openpi_env, raw) -> None:
    sim = AssemblySim(env=openpi_env.env, raw=raw, seed=0)
    right = read_arm23(sim, "right")
    left = read_arm23(sim, "left")
    r_rot = R.from_quat(right[3:7], scalar_first=True).as_rotvec()
    l_rot = R.from_quat(left[3:7], scalar_first=True).as_rotvec()
    action = np.concatenate(
        [right[:3], r_rot, right[7:23], left[:3], l_rot, left[7:23]],
        axis=0,
    ).astype(np.float32)
    openpi_env.step(action)


def _fmt_m(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x:.3f}"


@dataclass
class RegraspRecoveryHook:
    """Monitor bimanual grasp; step-wise demo_warp regrasp when needed."""

    bank_dir: Path | None = None
    max_recoveries_per_episode: int = 5
    lift_ready_m: float = LIFT_READY_M
    verbose_log: bool = True
    log_interval: int = DEFAULT_LOG_INTERVAL
    on_physics_step: Callable[[], None] | None = None
    _templates_loaded: bool = field(default=False, init=False, repr=False)
    _templates: list = field(default_factory=list, init=False, repr=False)
    _recoveries_this_ep: int = field(default=0, init=False, repr=False)
    _last_reports: list[RegraspReport] = field(default_factory=list, init=False, repr=False)
    _labeler: AssemblyContactLabeler | None = field(default=None, init=False, repr=False)
    _peg_grasp_seen: bool = field(default=False, init=False, repr=False)
    _insert_monitor_armed: bool = field(default=False, init=False, repr=False)
    _insert_done: bool = field(default=False, init=False, repr=False)
    _session: RegraspSession | None = field(default=None, init=False, repr=False)
    _regrasp_queue: list[tuple[Side, ObjectName]] = field(default_factory=list, init=False, repr=False)
    _phase: str = field(default="vla_grasp", init=False, repr=False)
    _last_log_step: int = field(default=-1, init=False, repr=False)
    _last_regrasp_progress: int = field(default=0, init=False, repr=False)

    def reset_episode(self, gym_env=None) -> None:
        self._recoveries_this_ep = 0
        self._last_reports.clear()
        self._peg_grasp_seen = False
        self._insert_monitor_armed = False
        self._insert_done = False
        self._session = None
        self._regrasp_queue.clear()
        self._phase = "vla_grasp"
        self._last_log_step = -1
        self._last_regrasp_progress = 0
        if gym_env is not None:
            raw = _raw_env(gym_env)
            if self._labeler is None:
                self._labeler = AssemblyContactLabeler(raw)
            self._labeler.reset_reference(raw)
        if self.verbose_log:
            print("  skill_graph: [phase] vla_grasp — VLA 自由抓取/抬升，recovery 暂不介入", flush=True)

    def is_busy(self) -> bool:
        return self._session is not None and self._session.busy

    def _ensure_templates(self) -> bool:
        if self._templates_loaded:
            return bool(self._templates)
        self._templates = load_bank(self.bank_dir or template_bank_dir())
        self._templates_loaded = True
        return bool(self._templates)

    def _ensure_labeler(self, raw) -> AssemblyContactLabeler:
        if self._labeler is None:
            self._labeler = AssemblyContactLabeler(raw)
        if self._labeler._peg_rest_z is None:  # noqa: SLF001
            self._labeler.reset_reference(raw)
        return self._labeler

    def _snapshot(self, raw, labeler: AssemblyContactLabeler) -> dict[str, float | bool | None]:
        outcome = labeler.compute(raw)
        tray_dz = _object_lift_m(raw, labeler, "tray")
        peg_dz = _object_lift_m(raw, labeler, "peg")
        tray_z = _world_z(raw, labeler._tray_body_id)  # noqa: SLF001
        peg_z = _world_z(raw, labeler._peg_body_id)  # noqa: SLF001
        return {
            "tray_dz": tray_dz,
            "peg_dz": peg_dz,
            "tray_z": tray_z,
            "peg_z": peg_z,
            "tray_up": _tray_held(raw, labeler, min_lift_m=self.lift_ready_m),
            "peg_up": _peg_lifted(raw, labeler, min_lift_m=self.lift_ready_m),
            "insert_ok": bool(outcome.insert_ok),
        }

    def _infer_phase(self, snap: dict[str, float | bool | None]) -> str:
        if self.is_busy() and self._session is not None:
            return f"regrasp_{self._session.object_name}"
        if snap["insert_ok"]:
            return "insert_success"
        if self._insert_monitor_armed:
            return "insert_monitor"
        if self._peg_grasp_seen:
            if snap["peg_up"] and not snap["tray_up"]:
                return "peg_only_need_tray"
            if snap["peg_up"] and snap["tray_up"]:
                return "bimanual_lifted"
            return "peg_seen"
        return "vla_grasp"

    def _phase_note(self, phase: str) -> str:
        notes = {
            "vla_grasp": "VLA 抓抬阶段，peg 未稳定抬起",
            "peg_seen": "已检测到 peg 抬起过，继续观察 tray",
            "peg_only_need_tray": "peg 已抬、tray 在桌面 — 应重抓 tray",
            "bimanual_lifted": "双手均已抬离桌面，等待进入插孔监控",
            "insert_monitor": "插孔监控中 — tray/peg 高度掉落则 recovery",
            "insert_success": "检测到 insert 接触",
        }
        if phase.startswith("regrasp_"):
            obj = phase.split("_", 1)[1]
            return f"demo_warp 重抓 {obj}（逐步执行）"
        return notes.get(phase, phase)

    def _set_phase(self, phase: str, *, timestamp: int, snap: dict) -> None:
        if phase == self._phase:
            return
        prev = self._phase
        self._phase = phase
        if not self.verbose_log:
            return
        print(
            "  skill_graph: "
            f"[t={timestamp:04d}] phase {prev} → {phase} | {self._phase_note(phase)} | "
            f"tray_dz={_fmt_m(snap['tray_dz'])} peg_dz={_fmt_m(snap['peg_dz'])} "
            f"tray_z={snap['tray_z']:.3f} peg_z={snap['peg_z']:.3f}",
            flush=True,
        )

    def log_status(self, gym_env, *, timestamp: int, force: bool = False) -> None:
        """Periodic chain-of-thought snapshot while VLA runs."""
        if not self.verbose_log or self.is_busy():
            return
        if not force and self._last_log_step >= 0 and (timestamp - self._last_log_step) < self.log_interval:
            return
        raw = _raw_env(gym_env)
        labeler = self._ensure_labeler(raw)
        self._update_phases(raw)
        snap = self._snapshot(raw, labeler)
        phase = self._infer_phase(snap)
        self._set_phase(phase, timestamp=timestamp, snap=snap)
        self._last_log_step = timestamp
        if force or (timestamp % self.log_interval == 0):
            print(
                "  skill_graph: "
                f"[t={timestamp:04d}] status phase={phase} | "
                f"tray_up={snap['tray_up']} peg_up={snap['peg_up']} | "
                f"tray_dz={_fmt_m(snap['tray_dz'])} peg_dz={_fmt_m(snap['peg_dz'])} | "
                f"monitor={'on' if self._insert_monitor_armed else 'off'} "
                f"recoveries={self._recoveries_this_ep}/{self.max_recoveries_per_episode}",
                flush=True,
            )

    def _update_phases(self, raw) -> None:
        labeler = self._ensure_labeler(raw)
        snap = self._snapshot(raw, labeler)
        if snap["insert_ok"]:
            self._insert_done = True
        if snap["peg_up"]:
            self._peg_grasp_seen = True
        if self._insert_monitor_armed or self._insert_done:
            return
        if _insert_phase_ready(raw, labeler, lift_ready_m=self.lift_ready_m):
            self._insert_monitor_armed = True

    def _needed_regrasps(self, raw, labeler: AssemblyContactLabeler) -> list[tuple[Side, ObjectName, str]]:
        peg_up = _peg_lifted(raw, labeler, min_lift_m=self.lift_ready_m)
        tray_up = _tray_held(raw, labeler, min_lift_m=self.lift_ready_m)
        needs: list[tuple[Side, ObjectName, str]] = []

        if self._peg_grasp_seen and peg_up and not tray_up:
            needs.append(("left", "tray", "peg_up但tray仍在桌面"))

        if self._insert_monitor_armed and not self._insert_done:
            if not tray_up and not any(x[1] == "tray" for x in needs):
                needs.append(("left", "tray", "插孔阶段tray高度丢失"))
            if not _peg_lifted(raw, labeler, min_lift_m=self.lift_ready_m) and not any(
                x[1] == "peg" for x in needs
            ):
                needs.append(("right", "peg", "插孔阶段peg落回桌面"))
        return needs

    def maybe_start_recovery(self, gym_env, *, prefer_episode: int | None = None, timestamp: int = 0) -> bool:
        if self.is_busy() or self._recoveries_this_ep >= self.max_recoveries_per_episode:
            return False
        if not self._ensure_templates():
            return False

        raw = _raw_env(gym_env)
        self._update_phases(raw)
        labeler = self._ensure_labeler(raw)
        snap = self._snapshot(raw, labeler)
        reason = ""

        if self._regrasp_queue:
            side, object_name, reason = self._regrasp_queue.pop(0)
        else:
            needs = self._needed_regrasps(raw, labeler)
            if not needs:
                return False
            side, object_name, reason = needs[0]
            self._regrasp_queue = needs[1:]

        sim = AssemblySim(env=gym_env, raw=raw, seed=0)
        self._session = RegraspSession.begin(
            sim,
            self._templates,
            side=side,
            object_name=object_name,
            prefer_episode=prefer_episode,
            compact_approach=object_name == "peg",
        )
        if not self._session.busy:
            self._finish_session(timestamp=timestamp)
            return False

        self._last_regrasp_progress = 0
        phase = f"regrasp_{object_name}"
        self._set_phase(phase, timestamp=timestamp, snap=snap)
        if self.verbose_log:
            print(
                "  skill_graph: "
                f"[t={timestamp:04d}] → 启动 {object_name} 重抓 | 原因: {reason} | "
                f"首模板 {self._session.planned_steps} 步 | "
                f"tray_dz={_fmt_m(snap['tray_dz'])} peg_dz={_fmt_m(snap['peg_dz'])}",
                flush=True,
            )
        return True

    def step_recovery(self, openpi_env, *, prefer_episode: int | None = None, timestamp: int = 0) -> bool:
        if not self.is_busy():
            return False
        raw = _raw_env(openpi_env.env)
        sim = AssemblySim(env=openpi_env.env, raw=raw, seed=0)
        finished = self._session.step_once(sim)  # type: ignore[union-attr]
        _sync_openpi_env(openpi_env, raw)

        if self.verbose_log and self._session is not None:
            done = self._session._steps_done  # noqa: SLF001
            if done - self._last_regrasp_progress >= REGRASP_PROGRESS_INTERVAL or finished:
                print(
                    "  skill_graph: "
                    f"[t={timestamp:04d}] regrasp {self._session.object_name} "
                    f"{self._session.progress_text}",
                    flush=True,
                )
                self._last_regrasp_progress = done

        if not finished:
            return False
        self._finish_session(timestamp=timestamp)
        if self._regrasp_queue and self._recoveries_this_ep < self.max_recoveries_per_episode:
            self.maybe_start_recovery(openpi_env.env, prefer_episode=prefer_episode, timestamp=timestamp)
        return True

    def _finish_session(self, *, timestamp: int = 0) -> None:
        if self._session is None or self._session.report is None:
            self._session = None
            return
        report = self._session.report
        self._last_reports.append(report)
        self._recoveries_this_ep += 1
        if self.verbose_log:
            print(
                "  skill_graph: "
                f"[t={timestamp:04d}] 重抓结束 {report.template_id} "
                f"{'成功' if report.success else '失败'} | "
                f"contacts={report.contact_count} steps={report.steps} | "
                f"→ 交还 VLA",
                flush=True,
            )
        self._session = None

    def check_and_recover(self, gym_env, *, prefer_episode: int | None = None, timestamp: int = 0) -> bool:
        if self.is_busy():
            return False
        return self.maybe_start_recovery(gym_env, prefer_episode=prefer_episode, timestamp=timestamp)

    def episode_summary(self) -> str:
        if not self._last_reports:
            return f"regrasp=0 final_phase={self._phase}"
        ok = sum(1 for r in self._last_reports if r.success)
        last = self._last_reports[-1]
        return (
            f"regrasp={ok}/{len(self._last_reports)} last={last.template_id} "
            f"contacts={last.contact_count} final_phase={self._phase}"
        )
