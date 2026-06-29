#!/usr/bin/env python3
"""Sweep ContactOpt caps_rad for Allegro+industreal (demo 0 peg refine)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "dexjoco"))

from interaction_retarget.constants import default_sidecar_dir
from interaction_retarget.contact_refine.config import CAPS_RAD_SWEEP, ContactCapsConfig
from interaction_retarget.contact_refine import refine_demo_contact
from interaction_retarget.grasp.agent_tpsr import make_peg_agent_tpsr
from interaction_retarget.grasp.pipeline import _ik_grasp_ready
from interaction_retarget.sim.contact import AssemblyContactDetector
from interaction_retarget.sim.replay import make_assembly_env
from interaction_retarget.sim.settle import read_arm_action, vec_to_arm_action
from interaction_retarget.skill_replay.contact_align import skill_side_config
from interaction_retarget.skill_replay.demo_grasp import episode_canonical_from_sidecar
from interaction_retarget.skill_replay.deploy import _manifest_entry, _restore_demo_layout
from interaction_retarget.skill_replay.library import SkillLibrary


def main() -> None:
    sidecar = default_sidecar_dir()
    lib = SkillLibrary(sidecar, exclude_fallback=True)
    entry = _manifest_entry(lib, 0)
    canonical = episode_canonical_from_sidecar(sidecar, entry, object_name="peg", seed_base=0)
    peg_cfg = skill_side_config("peg", fast=False)
    agent = make_peg_agent_tpsr(canonical, fast=False)
    agent.side_cfg = peg_cfg

    env = make_assembly_env(seed=0, randomize=False)
    raw = env.unwrapped
    detector = AssemblyContactDetector(raw)
    try:
        env.reset()
        detector.reset_reference(raw)
        _restore_demo_layout(env, entry, detector)
        right_home = vec_to_arm_action(read_arm_action(raw, "right"))
        left_hold = vec_to_arm_action(read_arm_action(raw, "left"))

        right_target, peg_ik = agent.plan(
            raw,
            hold_right=right_home,
            hold_left=left_hold,
            detector=detector,
            side_cfg=peg_cfg,
        )
        right_hold, left_hold, _, _ = agent.execute(
            raw,
            target23=right_target,
            hold_right=right_home,
            hold_left=left_hold,
            detector=detector,
            ik=peg_ik,
            skip_approach=_ik_grasp_ready(peg_ik),
            side_cfg=peg_cfg,
        )
        pre_c = int(detector.compute(raw).peg_contact_count)

        print(f"demo=0 peg pre-refine contact={pre_c}")
        print("caps_rad  contact  site_rmse  co_loss  gt_loss  score")
        best_rad = CAPS_RAD_SWEEP[0]
        best_c = -1
        for rad in CAPS_RAD_SWEEP:
            caps = ContactCapsConfig(caps_rad=float(rad), caps_top=0.004, caps_bot=-0.012)
            rh, lh, rep = refine_demo_contact(
                raw,
                side="right",
                object_name="peg",
                canonical=canonical,
                hold_right=right_hold.copy(),
                hold_left=left_hold.copy(),
                detector=detector,
                max_iters=16,
                caps=caps,
            )
            _ = rh, lh
            print(
                f"{rad:7.3f}  {rep.contact_count:7d}  {rep.contact_site_rmse_m*1e3:7.1f}mm  "
                f"{rep.contactopt_loss:6.3f}  {rep.grasptta_loss:7.2f}  {rep.total_score:7.1f}"
            )
            if rep.contact_count > best_c:
                best_c = rep.contact_count
                best_rad = rad
        print(f"best caps_rad={best_rad} contact={best_c}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
