"""Pre-insert state: bimanual grasp + lift done, before PoseInsert runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hybrid_insert.assembly_contacts import AssemblyContactLabeler

from interaction_retarget.constants import PEG_BODY


def is_pre_insert_ready(
    raw_env,
    labeler: AssemblyContactLabeler,
    *,
    lift_ready_m: float,
    peg_rest_z: float | None = None,
) -> bool:
    """Tray + peg grasped and lifted; insert has not started."""
    outcome = labeler.compute(raw_env)
    if peg_rest_z is None:
        peg_rest_z = float(labeler._peg_rest_z)  # noqa: SLF001
    peg_id = int(raw_env._model.body(PEG_BODY).id)
    peg_dz = float(raw_env._data.xpos[peg_id, 2]) - float(peg_rest_z)
    return bool(outcome.tray_ok and outcome.peg_ok and peg_dz >= float(lift_ready_m))


def resolve_peg_lift_end_frame(entry: dict[str, Any], sidecar_dir: Path | str) -> int:
    """Zarr frame index at end of peg lift (start of insert phase)."""
    from interaction_retarget.grasp.lift_reference import extract_demo_lift_reference, load_demo_lift_reference
    from interaction_retarget.skill_replay.library import SkillLibrary

    ep = int(entry["episode_index"])
    sidecar_dir = Path(sidecar_dir)
    cache = SkillLibrary(sidecar_dir)._lift_cache_path(ep)
    if cache.is_file():
        return int(load_demo_lift_reference(cache)["peg_lift_end_frame"])

    ref = extract_demo_lift_reference(sidecar_dir, episode_index=ep, pick_seed=0)
    return int(ref.frames.peg_lift_end_frame)
