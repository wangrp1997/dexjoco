#!/usr/bin/env python3
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "dexjoco"))

from dexquery.data.action_utils import policy_dual_arm_to_raw, rotvec_dual_arm_to_policy
from dexquery.data.assembly_contacts import AssemblyContactLabeler
from dexquery.data.episode_replay import make_assembly_env
from dexquery.data.zarr_io import discover_zarr_demos, load_zarr_episode
from dexjoco.tasks import CONFIG_MAPPING
from dexjoco.tasks.state_restorers import restore_initial_state


def stat(name, arr):
    if not arr:
        return f"{name}: none"
    a = np.asarray(arr, dtype=np.float64)
    return (
        f"{name}: min={a.min():.4f} p50={np.percentile(a, 50):.4f} "
        f"p90={np.percentile(a, 90):.4f} max={a.max():.4f}"
    )


demos = discover_zarr_demos(
    Path("/mnt/ssd/datasets/dexjoco_raw/dexjoco_raw_datasets/bimanual_assembly")
)
config = CONFIG_MAPPING["bimanual_assembly"]()

for ep_idx in [0, 3, 10]:
    actions, init = load_zarr_episode(demos[ep_idx])
    env = make_assembly_env(seed=ep_idx, randomize=False)
    raw = env.unwrapped
    lab = AssemblyContactLabeler(raw)
    env.reset()
    restore_initial_state(env, "bimanual_assembly", config, init)

    peg_rest = float(raw._data.body("industreal_round_peg_8mm").xpos[2])
    tray_rest = float(raw._data.body("industreal_tray_insert_round_peg_8mm").xpos[2])
    table_z = float(raw._model.body("table").pos[2])

    peg_dz_contact, tray_dz_contact = [], []
    peg_dz_all, tray_dz_all = [], []

    for a44 in actions:
        raw.step(policy_dual_arm_to_raw(rotvec_dual_arm_to_policy(a44)))
        o = lab.compute(raw)
        peg_dz = float(raw._data.body("industreal_round_peg_8mm").xpos[2]) - peg_rest
        tray_dz = float(raw._data.body("industreal_tray_insert_round_peg_8mm").xpos[2]) - tray_rest
        peg_dz_all.append(peg_dz)
        tray_dz_all.append(tray_dz)
        if o.peg_ok:
            peg_dz_contact.append(peg_dz)
        if o.tray_ok:
            tray_dz_contact.append(tray_dz)

    print(f"\n=== ep {ep_idx} table_z={table_z:.4f} peg_rest={peg_rest:.4f} tray_rest={tray_rest:.4f} ===")
    print(stat("peg_dz_all", peg_dz_all))
    print(stat("tray_dz_all", tray_dz_all))
    print(stat("peg_dz|contact", peg_dz_contact))
    print(stat("tray_dz|contact", tray_dz_contact))
    env.close()
