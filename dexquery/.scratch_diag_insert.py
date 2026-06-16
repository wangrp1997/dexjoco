#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
import zarr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "dexjoco"))

from dexquery.data.zarr_io import (
    discover_zarr_demos,
    load_zarr_episode,
    _trim_static_prefix_like_lerobot,
)
from dexquery.data.episode_replay import replay_episode_actions, make_assembly_env
from dexquery.data.assembly_contacts import AssemblyContactLabeler
from dexquery.data.action_utils import rotvec_dual_arm_to_policy, policy_dual_arm_to_raw
from dexjoco.tasks import CONFIG_MAPPING
from dexjoco.tasks.state_restorers import restore_initial_state

zarr_root = Path("/mnt/ssd/datasets/dexjoco_raw/dexjoco_raw_datasets/bimanual_assembly")
demos = discover_zarr_demos(zarr_root)
config = CONFIG_MAPPING["bimanual_assembly"]()

for idx in [0, 1, 2, 10, 50]:
    zp = demos[idx]
    actions, init = load_zarr_episode(zp)
    outcomes, _ = replay_episode_actions(actions, seed=idx, initial_state=init)
    ins = sum(o.insert_ok for o in outcomes)
    tray = sum(o.tray_ok for o in outcomes)
    peg = sum(o.peg_ok for o in outcomes)
    print(
        f"ep{idx} steps={len(outcomes)} tray={tray} peg={peg} insert={ins} "
        f"name={zp.parent.name}"
    )

zp = demos[0]
actions, init = load_zarr_episode(zp)
env = make_assembly_env(seed=0)
raw = env.unwrapped
labeler = AssemblyContactLabeler(raw)
env.reset()
restore_initial_state(env, "bimanual_assembly", config, init)
min_dist = 1e9
for t, a44 in enumerate(actions):
    raw.step(policy_dual_arm_to_raw(rotvec_dual_arm_to_policy(a44)))
    peg = raw._data.body("industreal_round_peg_8mm").xpos
    site = raw._data.site("industreal_tray_insert_round_peg_8mm_socket_site").xpos
    d = float(np.linalg.norm(peg - site))
    min_dist = min(min_dist, d)
    if t >= len(actions) - 5:
        o = labeler.compute(raw)
        print(f"rotvec t={t} dist={d:.4f} insert_ok={o.insert_ok} ncon={raw._data.ncon}")
print(f"rotvec min peg-site dist {min_dist:.4f}")
env.close()

root = zarr.open(str(zp), "r")
act46 = np.asarray(root["data"]["action"][:], dtype=np.float64)
states = np.asarray(root["data"]["state"][:], dtype=np.float64)
act46, init2 = _trim_static_prefix_like_lerobot(act46, states)
env2 = make_assembly_env(seed=0)
raw2 = env2.unwrapped
lab2 = AssemblyContactLabeler(raw2)
env2.reset()
restore_initial_state(env2, "bimanual_assembly", config, init2)
ins2 = 0
min_d2 = 1e9
for t, a in enumerate(act46):
    raw2.step({"right": a[:23], "left": a[23:46]})
    peg = raw2._data.body("industreal_round_peg_8mm").xpos
    site = raw2._data.site("industreal_tray_insert_round_peg_8mm_socket_site").xpos
    min_d2 = min(min_d2, float(np.linalg.norm(peg - site)))
    if lab2.compute(raw2).insert_ok:
        ins2 += 1
print(f"native46 insert frames={ins2}/{len(act46)} min_dist={min_d2:.4f}")
env2.close()

m = make_assembly_env(seed=0).unwrapped._model
peg_gid = m.geom("industreal_round_peg_8mm_collision").id
bottom_gid = m.geom("industreal_tray_insert_round_peg_8mm_bottom_contact").id
print(f"env peg_geom={peg_gid} bottom={bottom_gid}")
print(f"labeler insert_geom={AssemblyContactLabeler(make_assembly_env(seed=0).unwrapped)._insert_geom_id}")
