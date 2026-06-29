# DexGraspBench (ported subset)

Source: `refs/DexGraspBench`

| Mirror | Original | Notes |
|--------|----------|-------|
| `src/task/eval_func/fc_metric/qp.py` | 同名 | verbatim, import 改 dexjoco |
| `src/util/rot_util.py` | 同名 | verbatim |
| `src/task/eval_func/tabletop_mocap.py` | 同名 | 参考；adapter: `grasp/staged_grasp.py` |
| `src/task/eval_func/fc_mocap.py` | 同名 | 参考 |
| `src/util/hand_util.py` | 同名 | 参考 |

Re-export: `interaction_retarget/tpsr/contact_qp.py`, `tpsr/rot_util.py`
