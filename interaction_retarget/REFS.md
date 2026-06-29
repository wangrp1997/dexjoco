# interaction_retarget ↔ refs 对照

各步骤在 `interaction_retarget/` 下建立与 `refs/` **同名子目录**；能 verbatim 的 cp，需 Allegro/MuJoCo 适配的改 adapter。

## 目录 mirror

| mirror 路径 | refs 源 | 状态 |
|---|---|---|
| `holosoma/holosoma_retargeting/src/laplacian_utils.py` | holosoma `utils.py` L394–461 | numpy 子集 |
| `DexGraspBench/src/task/eval_func/fc_metric/qp.py` | 同名 | verbatim + import 修 |
| `DexGraspBench/src/util/rot_util.py` | 同名 | verbatim |
| `DexGraspBench/src/task/eval_func/tabletop_mocap.py` | 同名 | 参考（adapter: `grasp/staged_grasp.py`） |
| `DexGraspBench/src/task/eval_func/fc_mocap.py` | 同名 | 参考（adapter: `bench/verify.py`） |
| `DexGraspBench/src/util/hand_util.py` | 同名 | 参考 |
| `Dexonomy/dexonomy/sim/squeeze_qpos.py` | `mujoco_env.py` L681–721 | copy+Allegro 适配 |
| `Dexonomy/dexonomy/config/op/grasp.yaml` | 同名 | verbatim |
| `Dexonomy/dexonomy/op/gen_grasp.py` | 同名 | 参考（adapter: `tpsr/grasp_filter.py`） |
| `contactopt/contactopt/*.py` | contactopt | numpy 移植 |
| `GraspTTA/utils/*.py` | GraspTTA | numpy 移植（Allegro 指尖） |
| `GenHand/optimisation/{loss,icp}.py` | GenHand | 参考（adapter: `grasp/contact_targets.py`, `grasp/qpos_refine.py`） |
| `GenHand/simulation/*.py` | GenHand | 参考（adapter: `grasp/approach.py`） |
| `spider/spider/preprocess/detect_contact.py` | spider | 参考（adapter: `sim/contact.py`） |

## Phase-1 qpos 路径（主 — DITTO warp + 单 demo）

| 模块 | refs / 复用 |
|---|---|
| `grasp/ditto_warp.py` | [DITTO](https://github.com/robot-learning-freiburg/DITTO) `tracking_3D.warp_3D_trajectory` + `geometry.py` |
| `grasp/demo_approach.py` | DITTO warp + [DexGraspBench](https://github.com/JYChen18/DexGraspBench) `tabletop_mocap.py` 分阶段 |
| `grasp/qpos_pipeline.py` | `run_bimanual_demo_warp_grasp`（单 ep manifest，无 canonical npz） |
| `grasp/lift_reference.py` | 抬升段物体系路点（同 warp 思路） |
| `scripts/validate_grasp_qpos.py` | `--ep N --seed S --video` |

论文链接：[DITTO arXiv:2403.15203](https://arxiv.org/abs/2403.15203) · [DexGraspBench](https://github.com/JYChen18/DexGraspBench)

旧路径（可选）：`qpos_distill.py` / `canonical_*_npz` 已非主入口。

Laplacian 路径（`grasp/ik.py`）保留但非 phase-1 主路径。

## dexjoco 编排层（不 mirror）

| 模块 | 调用 |
|---|---|
| `grasp/ik.py` | holosoma Laplacian + pyroki 09 代价 |
| `grasp/staged_grasp.py` | DexGraspBench staged + Dexonomy squeeze |
| `tpsr/sim_refine.py` | Dexonomy sim refine lite |
| `contact_refine/` | ContactOpt + GraspTTA + δ* |
| `skill_replay/` | 在线 retrieve → grasp → lift → insert |

## 同步命令

```bash
bash scripts/sync_refs_to_interaction_retarget.sh
```

同步后若 `qp.py` 被覆盖，需保留 `from interaction_retarget.DexGraspBench.src.util.rot_util import ...`。

## pyroki / TopoRetarget

- `refs/pyroki/examples/09_hand_retargeting.py` — IK local/global 项（逻辑在 `grasp/ik.py`）
- `refs/TopoRetarget/` — 无代码，规格见 `constants.py`（N_h=21, N_o=50）
