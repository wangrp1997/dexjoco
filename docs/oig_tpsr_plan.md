# OIG + TPSR 集成计划

在 OIG（物体系 interaction 原型）之上，借 Dexonomy / BODex / DexGraspBench 的**物理验 grasp**能力，同时保留 demo 蒸馏的任务拓扑（轴抓、孔口净空、后续插孔）。

## 目录布局（均在 `interaction_retarget/`）

```
interaction_retarget/
  grasp/              # 已有 + 薄编排层
    distill.py        # OIG 离线
    ik.py             # seed only
    execute_tpsr.py   # 调 tpsr
    agent_tpsr.py
    pipeline_tpsr.py
  tpsr/               # Topology-Preserving Sim Refinement
    config.py
    constraints.py    # δ* 漂移 + 孔口禁区
    refine.py
  bench/              # DexGraspBench 式验稳
    config.py
    verify.py
  sim/                # 已有
  laplacian.py        # 已有
```

原则：核心算法进 `tpsr/`、`bench/`；`grasp/` 只保留编排与 execute 入口。

## 方法三节（paper）

1. **OIG Prototype Distillation** — many demo → δ*（`grasp/distill.py`）
2. **TPSR** — sim refine 锚定 δ* + 任务约束（孔口禁区）
3. **Deploy & Verify** — IK seed → approach → TPSR → DexGraspBench 式 hold 验稳 → 双臂装配链

## 数据流

```
随机 T_world_obj
  → load canonical δ*
  → Laplacian IK（seed only，非 success 判据）
  → GenHand pre→grasp→close
  → TPSR refine（contact + 拓扑 + 孔口）
  → bench.verify_hold（替代纯 contact count）
  → tray lift → hold → peg grasp → …
```

## 从 refs 借什么

| 来源 | 借 | 不借 |
|------|----|------|
| Dexonomy | sim local refine、contact-aware control | 整套 dexsyn、human tmpl |
| BODex | FC QP 思路、分阶段 contact | cuRobo 全链在线生成 |
| DexGraspBench | 穿透/扛外力/hold 指标定义 | 原样 hand XML eval |

## 实施阶段

- **P0**（当前）：`tpsr/` + `bench/` 骨架；`pipeline_tpsr` + `validate_grasp_tpsr.py`
- **P1**：TPSR 迭代加强（Dexonomy 式多步 settle + 拓扑 loss）
- **P2**：peg 孔口禁区、tray socket 净空；ablation 去拓扑项
- **P3**：可选 BODex FC 离线 gold 对照

## 怎么结合（不是单纯加 Lap loss）

分三层，各干各的：

| 层 | 来源 | 干什么 |
|----|------|--------|
| **表示** | OIG | demo → δ* + lift 路点（物体系） |
| **放置** | Dexonomy 思路 | sim 里 refine 贴物体，**锚住 δ***（过滤或软约束，P1 可改联合优化） |
| **验收** | DexGraspBench 思路 | contact + 持稳 + 位移；**不负责生成 grasp** |

L0 成功要过 **三道关**（`bench/lift_verify.py`）：

1. **grasp_ok**：contact 够 + Lap 没漂太多（拓扑还在）
2. **lift_ok**：tray 抬到目标高度 + 手腕物体系姿态接近 demo lift 终点
3. **hold_ok**：提升后 contact 不断 + tray 不掉（`hold_tray_before_peg`）

提升轨迹仍用 GenHand + `demo_lift_reference` 物体系路点，不是 Dexonomy 生成。

## 验证范围（分阶段，**不含插孔**）

| 阶段 | 内容 | 脚本 |
|------|------|------|
| **L0** | 左臂 tray 抓取 → 提升 → 持稳 | `validate_grasp_tpsr.py`（默认 `--stage tray_lift`） |
| **L1** | L0 + 右臂 peg 抓取（仍不插孔） | `--stage bimanual_grasp` |
| **L2** | 插孔 / hybrid_insert | **另测**，不在 grasp pipeline 里 |

孔口禁区（`tpsr/constraints`）是抓取拓扑约束，为以后插孔留净空，**不是**在跑 insert。

## 成功指标（L0 tray_lift）

主：tray bench pass + lift 后 hold 稳定  
辅：Laplacian RMSE、contact count  
**不**要求 peg、**不**报 insert 成功率
