**结论：分阶段方案合理，插孔用 PoseInsert 比 OpenTrack 更贴任务；但要接一层 handoff + 右臂执行适配。**

---

## PoseInsert 在做什么

论文/代码核心是 **物体系相对位姿模仿 + Diffusion**：

| 组件 | 作用 |
|------|------|
| Pose encoder | source（peg）在 target（孔/tray）系下的 6D 位姿 |
| RGBD encoder（RPDP） | 视觉补精度 |
| Residual Gated Fusion | pose 与视觉融合 |
| Diffusion decoder | 预测未来 **20 步 × 9d** 相对位姿轨迹 |

**前提**（README 写明）：policy 启动前 **已经抓住 source 物体**。  
真机是 Mobile ALOHA **单臂** + FoundationPose 估 peg/孔位姿。

和你们任务的对齐点：
- 抓取/抬升 = 接触丰富、长 horizon → 适合 VLA
- 插孔 = 短、高精度、相对几何 → 正是 PoseInsert 设计场景

---

## 和你们 pipeline 的匹配度

```
ForceVLA / π0.5          PoseInsert (RPDP)
─────────────────        ────────────────────
tray 抓 + 抬              右臂：peg 相对 tray 孔
peg 抓 + 抬      handoff → Diffusion 出相对位姿轨迹
（可选）移到孔上方          左臂：持 tray 冻结
```

**合适的地方：**
1. 和 `docs/oig_tpsr_plan.md` 一致：L0/L1 抓抬，L2 插孔单独测
2. 已有 `dexjoco/hybrid_insert/`：handoff 条件（approach cylinder、左臂冻结）可直接复用
3. `interaction_retarget/skill_replay/insert.py` 已按 hybrid handoff 编排，PoseInsert 可替换 ALIGN/INSERT 几何段
4. ForceVLA 你们已有 `bimanual_assembly` ckpt，比从零上 π0.5 更顺

**要补的 gap（不是算法不对，是工程）：**

| 问题 | 说明 |
|------|------|
| 单臂 → 双臂 | PoseInsert 不管左臂；handoff 后左臂必须 lock（`hybrid_insert` 已支持） |
| 9d 位姿 → 23d 右臂 | 需 adapter：相对位姿 → mocap wrist + Allegro 指形锁定 |
| 感知 | 真机要 peg/tray 6D pose（FoundationPose 或 sim 特权 body pose） |
| 数据 | 要插孔段 demo：`source_pose/target_pose`（+ 可选 RGBD），不能直接用 46d zarr |
| approach 段 | PoseInsert **不负责**「从抬升位移到孔上方」；仍要 VLA 或现有 geometry approach |
| 真机栈 | 原仓库 ROS + Python3.8 + 硬编码路径，需 sim adapter |

---

## 和现有方案的对比

| 插孔方案 | 优点 | 缺点 |
|----------|------|------|
| **hybrid_insert（几何）** | 已有、sim 特权、零训练 | 难泛化、无视觉 |
| **PoseInsert** | 相对位姿 + 可选 RGBD，专精插孔 | 要数据 + adapter |
| **ForceVLA 端到端** | 一条 policy | 精密插孔通常弱 |
| **DexOpenTrack** | 闭环跟踪 | 对短插孔 motion 过重 |

插孔阶段 **优先 PoseInsert > OpenTrack**，判断成立。

---

## 推荐落地顺序

**Phase A — 抓抬（不动 PoseInsert）**  
ForceVLA（或 π0.5）做到：tray/peg 抓稳 + 双物体抬到 insert 预备区。  
成功标准沿用 phase-1：tray ≥30mm、peg ≥10mm。

**Phase B — Sim Privileged PoseInsert**  
1. 从 zarr 切 `peg_lift_start` → insert success 段  
2. 用 sim body pose 生成 `source_in_target`（不必先上 RGBD）  
3. 训 **PoseDP**（纯 pose 版）  
4. 经 `EvalHybridInsert` handoff 接右臂 wrist 执行

**Phase C — Sim RPDP + 真机**  
加 ego/wrist RGBD；真机 FoundationPose；再 eval_ros_RPDP 思路。

**Handoff 触发**（建议沿用现有逻辑，不交给 VLA 判）：
- peg 已抓、tray 已抓且抬够
- peg tip 进入 approach cylinder（`hybrid_insert.geometry.in_approach_cylinder`）
→ 切 PoseInsert，左臂 freeze

---

## 一句话

**VLA 管抓抬、PoseInsert 管插孔，架构对；PoseInsert 不是 DexOpenTrack 的替代，而是 L2 专用模块。**  
关键在 handoff 条件、右臂 9d→mocap 执行层、插孔段训练数据——你们 `hybrid_insert` 已经铺了一半路。

若要推进，下一步可以是：**从 ep35 zarr 切 insert 段 + 写 sim 版 `source_pose/target_pose` 导出脚本**（privileged，先不碰 RGBD）。

---

## Phase B-1（已实现）

模块说明与命令见 [`dexjoco/pose_insert/README.md`](../dexjoco/pose_insert/README.md)。

输出目录与 sidecar 同级：`/mnt/hdd/dexjoco/poseinsert_sim/`。

```bash
export PYTHONPATH=~/dexjoco:~/dexjoco/dexjoco MUJOCO_GL=egl
python scripts/export_insert_poses.py --ep 35
python scripts/export_insert_poses.py --all
```