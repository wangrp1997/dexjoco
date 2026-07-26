# 双臂避障算法选型（有依据）

## 结论（先看这个）

| 用途 | 选用 | 原因 |
|------|------|------|
| **主规划（重抓接近）** | **Cartesian TrajOpt**（本仓库 `cartesian_trajopt.py`） | 在现有 MuJoCo mocap 控制下可直接优化；目标是碰撞代价+平滑，不是手写路点 |
| **安全层（在线）** | CBF-QP（`dual_arm_cbf_qp.py`） | 只做短视距修正；左手抓 tray 时冻结左手 |
| **GPU 关节空间升级** | **cuRoboV2**（conda 环境 `curobo`） | 开源里最强的碰撞无关运动生成；需导出 Panda+Allegro 配置后接入 |
| **否决** | 手写高点/侧向路点 | 不平滑、易奇异、不像智能体策略 |

之前只用 CBF-QP 当主方案 **依据不足**：CBF 适合反应式避障，不适合生成整段可学习的重抓轨迹。

## 算法对比（针对本任务）

任务约束：DexJoCo 双臂 Allegro、动作是 **mocap EE**、左手常抓着 tray、要离线生成可学习轨迹。

| 算法 | 类型 | 优点 | 缺点 | 对本任务 |
|------|------|------|------|----------|
| 手写路点绕行 | 启发式 | 实现快 | 刻意、不稳、奇异 | **否决** |
| CBF-QP | 反应式约束控制 | 在线、可证局部安全 | 短视、不规划全局路径；乱动左手会甩 tray | **仅安全层** |
| MPC / MPPI | 滚动优化 | 可预测未来 | 调参重；mocap 动力学模型弱 | 可作后续 |
| Mink 碰撞 IK | MuJoCo QP-IK | 和仿真一体 | 要 mujoco≥3.8，与 dexjoco 的 3.4 **冲突**（已回退卸载） | 暂不用 |
| OMPL (RRT*) | 采样规划 | 全局 | 关节空间配置/碰撞模型工作量大 | 备选 |
| **Cartesian TrajOpt** | 笛卡尔轨迹优化 | 直接优化 mocap 路径；左臂冻结；OSQP 再平滑 | 不是关节力矩级最优 | **当前主用** |
| **cuRobo / cuRoboV2** | GPU MotionGen+TrajOpt | 工业/学界常用；碰撞球+TrajOpt+可双臂 | 要 robot yml/URDF；双臂仍偏研究向 | **已安装，待接机器人配置** |

## 环境

### dexjoco（数据生成）

```bash
conda activate dexjoco
# mujoco==3.4.0（不要装 mink 1.2，会强升 mujoco）
# 已有：osqp, qpsolvers, torch, jax
```

### curobo（新建，Python 3.10）

```bash
conda activate curobo
# 已装：torch 2.11+cu128, nvidia-curobo (cuRoboV2 @ /home/wangrenpeng/deps/curobo)
python -c "from curobo.motion_planner import MotionPlanner; print('ok')"
```

自带 Franka / dual_ur10e 配置；接我们的 Panda+Allegro 需要导出 URDF + sphere 碰撞模型（下一步）。

## 管线里现在怎么用

1. 失败段：左手 **完全冻结**（demo hold），禁止 CBF 推 tray  
2. drop：只松接触指；先松握得松的、**拇指最后**（并行松圆 peg 会翻）  
3. 重抓接近：`CartesianCollisionTrajOpt` 优化右臂路径避开左臂/tray  
4. 到位 dwell → 合手 → dwell → 慢抬  
5. CBF-QP：仅执行时安全过滤，且 `freeze_left=True`
