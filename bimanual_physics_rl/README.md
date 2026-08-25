# Bimanual Physics RL

DexJoCo 双臂抓取加插孔项目。教师入口是 `causal.py`，并行 RL 入口是 `main.py`；两者都不使用旧的根状态恢复策略。

## 非特权策略状态

合规状态：历史关键位姿学生**不合规**；最终公开视觉判定实验合规但失败。符合约束的完整插孔成功为 **0/1**，公开 RGB 抓取/抬升诊断同样为 **0/1**。单回合结果只用于停止决策，不是稳定成功率。

此前的关键位姿学生虽然评估时只读取三路 RGB 和 46 维本体状态，但训练标签、12 个动作时刻和轨迹结构来自特权教师及示范。因此它属于特权监督和压缩示范回放，观察到的双抓 `4/4` 只能作为不合规诊断，必须从非特权成功率中剔除。相关历史产物位于：

```text
/mnt/hdd/dexjoco/outputs/bimanual_physics_rl/student_rgbp_v1/model_keypose_rgbhead.pt
/mnt/hdd/dexjoco/outputs/bimanual_physics_rl/student_rgbp_v1/eval_keypose_rgbhead_seed130000_n4.json
```

### 公开视觉判定实验（2026-08-25）

- 输入：`ego`、`wrist_left`、`wrist_right`、`state[:46]`；训练反馈只来自公开 RGB 特征变化和原生标量奖励。
- 禁止项：未使用示范、教师、模板、物体位姿、接触真值或阶段真值。
- 方法：HSV 目标分割、在线有限差分像素雅可比、通用 Allegro 闭手姿态；没有保存或推广失败原型代码。
- 固定评估：种子 `210000`，原生 `info["succeed"] = false`、奖励 `0`；抬升后的公开 RGB 仍显示插头与孔座留在桌面，抓取失败。
- 失败机制：原生奖励只在完整插入并连续保持 30 步后为 `1`，此前始终为 `0`；图像对准也不能确定物体进入真实抓取开口。
- 决策：停止该物理 RL/视觉伺服路线，不再继续 PPO/SAC 盲目探索或参数搜索。

## 特权教师基线

1. 从训练划分内的 episode 1 离线提取左右手抓取与抬升技能，并转换到物体初始坐标系。
2. 每次评估都从 DexJoCo 原生随机 `reset` 开始，只按当前插头和孔座初始位姿变换这两个固定技能。
3. 双抓完成后，仅使用当前 MuJoCo 状态做闭环控制：双臂分担横向误差，并分别绕插头尖端和孔中心分担姿态误差。
4. 下插时横向与轴向分别限幅，持续保持姿态闭环；阶段只有在误差连续 10 步达标后才推进。
5. 在线 RL 不直接输出手臂位姿，只在 `0.5x-1.5x` 内调节居中、定姿和下插三个正增益；控制方向和抓取轨迹不能被策略改写。

评估过程中不恢复示范状态，也不读取当前回合或留出集的未来示范动作。成功严格采用 DexJoCo/OpenPI 原生 `info["succeed"]`：插头与孔底接触连续保持 30 步。

## 特权基线已验证结果

原生随机初始状态，开启环境自带的插头/孔座质量随机化（各自独立 `0.75x-1.25x`）：

| 回合 | 完整成功 | 双抓 | 双物体抬升 | 成功步数 |
| ---: | ---: | ---: | ---: | ---: |
| 20 | **20/20** | 20/20 | 20/20 | 平均 913.7，范围 840-1249 |

种子为 `83000-83019`，原始结果位于：

```text
/mnt/hdd/dexjoco/outputs/bimanual_physics_rl/causal_eval/dynamics_v2_*.json
```

同一组 20 回合完整原生 reset 对照（基础 seed `85000/85005/85010/85015`）：

| 策略 | 训练 | 成功 | 平均步数 | 结论 |
| --- | ---: | ---: | ---: | --- |
| 零残差教师 | 0 | **20/20** | **916.15** | 默认 |
| 12 维位姿残差 PPO | 50k | 20/20 | 937.15 | 更慢，拒绝 |
| 12 维位姿残差 PPO | 100k | 18/20 | 971.00 | 破坏成功率，拒绝 |
| 3 维物理增益 PPO | 51.2k | **20/20** | 912.80 | 安全，但尚无显著优势 |

三增益策略相对教师平均快 3.35 步，但配对中位差为 0，提升由单个回合主导，因此不替代默认教师。正式 RL 模型和评估 JSON 位于：

```text
/mnt/hdd/dexjoco/outputs/bimanual_physics_rl/causal_gain_rl_50k_20260825/
```

上述 `20/20` 方案使用仿真特权物体状态，只是教师上界，不属于可交付结果。当前唯一符合约束的判定实验为抓取 `0/1`、完整插孔 `0/1`，项目结论是失败并停止。

## 复现

```bash
cd /home/wangrenpeng/dexjoco
export PYTHONPATH="$PWD:$PWD/dexjoco:$PYTHONPATH"
export MUJOCO_GL=egl
PY=/home/wangrenpeng/miniconda3/envs/dexjoco/bin/python

$PY -m bimanual_physics_rl.causal build-templates
$PY -m unittest bimanual_physics_rl.test_env
$PY -m bimanual_physics_rl.causal eval \
  --episodes 5 --seed 83000 --max-steps 1500 \
  --randomize-dynamics \
  --output /mnt/hdd/dexjoco/outputs/bimanual_physics_rl/causal_eval/check.json

$PY -m bimanual_physics_rl.main train \
  --causal-templates /mnt/hdd/dexjoco/outputs/bimanual_physics_rl/causal_templates/train_ep001_skill \
  --causal-warm-start --randomize-dynamics \
  --envs 8 --steps 50000 --rollout-steps 256 --batch-size 512 \
  --learning-rate 0.00003 --target-kl 0.02 \
  --output /mnt/hdd/dexjoco/outputs/bimanual_physics_rl/causal_gain_rl
```

可用不同 `--seed` 同时启动多个评估进程；四个 5 回合进程完成上述 20 回合评估。
