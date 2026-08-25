# Bimanual Physics RL

DexJoCo 双臂抓取加插孔项目。教师入口是 `causal.py`，并行 RL 入口是 `main.py`；两者都不使用旧的根状态恢复策略。

## 当前方法

1. 从训练划分内的 episode 1 离线提取左右手抓取与抬升技能，并转换到物体初始坐标系。
2. 每次评估都从 DexJoCo 原生随机 `reset` 开始，只按当前插头和孔座初始位姿变换这两个固定技能。
3. 双抓完成后，仅使用当前 MuJoCo 状态做闭环控制：双臂分担横向误差，并分别绕插头尖端和孔中心分担姿态误差。
4. 下插时横向与轴向分别限幅，持续保持姿态闭环；阶段只有在误差连续 10 步达标后才推进。
5. 在线 RL 不直接输出手臂位姿，只在 `0.5x-1.5x` 内调节居中、定姿和下插三个正增益；控制方向和抓取轨迹不能被策略改写。

评估过程中不恢复示范状态，也不读取当前回合或留出集的未来示范动作。成功严格采用 DexJoCo/OpenPI 原生 `info["succeed"]`：插头与孔底接触连续保持 30 步。

## 已验证结果

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

当前方案使用仿真特权物体状态，不是最终视觉策略。粗抓取阶段仍由一个训练示范生成的因果物体坐标技能初始化；下一项科学问题是把教师蒸馏成当前观测/视觉策略，并在更宽的摩擦、时延和几何扰动下验证三增益 RL 是否产生真实优势。

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
