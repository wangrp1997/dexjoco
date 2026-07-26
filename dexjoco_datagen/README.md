# dexjoco_datagen

失败两种 → MimicGen SE(3) 重抓。

避障选型见 [ALGORITHM.md](ALGORITHM.md)。

主规划：笛卡尔碰撞轨迹优化；安全层：CBF-QP（左手抓 tray 时冻结）。
cuRobo 在 conda 环境 `curobo`，待接机器人配置。

## 模式

| 模式 | 含义 |
|------|------|
| `drop` | 抬约 3 cm 后接触指同步轻轻减弱握力滑落 |
| `grasp_fail` | 欠合手刚抬一点就反应 |

## 生成

```bash
cd ~/dexjoco
export MUJOCO_GL=egl PYTHONPATH=~/dexjoco:~/dexjoco/dexjoco
conda activate dexjoco
python scripts/gen_regrasp_videos.py --clean --n 5 --episodes 10 12 14 20 25
```
