# pose_insert

抓抬完成 → PoseInsert 插孔。

| 模式 | 抓抬阶段 |
|------|----------|
| `demo_replay` | zarr 回放到 `peg_lift_end` |
| `policy` | ForceVLA / π0.5 等（eval 里 `observe` 直到抓抬达标 → `merge`） |

**标准流程（默认）**
- **训练** obs/action：peg 相对孔 **9D**（PoseInsert 论文格式）
- **Eval 执行**：策略输出 9D 相对轨迹 → **双臂腕部跟踪**（右跟 peg、左跟孔/tray），手指 handoff 锁死
- 仅右臂 + 左锁：`--no-bimanual`

## 环境

```bash
conda activate dexjoco
cd ~/dexjoco
export PYTHONPATH=~/dexjoco:~/dexjoco/dexjoco MUJOCO_GL=egl
pip install einops diffusers
```

## 1. 导出

```bash
python scripts/export_insert_poses.py --all
# -> /mnt/hdd/dexjoco/poseinsert_sim/bimanual_assembly/train/{ep}/
# 主要用 source_in_target.npy；dual_wrist_action.npy 仅 --wrist12 训练需要
```

删旧失败 ep（若还在）：

```bash
rm -rf /mnt/hdd/dexjoco/poseinsert_sim/bimanual_assembly/train/{0,1,2,7}
```

## 2. 训练（9D pose，默认）

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/train_pose_insert_sim.py \
  --data-root /mnt/hdd/dexjoco/poseinsert_sim/bimanual_assembly \
  --ckpt-dir /mnt/hdd/dexjoco/poseinsert_sim/checkpoints \
  --num-epochs 500 --batch-size 240 --save-every 50
# -> .../checkpoints/policy_last.ckpt
```

冒烟：

```bash
python scripts/train_pose_insert_sim.py --smoke
```

## 3. Eval（默认双臂跟踪 9D）

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/eval_pose_insert_sim.py --ep 35 \
  --ckpt /mnt/hdd/dexjoco/poseinsert_sim/checkpoints/policy_last.ckpt \
  --video --debug
```

仅右臂（左锁）：

```bash
python scripts/eval_pose_insert_sim.py --ep 35 --no-bimanual \
  --ckpt /mnt/hdd/dexjoco/poseinsert_sim/checkpoints/policy_last.ckpt
```

## 4. VLA + PoseInsert

ForceVLA eval：`EvalBimanualPoseInsert.observe` 直到抓抬达标 → `merge` 双臂插孔。

## 附录：12D 腕部 label 训练（不推荐）

```bash
python scripts/train_pose_insert_sim.py --wrist12 \
  --ckpt-dir /mnt/hdd/dexjoco/poseinsert_sim/checkpoints
# -> .../checkpoints/wrist12/policy_last.ckpt
```
