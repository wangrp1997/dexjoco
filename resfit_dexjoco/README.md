# ResFiT-style residual RL for DexJoCo

Frozen ForceVLA BC + learnable residual TD3 on `bimanual_assembly`.

> **与 ResFiT 原版的差异：** ResFiT 残差 policy 输入为 **图像 + state + base_action**；本版 v1 **不用图像**，只用 **state + base_action**（视觉由冻住的 ForceVLA 负责，残差在 sim 里学 proprio 修正）。

## Layout

```
resfit_dexjoco/
  env/                  # OpenPI env, residual wrapper, milestone reward, rl_obs
  bc/                   # ForceVLA websocket client + action buffer
  training/             # ResFiT-adapted TD3, offline loader, replay buffer
  scripts/
    smoke_zero_residual.py
    train_residual_td3.py
```

## Smoke test (Δa=0)

```bash
cd ~/dexjoco
conda activate dexjoco
export MUJOCO_GL=egl
python resfit_dexjoco/scripts/smoke_zero_residual.py \
  --config configs/rand_obj/bimanual_assembly.yaml \
  --seed 0 --port 8000 --episodes 1 --force-mode both
```

## Train residual TD3

**终端 1 — ForceVLA serve (GPU）**

**终端 2 — 训练（需 torch + pyarrow + ResFiT 依赖）**

```bash
cd ~/dexjoco
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=""   # 训练侧不用 GPU，serve 独占 GPU 0
export PYTHONPATH=~/dexjoco:~/dexjoco/dexjoco:~/dexjoco/third_party/residual-offpolicy-rl

python resfit_dexjoco/scripts/train_residual_td3.py \
  --seed 0 --port 8000 --force-mode both \
  --use-offline-data --offline-fraction 0.5
```

重训覆盖旧 checkpoint 加 `--overwrite`。

默认 checkpoint：`/mnt/ssd/checkpoints/resfit_dexjoco_ckpt/bimanual_assembly/forcevla_both/checkpoint_step_XXXXXX.pt`（与 ForceVLA/DexQuery 一样，路径里不含 seed）

W&B：project **`dexjoco`**，run 名 **`resfit_forcevla_both`**（对标 `forcevla_both`、`dexquery_bimanual_assembly`）；seed 只在 config 里；关闭加 `--no-wandb`。

### 默认（对齐 ResFiT v1）

| 项 | 默认 |
|---|---|
| RL obs | 46 + 44 = 90 维（`--privileged-sim-state` 开 → 61+44） |
| 离线数据 | `/mnt/ssd/datasets/dexjoco_lerobot_datasets/bimanual_assembly` |
| offline_fraction | 0.5（`--no-use-offline-data` 或 `--offline-fraction 0` 关闭） |
| 离线 base | GT-as-base，sparse terminal reward |
| 在线 reward | milestone（ResiP 三阶段 + success） |
| 算法 | proprio TD3（ResFiT Actor/Critic/normalization 复用） |

## Dependencies

- **sim + BC**：`conda activate dexjoco`
- **训练**：同上 + `torch`, `pyarrow`；`PYTHONPATH` 含 `third_party/residual-offpolicy-rl`
