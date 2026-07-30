# trex_dexjoco

T-Rex post-train on DexJoCo **bimanual_assembly**（动作 44 + 原生触觉重训 VQ-VAE）。

上游：`../refs/T-Rex`（`UPSTREAM_COMMIT.txt`）。细节见 [`docs/TRAIN_REPORT.md`](docs/TRAIN_REPORT.md)。

---

## 0. 下载权重（必做，不能用 π0.5）

落盘目录：`/mnt/hdd/checkpoints/trex/`

```bash
mkdir -p /mnt/hdd/checkpoints/trex
pip install -U "huggingface_hub[cli]"   # 若还没有 hf

# 底座 VLM（Qwen3-VL-2B）
hf download Qwen/Qwen3-VL-2B-Instruct \
  --local-dir /mnt/hdd/checkpoints/trex/Qwen3-VL-2B-Instruct

# T-Rex midtrain（后训练起点）
hf download miniFranka/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6 \
  --local-dir /mnt/hdd/checkpoints/trex/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6
```

下完检查：

```bash
ls /mnt/hdd/checkpoints/trex/Qwen3-VL-2B-Instruct/config.json
ls /mnt/hdd/checkpoints/trex/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6/model.pt
# 若 midtrain 目录结构不同，以实际文件为准；需能被 train.py --resume_checkpoint 读到
```

需要 HF token 时：`huggingface-cli login` 后再下。

依赖钉死（与上游一致，已在本机烟雾通过）：

```bash
conda activate dexjoco
pip install "transformers==4.57.3" "huggingface_hub>=0.34.2,<0.36" timm einops datasets qwen-vl-utils accelerate lerobot
```

烟雾测试（3 step）：`bash scripts_dexjoco/smoke_posttrain.sh`  
通过后再：`bash scripts_dexjoco/train_posttrain.sh`

---

## 1. 数据准备 + 训练

```bash
cd /home/wangrenpeng/dexjoco/trex_dexjoco
export PYTHONPATH=$PWD

# norm stats（若已生成可跳过）
python -m adapters.compute_norm_stats

# 重训 DexJoCo 触觉 VQ-VAE（必须先于 post-train）
bash scripts_dexjoco/train_vqvae.sh

# post-train（默认 WANDB offline；脚本内默认用当前可见 GPU）
bash scripts_dexjoco/train_posttrain.sh
```

### 常用启动方式

```bash
cd /home/wangrenpeng/dexjoco/trex_dexjoco
conda activate dexjoco
export PYTHONPATH=$PWD

# wandb 在线（需已 wandb login）
WANDB_MODE=online bash scripts_dexjoco/train_posttrain.sh

# 只用 2 号卡
CUDA_VISIBLE_DEVICES=2 bash scripts_dexjoco/train_posttrain.sh

# 2 号卡 + wandb 在线
CUDA_VISIBLE_DEVICES=2 WANDB_MODE=online bash scripts_dexjoco/train_posttrain.sh

# 短训默认：10 epoch、每 epoch 存盘、lr=3e-5、weight_decay=0.01
# 可覆盖：N_EPOCHS=5 SAVE_FREQ=1 LR=1e-5

# 若 DataLoader segfault：NUM_WORKERS=0（会变慢）；正常默认 NUM_WORKERS=4
```
级联参数未改：`total_steps=10`，`split_step=6`。

---

## 2. 仿真评估（与 π0.5 / ForceVLA 同格式）

结果目录（仓库根下 `outputs` → `/mnt/hdd/dexjoco/outputs`）：

```text
outputs/trex/bimanual_assembly_seed0_ckpt000013/
  episode_00_success/   # ego.mp4, wrist_left.mp4, wrist_right.mp4
  episode_01_failure/
  ...
  success_rate_12_50.txt
```

```bash
cd /home/wangrenpeng/dexjoco
conda activate dexjoco
export PYTHONPATH=/home/wangrenpeng/dexjoco/trex_dexjoco:/home/wangrenpeng/dexjoco:/home/wangrenpeng/dexjoco/dexjoco
export MUJOCO_GL=egl

# 默认用 checkpoint-13；换权重设 CHECKPOINT=...
CUDA_VISIBLE_DEVICES=2 \
CHECKPOINT=/mnt/hdd/checkpoints/trex_dexjoco_ckpt/bimanual_assembly/trex_posttrain_bimanual_assembly/trex_posttrain_bimanual_assembly_0728_2128/checkpoint-13-44646 \
SEED=0 EPISODES=50 OVERWRITE=1 \
bash trex_dexjoco/scripts_dexjoco/eval_posttrain.sh
```

说明：读 sim 指力（`force_mode=finger`）→ 触觉 `[8,3]`；动作 chunk=16；协议对齐 π0.5（50s / 1500 帧封顶）。
