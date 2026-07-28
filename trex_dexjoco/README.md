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

# post-train
bash scripts_dexjoco/train_posttrain.sh
```

级联参数未改：`total_steps=10`，`split_step=6`。
