# DexJoCo × T-Rex 适配报告（训练前）

决策：**动作 B（44）** + **触觉 T1（原生力重训 VQ-VAE）** + **deform 关** + 权重 `/mnt/hdd/checkpoints`。

---

## 1. 结论摘要

| 项 | 做法 | 说明 |
|----|------|------|
| 动作 | `ACTION_DIM=44` | 右/左各 xyz+rotvec+Allegro16；**不**凑 62 |
| 动作表示 | 绝对 chunk `[16,44]` | 边沿 pad；**不**发明 Sharpa 式 delta-base |
| 触觉 | `[8,3]` | 左4指×xyz + 右4指×xyz；**不**补零成 `[10,6]` |
| VQ-VAE | 在 DexJoCo 力上重训 | midtrain 内嵌 F6 VQ-VAE **不用** |
| midtrain | `resume_source=midtrain` | shape 不匹配的头（动作/触觉 VQ）自动跳过重初始化 |
| MoT / 级联 | 未改 | `total=10 split=6 dropout=0.1 loss=1.0` |
| deform | `0` | 无数据 |
| 相机 | ego + 2 wrist | ego 作 slow/head |

---

## 2. 已改代码（`trex_dexjoco/`）

- `utils/lerobot_common.py`：44 / 8×3 常量  
- `qwen_vla/modeling_vla.py`：按 `n_fingers` 切片；stats buffer 随 VQ 配置  
- `scripts/train.py`：`--data_format dexjoco`；F6 历史不依赖 deform  
- `tactile_vqvae/*`：`n_fingers=4 per_finger_dim=3`；手局部归一化  
- `adapters/`：数据准备 + `DexJoCoSftDataset`  
- `scripts_dexjoco/train_vqvae.sh`、`train_posttrain.sh`

级联 flow / MoT 核心逻辑未重写。

---

## 3. 训练前你必须做的步骤

### 3.1 下载权重 → `/mnt/hdd/checkpoints/trex/`

```bash
mkdir -p /mnt/hdd/checkpoints/trex
pip install -U "huggingface_hub[cli]"

hf download Qwen/Qwen3-VL-2B-Instruct \
  --local-dir /mnt/hdd/checkpoints/trex/Qwen3-VL-2B-Instruct

hf download miniFranka/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6 \
  --local-dir /mnt/hdd/checkpoints/trex/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6
```

不能用 π0.5 替代。详见仓库根目录 `README.md`。

### 3.2 算归一化统计

```bash
cd /home/wangrenpeng/dexjoco/trex_dexjoco
export PYTHONPATH=$PWD
python -m adapters.compute_norm_stats
# → /mnt/ssd/datasets/trex_dexjoco/bimanual_assembly/trex_norm_stats.json
```

### 3.3 训 DexJoCo VQ-VAE（先于 post-train）

```bash
bash scripts_dexjoco/train_vqvae.sh
# → /mnt/hdd/checkpoints/trex/vqvae_dexjoco_allegro8x3/latest.pt
```

可用 `--smoke_test 1` 先通管线（需在 `tactile_vqvae.train` 加对应 flag 已存在）。

### 3.4 Post-train

```bash
bash scripts_dexjoco/train_posttrain.sh
```

环境需能 `import torch, transformers, accelerate, lerobot`（建议单独 conda）。

---

## 4. 数据路径

| 用途 | 路径 |
|------|------|
| LeRobot | `/mnt/ssd/datasets/dexjoco_lerobot_datasets/bimanual_assembly` |
| 力标签 | `.../force_labels/forces.parquet`（54170 帧） |
| norm stats | `/mnt/ssd/datasets/trex_dexjoco/bimanual_assembly/trex_norm_stats.json` |
| VQ 数据 | `/mnt/ssd/datasets/trex_dexjoco/vqvae_f6_data/dexjoco_assembly/` |
| post-train ckpt | `/mnt/hdd/checkpoints/trex_dexjoco_ckpt/` |

---

## 5. 刻意不做 / 已知代价

1. **动作头、触觉 code embedder、VQ-VAE**：不继承 midtrain 对应权重（维不同）。  
2. **骨干 / 视觉 / 能对上的 MoT**：仍从 midtrain 加载。  
3. **绝对 chunk ≠ 上游 delta-base**：贴合 DexJoCo 控制，不是字面复现他们的动作数学。  
4. **腕部 FT 未接入**：只用指尖 3 轴力（与现有 force_labels 一致）。  
5. **仿真评测客户端**：未做（Q7）；先训通再接 MuJoCo。  
6. **flare**：仍用 ego 未来帧；需确认 LeRobot 随机访问够快。

---

## 6. 建议验收顺序

1. `compute_norm_stats` 成功  
2. `prep_vqvae_data` + VQ-VAE smoke / 短训  
3. post-train 跑通 1–2 step（bsz=1）看 loss  
4. 再开满 epoch  

有问题先停，不要盲训满程。
