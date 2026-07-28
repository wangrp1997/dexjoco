# 适配结论（已拍板）

- **Q1 动作：B** — `ACTION_DIM=44`，midtrain 动作头重初始化  
- **Q2 触觉：T1** — 原生 `[8,3]`，重训 VQ-VAE；不用 midtrain F6 编码器  
- **Q3 deform：关**  
- **Q6 权重**：`/mnt/hdd/checkpoints/trex_...`

细节与训练步骤见 [`TRAIN_REPORT.md`](TRAIN_REPORT.md)。
