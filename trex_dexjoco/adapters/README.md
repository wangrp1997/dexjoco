# Adapters: DexJoCo → T-Rex post-train (decisions B + T1)

Do **not** pad zeros or remap to Sharpa 62 / [10,6].

| Script | Role |
|--------|------|
| `prep_vqvae_data.py` | force parquet → `[N,8,3]` hdf5 tree for VQ-VAE |
| `compute_norm_stats.py` | q01/q99 for absolute action chunks + tactile |
| `dexjoco_dataset.py` | `--data_format dexjoco` train loader |

Order before training: see `docs/TRAIN_REPORT.md`.
