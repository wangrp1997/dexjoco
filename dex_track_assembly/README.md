# dex_track_assembly

## 第一步：本包 Python 环境

```bash
# 未装 uv 时先装（只需一次）
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env   # 或重新开终端
```

```bash
cd ~/dexjoco/dex_track_assembly
uv sync -i https://pypi.org/simple
```

`uv sync` 会按 `pyproject.toml` 自动拉 **Python 3.12.9** 到本目录 `.venv`，不占用 conda 的 `dexjoco` 环境。

```bash
cat > ~/dexjoco/dex_track_assembly/.env <<'EOF'
export GLI_PATH=/home/wangrenpeng/dexjoco/dex_track_assembly
export WANDB_PROJECT=dex_track_assembly
export WANDB_ENTITY=rpwang
# API key 已在 ~/.netrc，一般不用写；若训练报错再设 WANDB_API_KEY
EOF
```

```bash
cd ~/dexjoco/dex_track_assembly
source .venv/bin/activate
source .env
```

```bash
python -c "import jax; print('devices:', jax.devices())"
python -c "from track_mj.utils.dataset.traj_class import Trajectory; print('track_mj ok')"
```

## 第二步：zarr 特权回放 → ref npz

转换脚本用 **dexjoco conda** 跑仿真回放（不要用本目录 `.venv` 的 python）。

```bash
# 一次性：dexjoco 里装 Trajectory 保存依赖
~/miniconda3/envs/dexjoco/bin/pip install 'flax==0.10.4' 'jax==0.4.38'
```

```bash
conda activate dexjoco
deactivate 2>/dev/null || true   # 若 shell 里还挂着 dex_track .venv，先退出

cd ~/dexjoco/dex_track_assembly
export GLI_PATH=~/dexjoco/dex_track_assembly
export PYTHONPATH=~/dexjoco:~/dexjoco/dexjoco:~/dexjoco/dex_track_assembly
export MUJOCO_GL=egl
```

```bash
# manifest 里 100 条 demo，整段 full 批量导出
~/miniconda3/envs/dexjoco/bin/python scripts/convert_dexjoco_zarr.py \
  --all --segment full --skip-existing

# 可选：按段导出（抓取抬升 / 插孔）
~/miniconda3/envs/dexjoco/bin/python scripts/convert_dexjoco_zarr.py \
  --all --segment grasp_lift --skip-existing
~/miniconda3/envs/dexjoco/bin/python scripts/convert_dexjoco_zarr.py \
  --all --segment insert --skip-existing

# 单条调试
~/miniconda3/envs/dexjoco/bin/python scripts/convert_dexjoco_zarr.py \
  --ep 35 --segment full
```

输出目录：`/mnt/hdd/dexjoco/dex_track_assembly/bimanual_assembly/mocap/PandaBimanual/ep{NNN}_{segment}.npz`

`--segment`：`full` 整段；`grasp_lift` 到 peg 抬升后；`insert` 从 peg 抬升起。两层 tqdm：外层 episode、内层帧回放。

## 第三步：PPO 跟踪训练

用本目录 `.venv`（JAX/MJX），读 HDD 上的 ref npz。

```bash
cd ~/dexjoco/dex_track_assembly
source .venv/bin/activate
source .env
export MUJOCO_GL=egl
```

```bash
# 确认 mujoco==3.3.1（MJX 需要，与 pyproject 一致）
pip install 'mujoco==3.3.1'
```

```bash
# 看哪张卡空闲（显存占用低、无大进程）
nvidia-smi
# 指定 GPU，例如用 2 号卡（debug / 正式训练都建议设）
export CUDA_VISIBLE_DEVICES=2
```

```bash
# 最快：只测 env reset/step（2 env、1 条轨迹，不编 PPO 训练环）
CUDA_VISIBLE_DEVICES=2 python scripts/smoke_assembly_env.py
```

```bash
# smoke 训练：2 env、batch 2×1、unroll 2、1 条轨迹；--num-timesteps 0 只编到 reset、跳过 PPO 环
CUDA_VISIBLE_DEVICES=2 python -m track_mj.learning.train.train_ppo_track \
  --task AssemblyTrackingGeneral \
  --exp_name smoke_assembly_track \
  --num-timesteps 0
```

```bash
# debug 流程冒烟（exp_name 含 debug：16 env、480 步 ≈ 3 次 PPO 更新）
python -m track_mj.learning.train.train_ppo_track \
  --task AssemblyTrackingGeneral \
  --exp_name debug_assembly_track
```

```bash
# 正式训练（512 并行 env，batch 128×4；先 nvidia-smi 选空闲卡）
CUDA_VISIBLE_DEVICES=2 python -m track_mj.learning.train.train_ppo_track \
  --task AssemblyTrackingGeneral \
  --exp_name asm_track_full_v1 \
  --num-timesteps 200000000
```

checkpoint：`/mnt/hdd/dexjoco/dex_track_assembly/checkpoints/{时间戳}_AssemblyTrackingGeneral_{exp_name}/`

训练日志：`/mnt/hdd/dexjoco/outputs/dex_track_assembly/logs/track/{同上}/`
