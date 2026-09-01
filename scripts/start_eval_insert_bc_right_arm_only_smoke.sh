#!/usr/bin/env bash
set -euo pipefail
REPO=/home/wangrenpeng/dexjoco
OUT=/mnt/hdd/dexjoco/outputs/insert_bc_handoff_eval_rightonly_smoke
EPISODES="0,1,2,3,4"
ACT_CKPT=/mnt/hdd/dexjoco/outputs/insert_bc_act_diffusion/act_steps50000_bs4_seed0/checkpoints/050000/pretrained_model
DIFF_CKPT=/mnt/hdd/dexjoco/outputs/insert_bc_act_diffusion/diffusion_steps50000_bs4_seed0/checkpoints/050000/pretrained_model
CONFIG=$REPO/configs/rand_obj/bimanual_assembly.yaml
PY_DEX=/home/wangrenpeng/miniconda3/envs/dexjoco/bin/python
SERVE=/home/wangrenpeng/miniconda3/envs/lerobot/bin/dexjoco-lerobot-serve
source /home/wangrenpeng/miniconda3/etc/profile.d/conda.sh
mkdir -p "$OUT"

cat > "$OUT/run_act.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source /home/wangrenpeng/miniconda3/etc/profile.d/conda.sh
export PYTHONPATH=$REPO/dexjoco:$REPO:$REPO/scripts MUJOCO_GL=egl HF_HUB_OFFLINE=1
conda activate lerobot
CUDA_VISIBLE_DEVICES=2 $SERVE --host 127.0.0.1 --port 8093 > "$OUT/act_server.log" 2>&1 &
SPID=\$!
sleep 12
conda activate dexjoco
CUDA_VISIBLE_DEVICES=2 $PY_DEX -u $REPO/scripts/eval_lerobot_demo_handoff_insert.py \\
  --config $CONFIG --checkpoint $ACT_CKPT --policy-type act \\
  --episodes $EPISODES --seed 0 --host 127.0.0.1 --port 8093 --policy-device cuda:0 \\
  --output $OUT/act_ckpt050000_seed0 --overwrite --right-arm-only \\
  2>&1 | tee "$OUT/act_eval.log"
kill \$SPID 2>/dev/null || true
EOF

cat > "$OUT/run_diff.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source /home/wangrenpeng/miniconda3/etc/profile.d/conda.sh
export PYTHONPATH=$REPO/dexjoco:$REPO:$REPO/scripts MUJOCO_GL=egl HF_HUB_OFFLINE=1
conda activate lerobot
CUDA_VISIBLE_DEVICES=1 $SERVE --host 127.0.0.1 --port 8094 > "$OUT/diff_server.log" 2>&1 &
SPID=\$!
sleep 12
conda activate dexjoco
CUDA_VISIBLE_DEVICES=1 $PY_DEX -u $REPO/scripts/eval_lerobot_demo_handoff_insert.py \\
  --config $CONFIG --checkpoint $DIFF_CKPT --policy-type diffusion \\
  --episodes $EPISODES --seed 0 --host 127.0.0.1 --port 8094 --policy-device cuda:0 \\
  --output $OUT/diffusion_ckpt050000_seed0 --overwrite --right-arm-only \\
  2>&1 | tee "$OUT/diff_eval.log"
kill \$SPID 2>/dev/null || true
EOF

chmod +x "$OUT/run_act.sh" "$OUT/run_diff.sh"
tmux kill-session -t eval_ronly_act 2>/dev/null || true
tmux kill-session -t eval_ronly_diff 2>/dev/null || true
tmux new-session -d -s eval_ronly_act "$OUT/run_act.sh"
tmux new-session -d -s eval_ronly_diff "$OUT/run_diff.sh"
echo "Started eval_ronly_act + eval_ronly_diff -> $OUT"
