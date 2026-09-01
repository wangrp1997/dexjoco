#!/usr/bin/env bash
# Parallel handoff insert eval: ACT (GPU2) + Diffusion (GPU1), 20 demos each.
set -euo pipefail

REPO=/home/wangrenpeng/dexjoco
OUT=/mnt/hdd/dexjoco/outputs/insert_bc_handoff_eval
EPISODES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19"
ACT_CKPT=/mnt/hdd/dexjoco/outputs/insert_bc_act_diffusion/act_steps50000_bs4_seed0/checkpoints/050000/pretrained_model
DIFF_CKPT=/mnt/hdd/dexjoco/outputs/insert_bc_act_diffusion/diffusion_steps50000_bs4_seed0/checkpoints/050000/pretrained_model
CONFIG=$REPO/configs/rand_obj/bimanual_assembly.yaml
PY_DEX=/home/wangrenpeng/miniconda3/envs/dexjoco/bin/python
SERVE_LEROBOT=/home/wangrenpeng/miniconda3/envs/lerobot/bin/dexjoco-lerobot-serve

source /home/wangrenpeng/miniconda3/etc/profile.d/conda.sh
mkdir -p "$OUT"

cat > "$OUT/run_act_eval.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source /home/wangrenpeng/miniconda3/etc/profile.d/conda.sh
export PYTHONPATH=$REPO/dexjoco:$REPO:$REPO/scripts
export MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
conda activate lerobot
CUDA_VISIBLE_DEVICES=2 $SERVE_LEROBOT --host 127.0.0.1 --port 8091 \\
  > "$OUT/act_policy_server.log" 2>&1 &
SPID=\$!
sleep 12
conda activate dexjoco
CUDA_VISIBLE_DEVICES=2 $PY_DEX -u $REPO/scripts/eval_lerobot_demo_handoff_insert.py \\
  --config $CONFIG \\
  --checkpoint $ACT_CKPT \\
  --policy-type act \\
  --episodes $EPISODES \\
  --seed 0 \\
  --host 127.0.0.1 \\
  --port 8091 \\
  --policy-device cuda:0 \\
  --output $OUT/act_ckpt050000_seed0 \\
  --overwrite \\
  2>&1 | tee "$OUT/act_eval.log"
kill \$SPID 2>/dev/null || true
wait \$SPID 2>/dev/null || true
EOF

cat > "$OUT/run_diffusion_eval.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source /home/wangrenpeng/miniconda3/etc/profile.d/conda.sh
export PYTHONPATH=$REPO/dexjoco:$REPO:$REPO/scripts
export MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
conda activate lerobot
CUDA_VISIBLE_DEVICES=1 $SERVE_LEROBOT --host 127.0.0.1 --port 8092 \\
  > "$OUT/diffusion_policy_server.log" 2>&1 &
SPID=\$!
sleep 12
conda activate dexjoco
CUDA_VISIBLE_DEVICES=1 $PY_DEX -u $REPO/scripts/eval_lerobot_demo_handoff_insert.py \\
  --config $CONFIG \\
  --checkpoint $DIFF_CKPT \\
  --policy-type diffusion \\
  --episodes $EPISODES \\
  --seed 0 \\
  --host 127.0.0.1 \\
  --port 8092 \\
  --policy-device cuda:0 \\
  --output $OUT/diffusion_ckpt050000_seed0 \\
  --overwrite \\
  2>&1 | tee "$OUT/diffusion_eval.log"
kill \$SPID 2>/dev/null || true
wait \$SPID 2>/dev/null || true
EOF

chmod +x "$OUT/run_act_eval.sh" "$OUT/run_diffusion_eval.sh"

tmux kill-session -t eval_insert_act 2>/dev/null || true
tmux kill-session -t eval_insert_diff 2>/dev/null || true
tmux new-session -d -s eval_insert_act "$OUT/run_act_eval.sh"
tmux new-session -d -s eval_insert_diff "$OUT/run_diffusion_eval.sh"

echo "Started tmux eval_insert_act (GPU2:8091) and eval_insert_diff (GPU1:8092)"
echo "Logs: $OUT/act_eval.log  $OUT/diffusion_eval.log"
