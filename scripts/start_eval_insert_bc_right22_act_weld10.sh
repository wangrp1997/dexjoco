#!/usr/bin/env bash
# Right22 ACT 50k handoff eval (10 ep): tray weld left + freeze left mocap.
set -euo pipefail

REPO=/home/wangrenpeng/dexjoco
OUT=/mnt/hdd/dexjoco/outputs/insert_bc_handoff_eval_right22_act_weld10
EPISODES="0,1,2,3,4,5,6,7,8,9"
ACT_CKPT=/mnt/hdd/dexjoco/outputs/insert_bc_act_diffusion_right22/act_steps50000_bs4_seed0/checkpoints/050000/pretrained_model
CONFIG=$REPO/configs/rand_obj/bimanual_assembly.yaml
PY_DEX=/home/wangrenpeng/miniconda3/envs/dexjoco/bin/python
SERVE=/home/wangrenpeng/miniconda3/envs/lerobot/bin/dexjoco-lerobot-serve

source /home/wangrenpeng/miniconda3/etc/profile.d/conda.sh
mkdir -p "$OUT"

cat > "$OUT/run_act.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source /home/wangrenpeng/miniconda3/etc/profile.d/conda.sh
export PYTHONPATH=$REPO/dexjoco:$REPO:$REPO/scripts MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
conda activate lerobot
CUDA_VISIBLE_DEVICES=2 $SERVE --host 127.0.0.1 --port 8097 > "$OUT/act_server.log" 2>&1 &
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
  --port 8097 \\
  --policy-device cuda:0 \\
  --output $OUT/act_ckpt050000_seed0 \\
  --overwrite \\
  --no-hybrid-insert \\
  --tray-weld-left \\
  --left-grip-scale 1.12 \\
  2>&1 | tee "$OUT/act_eval.log"
kill \$SPID 2>/dev/null || true
EOF

chmod +x "$OUT/run_act.sh"
tmux kill-session -t eval_right22_act_weld10 2>/dev/null || true
tmux new-session -d -s eval_right22_act_weld10 "$OUT/run_act.sh"
echo "Started tmux eval_right22_act_weld10 -> $OUT"
