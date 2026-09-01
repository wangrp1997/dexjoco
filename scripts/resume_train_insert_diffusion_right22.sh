#!/usr/bin/env bash
set -euo pipefail
cd /home/wangrenpeng/dexjoco
tmux kill-session -t train_right22_diff 2>/dev/null || true
tmux new-session -d -s train_right22_diff \
  "bash scripts/train_insert_act_diffusion_right22.sh diffusion resume 2>&1 | tee -a /mnt/hdd/dexjoco/outputs/insert_bc_act_diffusion_right22/_train_logs/diffusion_resume.log"
echo "Started tmux train_right22_diff (resume diffusion -> 50k)"
