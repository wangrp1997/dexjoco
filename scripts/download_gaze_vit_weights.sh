#!/usr/bin/env bash
# One-time: cache timm ViT-S/16 weights via HF mirror (shows progress).
set -euo pipefail
source /home/wangrenpeng/miniconda3/etc/profile.d/conda.sh
conda activate dexjoco
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export PYTHONPATH=/home/wangrenpeng/dexjoco

# 国内镜像（官方 hub 慢/为 0 时用）
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

nros-proxy-on 2>/dev/null || true

REPO="timm/vit_small_patch16_224.augreg_in21k_ft_in1k"
FILE="pytorch_model.bin"

echo "HF_ENDPOINT=$HF_ENDPOINT"
echo "==> downloading $REPO / $FILE (~86MB)"

if command -v hf >/dev/null 2>&1; then
  hf download "$REPO" "$FILE"
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "$REPO" "$FILE"
else
  python - <<'PY'
import os
from huggingface_hub import hf_hub_download

endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
print("hub endpoint:", endpoint)
path = hf_hub_download(
    repo_id="timm/vit_small_patch16_224.augreg_in21k_ft_in1k",
    filename="pytorch_model.bin",
    endpoint=endpoint,
)
print("saved:", path)
PY
fi

echo "==> verify timm load"
python - <<'PY'
import timm
m = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=0)
print("timm ok, embed_dim =", m.embed_dim)
PY

echo "done."
