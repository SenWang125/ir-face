#!/bin/bash
# Download MiniFASNet anti-spoofing ONNX models (V2 scale=2.7 + V1SE scale=4.0).
# Run as root: sudo bash download-antispoof.sh
set -e

DEST=/etc/ir-face/antispoof

[[ $EUID -eq 0 ]] || { echo "Run as root: sudo bash download-antispoof.sh"; exit 1; }

mkdir -p "$DEST"

BASE="https://github.com/yakhyo/face-anti-spoofing/releases/download/weights"

for model in MiniFASNetV2.onnx MiniFASNetV1SE.onnx; do
    if [[ -f "$DEST/$model" ]]; then
        echo "  $model already present — skipping"
        continue
    fi
    echo "  Downloading $model..."
    curl -L --progress-bar -o "$DEST/$model" "$BASE/$model"
done

echo ""
echo "Models saved to $DEST/"
echo "Enable in config: ir-face config antispoof_enabled true"
