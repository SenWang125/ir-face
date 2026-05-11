#!/bin/bash
# Download InsightFace model packs to /etc/ir-face/insightface/models/
# Run as root: sudo bash download-models.sh
set -e

INSTALL_DIR=/opt/ir-face
VENV=$INSTALL_DIR/venv
MODELS_DIR=/etc/ir-face/insightface

[[ $EUID -eq 0 ]] || { echo "Run as root: sudo bash download-models.sh"; exit 1; }
[[ -d "$VENV" ]] || { echo "Run install.sh first."; exit 1; }

echo "Downloading InsightFace model packs..."
echo "Models will be saved to: $MODELS_DIR/models/"
echo ""

"$VENV/bin/python3" - << 'PY'
import os, sys
sys.stdout.reconfigure(line_buffering=True)

MODELS_DIR = "/etc/ir-face/insightface"
os.makedirs(MODELS_DIR, exist_ok=True)

from insightface.app import FaceAnalysis

packs = ["buffalo_s", "buffalo_m"]
for pack in packs:
    print(f"  Downloading {pack}...", flush=True)
    try:
        app = FaceAnalysis(name=pack, root=MODELS_DIR,
                           allowed_modules=["detection", "recognition"])
        app.prepare(ctx_id=-1)
        print(f"  {pack}: OK")
    except Exception as e:
        print(f"  {pack}: FAILED — {e}")

print()
print("Done. Models at:", os.path.join(MODELS_DIR, "models"))
PY
