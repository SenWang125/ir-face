#!/bin/bash
# ir-face install script
# Run as root: sudo bash install.sh
set -e

INSTALL_DIR=/opt/ir-face
ETC_DIR=/etc/ir-face
VENV=$INSTALL_DIR/venv
BIN=/usr/local/bin

# ── helpers ───────────────────────────────────────────────────────────────────
die()  { echo "ERROR: $*" >&2; exit 1; }
info() { echo "  $*"; }

[[ $EUID -eq 0 ]] || die "Run as root: sudo bash install.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== ir-face installer ==="
echo "Install dir : $INSTALL_DIR"
echo "Config dir  : $ETC_DIR"
echo ""

# ── 1. Python ─────────────────────────────────────────────────────────────────
echo "[1/7] Checking Python..."
PY=$(command -v python3) || die "python3 not found"
PY_VER=$($PY -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
info "Found $PY ($PY_VER)"

# ── 2. Install dir + venv ─────────────────────────────────────────────────────
echo "[2/7] Setting up venv at $VENV..."
mkdir -p "$INSTALL_DIR"
if [[ ! -d "$VENV" ]]; then
    $PY -m venv "$VENV"
    info "Created venv"
else
    info "Venv already exists"
fi

# ── 3. Python packages ────────────────────────────────────────────────────────
echo "[3/7] Installing Python packages..."

# Detect NVIDIA GPU
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    ORT_PKG=onnxruntime-gpu
    info "NVIDIA GPU detected — installing onnxruntime-gpu"
else
    ORT_PKG=onnxruntime
    info "No GPU detected — installing onnxruntime (CPU)"
fi

"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet "$ORT_PKG" insightface opencv-python-headless numpy

# ── 4. Copy scripts ───────────────────────────────────────────────────────────
echo "[4/7] Installing scripts to $INSTALL_DIR..."
for f in ir_compare.py ir_enroll.py ir_face_daemon.py ir_face_cli.py; do
    cp "$SCRIPT_DIR/$f" "$INSTALL_DIR/$f"
    info "  $f"
done

# Patch INSTALL_DIR into wrappers and install to /usr/local/bin
for w in ir-face ir-compare ir-enroll ir-face-daemon; do
    sed "s|INSTALL_DIR=/opt/ir-face|INSTALL_DIR=$INSTALL_DIR|g" \
        "$SCRIPT_DIR/wrappers/$w" > "$BIN/$w"
    chmod +x "$BIN/$w"
    info "  $BIN/$w"
done

# ── 5. /etc/ir-face structure ─────────────────────────────────────────────────
echo "[5/7] Setting up $ETC_DIR..."
mkdir -p "$ETC_DIR/models" "$ETC_DIR/insightface/models"

if [[ ! -f "$ETC_DIR/config.ini" ]]; then
    cp "$SCRIPT_DIR/config.ini.example" "$ETC_DIR/config.ini"
    info "Wrote default config.ini"
else
    info "config.ini already exists — not overwritten"
fi

# Detect IR camera device
IR_DEV=""
for dev in /dev/video*; do
    if v4l2-ctl -d "$dev" --info 2>/dev/null | grep -qi "infrared\|ir\|1:1.2"; then
        IR_DEV=$dev
        break
    fi
done
if [[ -n "$IR_DEV" ]]; then
    sed -i "s|^device = .*|device = $IR_DEV|" "$ETC_DIR/config.ini"
    info "Detected IR camera: $IR_DEV"
else
    info "Could not auto-detect IR camera — set [video] device manually in config.ini"
fi

# ── 6. systemd service ────────────────────────────────────────────────────────
echo "[6/7] Installing systemd service..."
cat > /etc/systemd/system/ir-face.service << UNIT
[Unit]
Description=IR Face Authentication Daemon
After=multi-user.target linux-enable-ir-emitter.service
Wants=linux-enable-ir-emitter.service

[Service]
Type=simple
ExecStart=$BIN/ir-face-daemon
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
info "Service installed"

# ── 7. PAM ────────────────────────────────────────────────────────────────────
echo "[7/7] PAM configuration..."
echo ""
echo "  Choose PAM auth order:"
echo "    1) Password first  — face unlock only after wrong password (like howdy)"
echo "    2) Face first      — face unlock attempted before password prompt"
echo "    3) Skip            — configure PAM manually later"
read -rp "  Choice [1/2/3]: " PAM_CHOICE

pam_password_first() {
    local file=$1 include=$2
    cat > "$file" << PAM
#%PAM-1.0
auth  sufficient  pam_unix.so   try_first_pass likeauth nullok
auth  sufficient  pam_exec.so   quiet expose_authtok  $BIN/ir-compare
auth  optional    pam_permit.so
auth  required    pam_env.so
auth  include     $include
account include   $include
session include   $include
PAM
}

pam_face_first() {
    local file=$1 include=$2
    cat > "$file" << PAM
#%PAM-1.0
auth  sufficient  pam_exec.so   quiet expose_authtok  $BIN/ir-compare
auth  sufficient  pam_unix.so   try_first_pass likeauth nullok
auth  optional    pam_permit.so
auth  required    pam_env.so
auth  include     $include
account include   $include
session include   $include
PAM
}

case "$PAM_CHOICE" in
    1)
        pam_password_first /etc/pam.d/sudo   system-auth
        pam_password_first /etc/pam.d/login  system-local-login
        info "PAM: password-first applied to sudo + login"
        # sddm
        cat > /etc/pam.d/sddm << 'PAM'
#%PAM-1.0
auth      sufficient    pam_unix.so       try_first_pass nullok
auth      sufficient    pam_exec.so       quiet expose_authtok  /usr/local/bin/ir-compare
auth      optional      pam_permit.so
auth      required      pam_env.so
auth      include       system-login
-auth     optional      pam_gnome_keyring.so
-auth     optional      pam_kwallet5.so
account   include       system-login
password  include       system-login
-password optional      pam_gnome_keyring.so use_authtok
session   optional      pam_keyinit.so force revoke
session   include       system-login
-session  optional      pam_gnome_keyring.so auto_start
-session  optional      pam_kwallet5.so auto_start
PAM
        info "PAM: password-first applied to sddm"
        ;;
    2)
        pam_face_first /etc/pam.d/sudo   system-auth
        pam_face_first /etc/pam.d/login  system-local-login
        info "PAM: face-first applied to sudo + login"
        ;;
    3)
        info "Skipping PAM — configure /etc/pam.d/{sudo,login,sddm} manually"
        ;;
    *)
        info "Invalid choice — skipping PAM"
        ;;
esac

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "=== Installation complete ==="
echo ""
echo "Next steps:"
echo "  1. Start daemon:   sudo systemctl enable --now ir-face.service"
echo "  2. Enroll face:    sudo ir-face add"
echo "  3. Test:           ir-face test"
echo ""
echo "Config: $ETC_DIR/config.ini"
echo "  - Set [video] device to your IR camera (e.g. /dev/video2)"
echo "  - Adjust [core] det_pack / rec_pack (buffalo_s / buffalo_m / buffalo_l)"
echo ""
echo "Download InsightFace models if not present:"
echo "  python3 -c \"import insightface; insightface.app.FaceAnalysis(name='buffalo_m', root='$ETC_DIR/insightface').prepare(ctx_id=-1)\""
