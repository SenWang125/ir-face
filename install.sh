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
ask()  { read -rp "  $1" "$2"; }

[[ $EUID -eq 0 ]] || die "Run as root: sudo bash install.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== ir-face installer ==="
echo "Install dir : $INSTALL_DIR"
echo "Config dir  : $ETC_DIR"
echo ""

# ── distro detection ──────────────────────────────────────────────────────────
detect_distro() {
    if [[ -f /etc/arch-release ]]; then
        DISTRO=arch
        PAM_AUTH=system-auth
        PAM_LOGIN=system-local-login
        PAM_SESSION=system-login
    elif [[ -f /etc/debian_version ]]; then
        DISTRO=debian
        PAM_AUTH=common-auth
        PAM_LOGIN=common-auth
        PAM_SESSION=common-session
    elif [[ -f /etc/fedora-release ]] || [[ -f /etc/redhat-release ]]; then
        DISTRO=rhel
        PAM_AUTH=password-auth
        PAM_LOGIN=password-auth
        PAM_SESSION=password-auth
    elif [[ -f /etc/SuSE-release ]] || [[ -f /etc/opensuse-release ]]; then
        DISTRO=suse
        PAM_AUTH=common-auth
        PAM_LOGIN=common-auth
        PAM_SESSION=common-session
    else
        # Sniff from /etc/pam.d/sudo as last resort
        if grep -qE "@?include\s+common-auth" /etc/pam.d/sudo 2>/dev/null; then
            DISTRO=debian
            PAM_AUTH=common-auth
            PAM_LOGIN=common-auth
            PAM_SESSION=common-session
        else
            DISTRO=arch
            PAM_AUTH=system-auth
            PAM_LOGIN=system-local-login
            PAM_SESSION=system-login
        fi
    fi
    info "Distro: $DISTRO  (PAM auth include: $PAM_AUTH)"
}

# ── 1. Python ─────────────────────────────────────────────────────────────────
echo "[1/7] Checking Python..."
PY=$(command -v python3) || die "python3 not found"
PY_VER=$($PY -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
[[ "$PY_VER" < "3.8" ]] && die "Python 3.8+ required (found $PY_VER)"
info "Found $PY ($PY_VER)"

detect_distro

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
    info "$f"
done

# Patch INSTALL_DIR into wrappers and install to /usr/local/bin
for w in ir-face ir-compare ir-enroll ir-face-daemon; do
    sed "s|INSTALL_DIR=/opt/ir-face|INSTALL_DIR=$INSTALL_DIR|g" \
        "$SCRIPT_DIR/wrappers/$w" > "$BIN/$w"
    chmod +x "$BIN/$w"
    info "$BIN/$w"
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
    if v4l2-ctl -d "$dev" --info 2>/dev/null | grep -qiE "infrared|ir\b|1:1\.2|face|depth"; then
        IR_DEV=$dev
        break
    fi
done
if [[ -n "$IR_DEV" ]]; then
    sed -i "s|^device = .*|device = $IR_DEV|" "$ETC_DIR/config.ini"
    info "Detected IR camera: $IR_DEV"
else
    info "Could not auto-detect IR camera — set [video] device manually in $ETC_DIR/config.ini"
fi

# ── 6. systemd service ────────────────────────────────────────────────────────
echo "[6/7] Installing systemd service..."

# Only depend on linux-enable-ir-emitter if it is actually installed
AFTER_UNITS="multi-user.target"
WANTS_UNITS=""
if systemctl list-unit-files linux-enable-ir-emitter.service &>/dev/null 2>&1; then
    AFTER_UNITS="multi-user.target linux-enable-ir-emitter.service"
    WANTS_UNITS="Wants=linux-enable-ir-emitter.service"
fi

cat > /etc/systemd/system/ir-face.service << UNIT
[Unit]
Description=IR Face Authentication Daemon
After=$AFTER_UNITS
$WANTS_UNITS

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
echo "    1) Password first — enter password to unlock; face runs as fallback on wrong/empty password (recommended)"
echo "    2) Face first     — face recognition runs before password prompt"
echo "    3) Skip           — configure PAM manually later"
ask "Choice [1/2/3]: " PAM_CHOICE

pam_write_file() {
    local file=$1 mode=$2 auth_inc=$3 acct_inc=$4 sess_inc=$5
    local exec_line="auth  sufficient  pam_exec.so  quiet expose_authtok  $BIN/ir-compare"

    if [[ "$mode" == "face-first" ]]; then
        local line1="$exec_line"
        local line2="auth  sufficient  pam_unix.so  try_first_pass likeauth nullok"
    else
        local line1="auth  sufficient  pam_unix.so  try_first_pass likeauth nullok"
        local line2="$exec_line"
    fi

    cat > "$file" << PAM
#%PAM-1.0
# Managed by ir-face — see /etc/ir-face/config.ini to disable
$line1
$line2
auth     optional   pam_permit.so
auth     required   pam_env.so
auth     include    $auth_inc
account  include    $acct_inc
session  include    $sess_inc
PAM
}

configure_pam_targets() {
    local mode=$1
    # sudo
    pam_write_file /etc/pam.d/sudo  "$mode" "$PAM_AUTH" "$PAM_AUTH" "$PAM_AUTH"
    info "Updated /etc/pam.d/sudo"
    # login (TTY)
    if [[ -f /etc/pam.d/login ]]; then
        pam_write_file /etc/pam.d/login "$mode" "$PAM_LOGIN" "$PAM_LOGIN" "$PAM_SESSION"
        info "Updated /etc/pam.d/login"
    fi
    # su
    if [[ -f /etc/pam.d/su ]]; then
        pam_write_file /etc/pam.d/su "$mode" "$PAM_AUTH" "$PAM_AUTH" "$PAM_AUTH"
        info "Updated /etc/pam.d/su"
    fi
    # polkit — GUI elevation dialogs (KDE, GNOME, etc.) use this
    # /usr/lib/pam.d/polkit-1 is the package default; /etc/pam.d/ overrides it
    pam_write_file /etc/pam.d/polkit-1 "$mode" "$PAM_AUTH" "$PAM_AUTH" "$PAM_AUTH"
    info "Updated /etc/pam.d/polkit-1 (GUI elevation)"
    # Display manager (GUI screen lock)
    configure_dm "$mode"
}

configure_dm() {
    local mode=$1
    local dm=""
    # Detect running display manager
    for candidate in sddm gdm lightdm lxdm; do
        systemctl is-active --quiet "$candidate" 2>/dev/null && { dm=$candidate; break; }
    done
    # Fallback: check for config files
    [[ -z "$dm" && -f /etc/pam.d/sddm ]]    && dm=sddm
    [[ -z "$dm" && -f /etc/pam.d/gdm ]]     && dm=gdm
    [[ -z "$dm" && -f /etc/pam.d/lightdm ]] && dm=lightdm

    case "$dm" in
        sddm)
            # Use system-login on Arch, common-session on Debian
            local dm_sess=$PAM_SESSION
            pam_write_file /etc/pam.d/sddm "$mode" "$PAM_AUTH" "$PAM_AUTH" "$dm_sess"
            info "Updated /etc/pam.d/sddm (display manager: sddm)"
            ;;
        gdm)
            local dm_pam=/etc/pam.d/gdm-password
            [[ -f "$dm_pam" ]] || dm_pam=/etc/pam.d/gdm
            pam_write_file "$dm_pam" "$mode" "$PAM_AUTH" "$PAM_AUTH" "$PAM_SESSION"
            info "Updated $dm_pam (display manager: gdm)"
            ;;
        lightdm)
            pam_write_file /etc/pam.d/lightdm "$mode" "$PAM_AUTH" "$PAM_AUTH" "$PAM_SESSION"
            info "Updated /etc/pam.d/lightdm (display manager: lightdm)"
            ;;
        *)
            info "Display manager not detected — update /etc/pam.d/<dm> manually"
            info "See README for reference PAM stacks"
            ;;
    esac
}

case "$PAM_CHOICE" in
    1)
        configure_pam_targets password-first
        info "PAM: password-first (face as fallback)"
        ;;
    2)
        configure_pam_targets face-first
        info "PAM: face-first"
        ;;
    3)
        info "Skipping PAM — configure /etc/pam.d/{sudo,login,<dm>} manually"
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
echo "  1. Download InsightFace models:"
echo "       sudo bash $SCRIPT_DIR/download-models.sh"
echo "  2. Start daemon:"
echo "       sudo systemctl enable --now ir-face.service"
echo "  3. Enroll your face:"
echo "       sudo ir-face add"
echo "  4. Test recognition:"
echo "       ir-face test"
echo ""
if [[ -z "$IR_DEV" ]]; then
    echo "  !! IR camera not auto-detected."
    echo "     Find yours with:  v4l2-ctl --list-devices"
    echo "     Then set it with: sudo ir-face config device /dev/videoN"
    echo ""
fi
echo "Config: $ETC_DIR/config.ini  (edit with: sudo ir-face config)"
