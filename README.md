# ir-face

IR face authentication for Linux — PAM-based face unlock using [InsightFace](https://github.com/deepinsight/insightface), similar to Windows Hello. Works with any V4L2-compatible IR camera.

## How it works

A background daemon (`ir-face-daemon`) loads face recognition models once at boot and serves each auth over a Unix socket in ~1–2 s (camera open + 3 matching frames). If the daemon is not running, `ir-compare` falls back to inline CPU model loading (~3–8 s).

**Auth flow (password-first, recommended):**
1. Enter password → correct: done immediately
2. Wrong password or empty field → `pam_exec.so` invokes `ir-compare`
3. IR camera captures frames → cosine similarity against enrolled embeddings
4. 3 consecutive matches above threshold → access granted
5. Timeout → password error returned to the application

**Auth flow (face-first, alternative):**
1. `pam_exec.so` invokes `ir-compare` before any password prompt
2. Face match → access granted; timeout → falls through to password

## Requirements

| Component | Notes |
|---|---|
| Linux | Any distro with PAM and systemd |
| IR camera | V4L2-compatible (built-in on many laptops; USB IR cameras also work) |
| IR emitter driver | Required only if your camera's IR emitter needs activation (see below) |
| Python 3.8+ | |
| NVIDIA GPU | Optional — daemon is CPU-only; GPU used only during enrollment |

## Installation

```bash
git clone <repo-url>
cd ir-face
sudo bash install.sh
```

`install.sh` will:
- Detect your distro and GPU
- Create a Python venv at `/opt/ir-face/venv`
- Install `onnxruntime-gpu` (or `onnxruntime`) + `insightface` + `opencv-python-headless`
- Copy scripts to `/opt/ir-face/` and wrappers to `/usr/local/bin/`
- Set up the systemd daemon service
- Configure PAM interactively for your display manager

Then download InsightFace models:
```bash
sudo bash download-models.sh
```

## Quick start

```bash
# Start the daemon (loads models once, keeps them warm)
sudo systemctl enable --now ir-face.service

# Enroll your face (prompts for profile name)
sudo ir-face add

# Test recognition
ir-face test
```

## Commands

```
ir-face list                        list enrolled profiles for current user
ir-face add                         enroll a new face profile (prompts for name)
ir-face add "Night"                 enroll with a specific profile name
ir-face remove                      remove a profile (prompts from list)
ir-face remove "Night"              remove a named profile
ir-face test                        verbose recognition test
ir-face enable                      enable face authentication
ir-face disable                     disable face authentication (password only)
ir-face config                      open config in $EDITOR
ir-face config show                 print all config values
ir-face config certainty 0.70       set a single config value
```

`add`, `remove`, `enable`, `disable`, and `config` writes require sudo.

## Configuration

`/etc/ir-face/config.ini` — edit directly (`sudo ir-face config`) or via `ir-face config <key> <value>`:

```ini
[core]
enabled = true          # Set false to fall through to password always
certainty = 0.65        # Cosine similarity threshold (0–1, higher = stricter)
required_hits = 3       # Consecutive matching frames required to grant access
det_pack = buffalo_s    # Detector model pack
rec_pack = buffalo_m    # Recognizer — must match the value used during enrollment
max_models = 5          # Max stored profiles per user
end_report = false      # Print timing + similarity info after each auth
antispoof_enabled = false   # Enable MiniFASNet anti-spoof check (run download-antispoof.sh first)
antispoof_threshold = 0.60  # Real-face probability threshold (0–1, lower = more permissive)

[video]
device = /dev/video2    # IR camera V4L2 device node
timeout = 5             # Seconds before giving up
dark_threshold = 8.0    # Frames below this mean brightness are considered dark
dark_max_ratio = 0.8    # Exit dark if this fraction of first 15 frames are dark
clahe_clip = 2.0        # CLAHE contrast enhancement strength
enrollment_frames = 60  # Frames to capture during enrollment

[snapshots]             # Optional — disabled by default
capture_successful = false
capture_failed = false
snapshot_dir = /tmp/ir-face-snapshots
```

**Important:** `rec_pack` must be identical at enrollment and at auth time. Re-enroll after changing it.

## Model packs

| Pack | Detector | Recognizer | Disk | Notes |
|---|---|---|---|---|
| buffalo_s | SCRFD-500M | MobileFaceNet | ~11 MB | Fast load, slightly lower accuracy |
| buffalo_m | SCRFD-2.5G | ArcFace R50 | ~175 MB | Best accuracy/speed balance |
| buffalo_l | SCRFD-10G | ArcFace R50 | ~200 MB | Heavier detector, same recognizer as buffalo_m |

**Recommended:** `det_pack = buffalo_s` + `rec_pack = buffalo_m`

## IR emitter setup

Some cameras have an always-on IR emitter and need no setup. Others (common on laptops) require a driver to activate the emitter.

**Does your camera need it?** Run `ir-face test` — if you see "IR emitter inactive" or very low similarity scores, you need the driver.

### linux-enable-ir-emitter (most laptops)

```bash
# Arch / Manjaro
paru -S linux-enable-ir-emitter
# or: yay -S linux-enable-ir-emitter

# Ubuntu / Debian
sudo apt install linux-enable-ir-emitter
# (may require a PPA or manual build — see project README)

# Fedora
sudo dnf install linux-enable-ir-emitter
```

Configure (interactive — point camera at your face when prompted):
```bash
sudo linux-enable-ir-emitter configure
sudo systemctl enable --now linux-enable-ir-emitter.service
```

If the emitter service is detected at install time, `ir-face.service` will depend on it automatically.

## PAM reference

`install.sh` configures PAM automatically. If you need to edit manually:

### Arch / RHEL / Fedora

```
# /etc/pam.d/sudo  (password-first, face as fallback)
auth  sufficient  pam_unix.so   try_first_pass likeauth nullok
auth  sufficient  pam_exec.so   quiet expose_authtok  /usr/local/bin/ir-compare
auth  optional    pam_permit.so
auth  required    pam_env.so
auth  include     system-auth
account include   system-auth
session include   system-auth
```

### Debian / Ubuntu

Same structure, replace `system-auth` with `common-auth`:

```
auth  sufficient  pam_unix.so   try_first_pass likeauth nullok
auth  sufficient  pam_exec.so   quiet expose_authtok  /usr/local/bin/ir-compare
auth  optional    pam_permit.so
auth  required    pam_env.so
auth  include     common-auth
account include   common-account
session include   common-session
```

Apply to `/etc/pam.d/sudo`, `/etc/pam.d/login`, and your display manager (`sddm`, `gdm-password`, `lightdm`).

**SDDM / screen lock tip:** Leave the password field empty and click Unlock — `pam_unix.so` rejects the empty input, which immediately triggers face recognition. No need to type a wrong password.

## Feedback during auth

- **Terminal (sudo, login):** prints `[ir-face] Authenticating...` and result directly to the TTY
- **GUI screen lock (SDDM, GDM, LightDM):** sends a desktop notification via D-Bus when the user's session is already active
- **Initial login (no active session):** auth completes silently

## Troubleshooting

**Camera not found:**
```bash
v4l2-ctl --list-devices                     # list all cameras
sudo ir-face config device /dev/videoN      # set the correct device
```

**Very low similarity scores (< 0.3):**
- IR emitter not active → check `linux-enable-ir-emitter` status
- `rec_pack` changed since enrollment → re-enroll: `sudo ir-face add`
- Lighting changed significantly → enroll in the same conditions or add another profile

**Face works in `ir-face test` but not in `sudo`:**
```bash
cat /etc/pam.d/sudo
journalctl -u ir-face.service -n 30
```

**Daemon not starting:**
```bash
systemctl status ir-face.service
journalctl -u ir-face.service -n 50
```

**Disable temporarily:**
```bash
sudo ir-face disable    # falls through to password
sudo ir-face enable     # re-enable
```

## Memory usage

The daemon holds models in RAM (CPU-only, no GPU held after startup):

| Config | Steady-state RAM |
|---|---|
| det=buffalo_s + rec=buffalo_m | ~120–150 MB |
| det=buffalo_m + rec=buffalo_m | ~160–180 MB |

## File layout (after install)

```
/opt/ir-face/          Python scripts + venv
/etc/ir-face/
  config.ini           Main config
  models/              Enrolled face embeddings (*.npy per user)
  insightface/models/  Downloaded ONNX model packs
/usr/local/bin/        ir-face  ir-compare  ir-enroll  ir-face-daemon
/etc/systemd/system/ir-face.service
/run/ir-face.sock      Daemon socket (runtime)
```

## Uninstall

```bash
sudo systemctl disable --now ir-face.service
sudo rm -f /etc/systemd/system/ir-face.service
sudo rm -rf /opt/ir-face /etc/ir-face
sudo rm -f /usr/local/bin/ir-face /usr/local/bin/ir-compare \
           /usr/local/bin/ir-enroll /usr/local/bin/ir-face-daemon
# Restore original PAM files for sudo, login, and your display manager
```
