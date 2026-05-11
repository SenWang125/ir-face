# ir-face

IR face authentication for Linux — a PAM-based face unlock using [InsightFace](https://github.com/deepinsight/insightface), similar to Windows Hello. Tested on Lenovo laptops with integrated IR cameras.

## How it works

A background daemon (`ir-face-daemon`) loads the face recognition models once at boot and serves each auth request over a Unix socket in ~1.5s (camera open + 3 matching frames). If the daemon isn't running, `ir-compare` falls back to inline CPU model loading (~3s warm cache).

**PAM flow (password-first, recommended):**
1. Enter password → correct: done
2. Wrong/empty password → `pam_exec.so` invokes `ir-compare`
3. IR camera captures frames → cosine similarity against enrolled embeddings
4. 3 consecutive matches above threshold → access granted; timeout → password error

## Requirements

| Requirement | Notes |
|---|---|
| Linux | Any distro with PAM |
| IR camera | V4L2-compatible, e.g. Lenovo integrated IR |
| IR emitter driver | `linux-enable-ir-emitter` for Lenovo/Asus |
| Python 3.10+ | |
| InsightFace models | Downloaded by `install.sh` |
| NVIDIA GPU | Optional — daemon is CPU-only; GPU used during enrollment only |

## Installation

```bash
git clone https://github.com/yourname/ir-face.git
cd ir-face
sudo bash install.sh
```

`install.sh` will:
- Create a Python venv at `/opt/ir-face/venv`
- Install `onnxruntime-gpu` (or `onnxruntime`) + `insightface` + `opencv-python-headless`
- Install scripts to `/opt/ir-face/` and wrappers to `/usr/local/bin/`
- Set up the systemd daemon service
- Configure PAM interactively (password-first or face-first)

Then download InsightFace models:
```bash
sudo bash download-models.sh
```

## Quick start

```bash
# Start the daemon (loads models at boot, keeps them warm)
sudo systemctl enable --now ir-face.service

# Enroll your face
sudo ir-face add

# Test recognition
ir-face test
```

## Usage

```bash
ir-face list                     # Show enrolled models for current user
sudo ir-face add                 # Enroll default profile
sudo ir-face add Night           # Enroll additional profile (e.g. low-light)
sudo ir-face remove Night        # Delete a profile
ir-face test                     # Verbose recognition test
ir-face config                   # Show all config values
ir-face config certainty 0.70    # Update a config value
```

## Configuration

`/etc/ir-face/config.ini` — edit directly or via `ir-face config`:

```ini
[core]
certainty = 0.65        # Cosine similarity threshold (0–1, higher = stricter)
required_hits = 3       # Consecutive matching frames required
det_pack = buffalo_s    # Detector model pack
rec_pack = buffalo_m    # Recognizer model pack  ← must match enrollment
max_models = 5          # Max stored profiles per user
end_report = false      # Print timing + similarity after each auth

[video]
device = /dev/video2    # IR camera V4L2 device
timeout = 5             # Seconds before giving up
dark_threshold = 8.0    # Frames below this brightness are considered dark
clahe_clip = 2.0        # CLAHE contrast enhancement strength
enrollment_frames = 60  # Frames captured per enrollment session

[snapshots]
capture_successful = false
capture_failed = false
snapshot_dir = /tmp/ir-face-snapshots
```

**Important:** `rec_pack` must be the same value at enrollment and at auth time. If you change it, re-enroll.

## Model packs

| Pack | Detector | Recognizer | Disk | Use case |
|---|---|---|---|---|
| buffalo_s | SCRFD-500M | MobileFaceNet | ~11 MB | Fast loading, slightly less accurate |
| buffalo_m | SCRFD-2.5G | ArcFace R50 | ~175 MB | Best accuracy/speed balance |
| buffalo_l | SCRFD-10G | ArcFace R50 | ~200 MB | Same recognizer as buffalo_m |

**Recommended:** `det_pack = buffalo_s` + `rec_pack = buffalo_m`
(tiny detector sufficient for close frontal IR; ArcFace R50 recognizer)

## IR emitter setup (Lenovo / Asus)

```bash
# Arch Linux
paru -S linux-enable-ir-emitter

# Configure (interactive, point at camera when prompted)
sudo linux-enable-ir-emitter configure

# Enable at boot
sudo systemctl enable --now linux-enable-ir-emitter.service
```

## PAM reference

Password-first (mirrors Windows Hello — face tries only after wrong password):
```
auth  sufficient  pam_unix.so   try_first_pass likeauth nullok
auth  sufficient  pam_exec.so   quiet expose_authtok  /usr/local/bin/ir-compare
auth  optional    pam_permit.so
auth  required    pam_env.so
auth  include     system-auth
```

Face-first:
```
auth  sufficient  pam_exec.so   quiet expose_authtok  /usr/local/bin/ir-compare
auth  sufficient  pam_unix.so   try_first_pass likeauth nullok
...
```

## Feedback during auth

- **Terminal (sudo, login):** prints `Checking face...` and `Face recognized.` / `Face not recognized.` directly to the TTY
- **GUI unlock (SDDM screen lock, etc.):** sends a desktop notification via D-Bus if the user's session is active
- **SDDM initial login:** no in-band feedback (PAM conversation not supported by `pam_exec.so`); auth completes silently

## Troubleshooting

**Camera not found:**
```bash
v4l2-ctl --list-devices          # List cameras
ir-face config device /dev/video4  # Update config
```
Also check the camera privacy switch (physical key on Lenovo laptops).

**Very low similarity (< 0.3):**
- IR emitter not active: `sudo linux-enable-ir-emitter run`
- `rec_pack` mismatch — re-enroll after changing it

**Face works in `ir-face test` but not in sudo:**
- Check: `cat /etc/pam.d/sudo`
- Check: `journalctl -u ir-face.service -n 30`

**Daemon not starting:**
```bash
systemctl status ir-face.service
journalctl -u ir-face.service -n 50
```

## Memory usage

The daemon holds models in RAM (CPU-only, no GPU):

| Config | Steady-state RAM |
|---|---|
| det=buffalo_s, rec=buffalo_m | ~120–150 MB |
| det=buffalo_m, rec=buffalo_m | ~160–180 MB |

## File layout (after install)

```
/opt/ir-face/          Python scripts + venv
/etc/ir-face/          Config, models, InsightFace packs
  config.ini
  models/              Enrolled face embeddings (*.npy)
  insightface/models/  ONNX model files
/usr/local/bin/        Wrappers: ir-face, ir-compare, ir-enroll, ir-face-daemon
/etc/systemd/system/ir-face.service
/run/ir-face.sock      Daemon socket (runtime)
```
