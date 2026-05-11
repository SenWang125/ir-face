#!/usr/bin/env python3
"""IR face recognition — PAM-compatible exit codes, matching howdy convention.

Exit codes:
  0  - authenticated
  10 - no model for this user
  11 - timeout / no match
  12 - no username provided
  13 - all frames too dark (IR emitter not active?)

Fast path: connects to ir-face daemon socket if running (models pre-loaded).
Direct path: loads only det + rec ONNX models inline (skips landmarks/genderage).
"""

import sys
import os
import time
import warnings
import configparser
import socket as _socket
import cv2
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

timings = {"st": time.time()}

EXIT_OK       = 0
EXIT_NO_MODEL = 10
EXIT_TIMEOUT  = 11
EXIT_NO_USER  = 12
EXIT_TOO_DARK = 13

CONFIG_PATH     = "/etc/ir-face/config.ini"
MODEL_DIR       = "/etc/ir-face/models"
INSIGHTFACE_DIR = "/etc/ir-face/insightface/models"
SOCKET_PATH     = "/run/ir-face.sock"

VERBOSE = "--verbose" in sys.argv or "-v" in sys.argv


def log(msg):
    if VERBOSE:
        print(msg, file=sys.stderr)


def _tty_print(msg):
    """Write directly to the user's terminal — visible even under pam_exec quiet."""
    try:
        with open("/dev/tty", "w") as tty:
            print(msg, file=tty, flush=True)
    except Exception:
        pass


def _gui_notify(summary, body):
    """Send a desktop notification to the authenticating user's session.

    Works for GUI screen unlock (D-Bus session already running).
    Silently does nothing for initial login (no session yet).
    """
    pam_user = os.environ.get("PAM_USER", "")
    if not pam_user:
        return
    try:
        import pwd, subprocess
        uid = pwd.getpwnam(pam_user).pw_uid
        bus = f"/run/user/{uid}/bus"
        if not os.path.exists(bus):
            return
        subprocess.run(
            ["sudo", "-u", pam_user,
             "env", f"DBUS_SESSION_BUS_ADDRESS=unix:path={bus}",
             "notify-send", "-t", "3000", summary, body],
            timeout=2, capture_output=True,
        )
    except Exception:
        pass


def _pam_feedback(msg, summary=None):
    """Deliver auth feedback via TTY and desktop notification."""
    _tty_print(f"\n[ir-face] {msg}")
    _gui_notify(summary or "ir-face", f"[ir-face] {msg}")


def save_snapshot(frame, label, username, elapsed):
    snap_dir = config.get("snapshots", "snapshot_dir", fallback="/tmp/ir-face-snapshots")
    os.makedirs(snap_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(snap_dir, f"{ts}_{label}_{username}.png")
    cv2.imwrite(path, frame)
    log(f"Snapshot saved: {path}")


def _patch_ort():
    """Switch cuDNN conv algo search EXHAUSTIVE → HEURISTIC to cut ~2s off model load."""
    import onnxruntime as ort
    _orig = ort.InferenceSession.__init__
    def _fast_init(self, model, sess_options=None, providers=None, provider_options=None, **kw):
        if providers and not provider_options:
            provider_options = [
                {"cudnn_conv_algo_search": "HEURISTIC"}
                if (p if isinstance(p, str) else p[0]) == "CUDAExecutionProvider"
                else {}
                for p in providers
            ]
        _orig(self, model, sess_options, providers, provider_options, **kw)
    ort.InferenceSession.__init__ = _fast_init


def _find_model(pack, prefix):
    """Find the first ONNX file in the model pack whose name starts with prefix."""
    d = os.path.join(INSIGHTFACE_DIR, pack)
    for f in sorted(os.listdir(d)):
        if f.startswith(prefix) and f.endswith(".onnx"):
            return os.path.join(d, f)
    return None


def load_models(det_pack, rec_pack, providers):
    """Load detection + recognition from (potentially different) model packs."""
    from insightface.model_zoo import model_zoo
    from insightface.utils.face_align import norm_crop

    det_path = _find_model(det_pack, "det_")
    rec_path = _find_model(rec_pack, "w600k")
    if not det_path or not rec_path:
        raise FileNotFoundError(
            f"Cannot find models — det in '{det_pack}', rec in '{rec_pack}'"
        )

    det = model_zoo.get_model(det_path, providers=providers)
    det.prepare(ctx_id=0, input_size=(640, 640), det_thresh=0.5)

    rec = model_zoo.get_model(rec_path, providers=providers)
    rec.prepare(ctx_id=0)

    return det, rec, norm_crop


def try_daemon_normal(username):
    try:
        sock = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        sock.settimeout(8)
        sock.connect(SOCKET_PATH)
        sock.sendall(username.encode() + b"\n")
        result = sock.recv(1)
        sock.close()
        return result[0] if result else None
    except Exception:
        return None


def try_daemon_verbose(username):
    try:
        sock = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        sock.settimeout(10)
        sock.connect(SOCKET_PATH)
        sock.sendall(f"{username} --verbose\n".encode())
        buf = b""
        while True:
            chunk = sock.recv(256)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.decode()
                if line.startswith("DONE "):
                    parts = line.split()
                    code  = int(parts[1])
                    info  = dict(p.split("=") for p in parts[2:] if "=" in p)
                    sock.close()
                    if config.getboolean("core", "end_report", fallback=False):
                        threshold = config.getfloat("core", "certainty", fallback=0.65)
                        print(f"\n  Open camera:    {info.get('cam','?')}ms",  file=sys.stderr)
                        print(f"  Recognition:    {info.get('recog','?')}ms", file=sys.stderr)
                        print(f"  Frames:         {info.get('frames','?')}",  file=sys.stderr)
                        print(f"  Best sim:       {float(info.get('sim',0)):.4f} (threshold {threshold})",
                              file=sys.stderr)
                    return code
                print(line, file=sys.stderr)
        sock.close()
    except Exception:
        pass
    return None


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not args:
        sys.exit(EXIT_NO_USER)

    username = args[0]
    config.read(CONFIG_PATH)

    if VERBOSE:
        print("[core]", file=sys.stderr)
        for k, v in config.items("core"):
            print(f"  {k} = {v}", file=sys.stderr)
        print(file=sys.stderr)
    else:
        _pam_feedback("Checking face...", "IR Face Auth")

    # Try daemon first (models already loaded — fast path)
    if VERBOSE:
        result = try_daemon_verbose(username)
    else:
        result = try_daemon_normal(username)

    if result is not None:
        if not VERBOSE:
            if result == EXIT_OK:
                _pam_feedback("Face recognized.", "IR Face Auth")
            elif result == EXIT_TIMEOUT:
                _pam_feedback("Face not recognized.", "IR Face Auth")
            elif result == EXIT_TOO_DARK:
                _pam_feedback("Face: IR emitter not active.", "IR Face Auth")
        sys.exit(result)

    # Direct path — load models inline
    log("Loading models directly...")

    model_path = os.path.join(MODEL_DIR, f"{username}.npy")
    if not os.path.exists(model_path):
        log(f"No model found at {model_path}")
        sys.exit(EXIT_NO_MODEL)

    data   = np.load(model_path, allow_pickle=True).item()
    stored = data["embeddings"]
    log(f"Loaded {len(stored)} enrolled embeddings for '{username}'")

    threshold     = config.getfloat("core",  "certainty",           fallback=0.65)
    required_hits = config.getint( "core",  "required_hits",        fallback=3)
    model_name    = config.get(   "core",  "recognition_model",     fallback="buffalo_m")
    det_pack      = config.get(   "core",  "det_pack",               fallback=model_name)
    rec_pack      = config.get(   "core",  "rec_pack",               fallback=model_name)
    end_report    = config.getboolean("core", "end_report",         fallback=False)
    device        = config.get(   "video", "device",                fallback="/dev/video2")
    timeout       = config.getint("video", "timeout",               fallback=5)
    dark_thresh   = config.getfloat("video", "dark_threshold",      fallback=8.0)
    dark_ratio    = config.getfloat("video", "dark_max_ratio",      fallback=0.8)
    clahe_clip    = config.getfloat("video", "clahe_clip",          fallback=2.0)
    cap_ok        = config.getboolean("snapshots", "capture_successful", fallback=False)
    cap_fail      = config.getboolean("snapshots", "capture_failed",     fallback=False)

    log(f"det={det_pack}  rec={rec_pack}")

    # CPU-only for direct path: CUDA context init costs ~3.5s per process,
    # which exceeds GPU inference savings for just a few frames.
    providers = ["CPUExecutionProvider"]

    _devnull_fd = os.open(os.devnull, os.O_WRONLY)
    _saved_fd2  = os.dup(2)
    os.dup2(_devnull_fd, 2)
    os.close(_devnull_fd)
    _devnull = open(os.devnull, "w")
    _old_stdout, sys.stdout = sys.stdout, _devnull

    timings["ll"] = time.time()
    det, rec, norm_crop = load_models(det_pack, rec_pack, providers)
    timings["ll"] = time.time() - timings["ll"]

    sys.stdout = _old_stdout
    _devnull.close()
    os.dup2(_saved_fd2, 2)
    os.close(_saved_fd2)

    log(f"Model loaded in {timings['ll']*1000:.0f}ms")

    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))

    timings["ic"] = time.time()
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        log(f"Cannot open {device}")
        sys.exit(EXIT_TIMEOUT)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    timings["ic"] = time.time() - timings["ic"]

    timings["fr"] = time.time()
    total     = 0
    dark      = 0
    best_sim  = 0.0
    hits      = 0
    snap_frame = None

    while time.time() - timings["fr"] < timeout:
        ret, frame = cap.read()
        if not ret:
            continue
        total += 1

        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if np.mean(gray) < dark_thresh:
            dark += 1
            if total >= 15 and dark / total > dark_ratio:
                cap.release()
                log("Too many dark frames — is the IR emitter running?")
                if not VERBOSE:
                    _pam_feedback("Face: IR emitter not active.", "IR Face Auth")
                sys.exit(EXIT_TOO_DARK)
            hits = 0
            continue

        enhanced  = clahe.apply(gray)
        frame_3ch = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        if snap_frame is None:
            snap_frame = frame_3ch

        bboxes, kpss = det.detect(frame_3ch, max_num=1)
        if bboxes is None or len(bboxes) == 0 or kpss is None:
            log(f"frame {total}: no face detected")
            hits = 0
            continue

        aimg = norm_crop(frame_3ch, landmark=kpss[0], image_size=112)
        feat = rec.get_feat(aimg).flatten()
        emb  = feat / np.linalg.norm(feat)
        sim  = max(float(np.dot(emb, s)) for s in stored)

        if sim > best_sim:
            best_sim   = sim
            snap_frame = frame_3ch

        if sim >= threshold:
            hits += 1
            log(f"frame {total}: sim={sim:.4f} — hit {hits}/{required_hits}")
            if hits >= required_hits:
                timings["tt"] = time.time() - timings["st"]
                timings["fl"] = time.time() - timings["fr"]
                cap.release()
                if cap_ok:
                    save_snapshot(snap_frame, "SUCCESS", username, timings["fl"])
                if end_report:
                    print(f"\nTime spent")
                    print(f"  Load model:     {timings['ll']*1000:.0f}ms")
                    print(f"  Open camera:    {timings['ic']*1000:.0f}ms")
                    print(f"  Recognition:    {timings['fl']*1000:.0f}ms")
                    print(f"  Total:          {timings['tt']*1000:.0f}ms")
                    print(f"\nFrames searched: {total} ({total/timings['fl']:.1f} fps)")
                    print(f"Dark frames:     {dark}")
                    print(f"Best similarity: {best_sim:.4f} (threshold {threshold})")
                if not VERBOSE:
                    _pam_feedback("Face recognized.", "IR Face Auth")
                sys.exit(EXIT_OK)
        else:
            hits = 0
            log(f"frame {total}: sim={sim:.4f} below threshold, reset")

    cap.release()
    if cap_fail and snap_frame is not None:
        save_snapshot(snap_frame, "FAILED", username, timeout)
    log(f"Timeout. Best similarity: {best_sim:.4f}, consecutive hits: {hits}/{required_hits}")
    if not VERBOSE:
        _pam_feedback("Face not recognized.", "IR Face Auth")
    sys.exit(EXIT_TIMEOUT)


config = configparser.ConfigParser()

if __name__ == "__main__":
    timings["in"] = time.time() - timings["st"]
    main()
