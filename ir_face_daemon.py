#!/usr/bin/env python3
"""IR face recognition daemon — loads models once, serves auth over Unix socket.

Protocol:
  client → "<username>\n"               normal:  single byte exit code
  client → "<username> --scored\n"      scored:  one "DONE" line with sim score
  client → "<username> --verbose\n"     verbose: streamed frame lines + "DONE" line

DONE line format:
  "DONE <code> cam=<ms> recog=<ms> sim=<float> frames=<n> dark=<n> profile=<url-encoded>\n"
"""

import os
import sys
import socket
import threading
import signal
import time
import warnings
import configparser
from urllib.parse import quote
import cv2
import numpy as np

warnings.filterwarnings("ignore")

SOCKET_PATH     = "/run/ir-face.sock"
CONFIG_PATH     = "/etc/ir-face/config.ini"
MODEL_DIR       = "/etc/ir-face/models"
INSIGHTFACE_DIR = "/etc/ir-face/insightface/models"

EXIT_OK       = 0
EXIT_NO_MODEL = 10
EXIT_TIMEOUT  = 11
EXIT_NO_USER  = 12
EXIT_TOO_DARK = 13

camera_lock = threading.Lock()
_warm_cap   = None   # camera pre-opened during model loading, used for first auth


def _open_camera(device):
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def _prewarm_camera(device):
    """Open the camera in a background thread while models are loading.
    Stores the handle in _warm_cap so the first auth session can reuse it."""
    global _warm_cap
    cap = _open_camera(device)
    with camera_lock:
        _warm_cap = cap  # None if open failed — recognize() will retry


def _patch_ort_cpu():
    """Use ORT_ENABLE_BASIC to reduce resident memory by ~30-40% vs the default
    ORT_ENABLE_ALL, at the cost of slightly slower per-frame inference.
    Acceptable tradeoff for a daemon serving 3-frame auth sessions."""
    import onnxruntime as ort
    _orig = ort.InferenceSession.__init__
    def _init(self, model, sess_options=None, providers=None, provider_options=None, **kw):
        if sess_options is None:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            )
            sess_options.intra_op_num_threads = 4
            sess_options.inter_op_num_threads = 2
        _orig(self, model, sess_options, providers, provider_options, **kw)
    ort.InferenceSession.__init__ = _init


def _find_model(pack, prefix):
    d = os.path.join(INSIGHTFACE_DIR, pack)
    for f in sorted(os.listdir(d)):
        if f.startswith(prefix) and f.endswith(".onnx"):
            return os.path.join(d, f)
    return None


def load_models(det_pack, rec_pack):
    from insightface.model_zoo import model_zoo
    from insightface.utils.face_align import norm_crop

    det_path = _find_model(det_pack, "det_")
    rec_path = _find_model(rec_pack, "w600k")
    if not det_path or not rec_path:
        raise FileNotFoundError(
            f"Cannot find models — det in '{det_pack}', rec in '{rec_pack}'"
        )

    providers = ["CPUExecutionProvider"]

    det = model_zoo.get_model(det_path, providers=providers)
    det.prepare(ctx_id=0, input_size=(640, 640), det_thresh=0.5)

    rec = model_zoo.get_model(rec_path, providers=providers)
    rec.prepare(ctx_id=0)

    return det, rec, norm_crop


def emit(stream, msg):
    if stream:
        try:
            stream.sendall((msg + "\n").encode())
        except Exception:
            pass


def recognize(det, rec, norm_crop, username, stream=None):
    model_path = os.path.join(MODEL_DIR, f"{username}.npy")
    if not os.path.exists(model_path):
        return EXIT_NO_MODEL, {}

    data       = np.load(model_path, allow_pickle=True).item()
    model_list = data.get("models", [])
    profiles   = [(m["label"], m["embeddings"]) for m in model_list] if model_list \
                 else [("Default", data["embeddings"])]

    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    threshold     = config.getfloat("core",  "certainty",       fallback=0.65)
    required_hits = config.getint( "core",  "required_hits",    fallback=3)
    device        = config.get(   "video", "device",            fallback="/dev/video2")
    timeout       = config.getint("video", "timeout",           fallback=5)
    dark_thresh   = config.getfloat("video", "dark_threshold",  fallback=8.0)
    dark_ratio    = config.getfloat("video", "dark_max_ratio",  fallback=0.8)
    clahe_clip    = config.getfloat("video", "clahe_clip",      fallback=2.0)

    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))

    with camera_lock:
        global _warm_cap
        t_cam = time.time()
        if _warm_cap is not None and _warm_cap.isOpened():
            cap       = _warm_cap
            _warm_cap = None          # consume; next auth opens fresh
        else:
            cap = _open_camera(device)
        if cap is None:
            return EXIT_TIMEOUT, {}
        cam_ms = int((time.time() - t_cam) * 1000)

        total      = 0
        dark       = 0
        hits       = 0
        best_sim   = 0.0
        best_label = profiles[0][0]
        t_recog    = time.time()
        deadline   = t_recog + timeout
        result     = EXIT_TIMEOUT
        info       = {}

        try:
            while time.time() < deadline:
                ret, frame = cap.read()
                if not ret:
                    continue
                total += 1

                gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if np.mean(gray) < dark_thresh:
                    dark += 1
                    if total >= 15 and dark / total > dark_ratio:
                        result = EXIT_TOO_DARK
                        break
                    hits = 0
                    continue

                enhanced  = clahe.apply(gray)
                frame_3ch = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

                bboxes, kpss = det.detect(frame_3ch, max_num=1)
                if bboxes is None or len(bboxes) == 0 or kpss is None:
                    emit(stream, f"frame {total}: no face detected")
                    hits = 0
                    continue

                aimg = norm_crop(frame_3ch, landmark=kpss[0], image_size=112)
                feat = rec.get_feat(aimg).flatten()
                emb  = feat / np.linalg.norm(feat)

                frame_sim, frame_label = max(
                    ((max(float(np.dot(emb, s)) for s in embs), label)
                     for label, embs in profiles),
                    key=lambda x: x[0],
                )
                if frame_sim > best_sim:
                    best_sim   = frame_sim
                    best_label = frame_label
                sim = frame_sim

                if sim >= threshold:
                    hits += 1
                    emit(stream, f"frame {total}: sim={sim:.4f} ({frame_label}) — hit {hits}/{required_hits}")
                    if hits >= required_hits:
                        result = EXIT_OK
                        break
                else:
                    emit(stream, f"frame {total}: sim={sim:.4f} below threshold, reset")
                    hits = 0
        finally:
            cap.release()   # always close — camera held only during the auth session

    recog_ms = int((time.time() - t_recog) * 1000)
    info = {"cam": cam_ms, "recog": recog_ms, "sim": best_sim,
            "frames": total, "dark": dark, "profile": best_label}
    return result, info


def handle_client(conn, det, rec, norm_crop):
    try:
        buf = b""
        while b"\n" not in buf:
            chunk = conn.recv(128)
            if not chunk:
                return
            buf += chunk
        parts    = buf.split(b"\n")[0].decode().strip().split()
        username = parts[0] if parts else ""
        verbose  = "--verbose" in parts
        scored   = "--scored"  in parts

        if not username:
            conn.sendall(b"DONE 12\n" if (verbose or scored) else bytes([EXIT_NO_USER]))
            return

        cfg = configparser.ConfigParser()
        cfg.read(CONFIG_PATH)
        if not cfg.getboolean("core", "enabled", fallback=True):
            conn.sendall(b"DONE 10\n" if (verbose or scored) else bytes([EXIT_NO_MODEL]))
            return

        # scored: no per-frame stream, just the final DONE line
        stream = conn if verbose else None
        result, info = recognize(det, rec, norm_crop, username, stream)

        if verbose or scored:
            cam     = info.get("cam", 0)
            recog   = info.get("recog", 0)
            sim     = info.get("sim", 0.0)
            frames  = info.get("frames", 0)
            dark    = info.get("dark", 0)
            profile = quote(info.get("profile", ""), safe="")
            conn.sendall(
                f"DONE {result} cam={cam} recog={recog} sim={sim:.4f} frames={frames} dark={dark} profile={profile}\n"
                .encode()
            )
        else:
            conn.sendall(bytes([result]))
    except Exception as e:
        print(f"[ir-face] handler error: {e}", file=sys.stderr)
        try:
            conn.sendall(b"DONE 11 sim=0.0000 profile=\n" if (b"--verbose" in buf or b"--scored" in buf) else bytes([EXIT_TIMEOUT]))
        except Exception:
            pass
    finally:
        conn.close()


def main():
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    model_name = config.get("core", "recognition_model", fallback="buffalo_m")
    det_pack   = config.get("core", "det_pack",          fallback=model_name)
    rec_pack   = config.get("core", "rec_pack",          fallback=model_name)

    device = config.get("video", "device", fallback="/dev/video2")

    print(f"[ir-face] Loading det={det_pack} rec={rec_pack} (CPU)...", file=sys.stderr, flush=True)
    t0 = time.time()

    _patch_ort_cpu()

    # Open the camera in the background while models load — amortises V4L2 init
    threading.Thread(target=_prewarm_camera, args=(device,), daemon=True).start()

    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved2     = os.dup(2)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)
    devnull    = open(os.devnull, "w")
    old_stdout, sys.stdout = sys.stdout, devnull

    det, rec, norm_crop = load_models(det_pack, rec_pack)

    sys.stdout = old_stdout
    devnull.close()
    os.dup2(saved2, 2)
    os.close(saved2)

    print(f"[ir-face] Ready in {time.time()-t0:.1f}s (CPU-only, no GPU held)", file=sys.stderr, flush=True)

    try:
        os.unlink(SOCKET_PATH)
    except FileNotFoundError:
        pass

    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCKET_PATH)
    os.chmod(SOCKET_PATH, 0o666)
    srv.listen(4)

    def _shutdown(sig, frame):
        if _warm_cap is not None:
            _warm_cap.release()
        srv.close()
        try:
            os.unlink(SOCKET_PATH)
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    while True:
        try:
            conn, _ = srv.accept()
        except OSError:
            break
        threading.Thread(
            target=handle_client, args=(conn, det, rec, norm_crop), daemon=True
        ).start()


if __name__ == "__main__":
    main()
