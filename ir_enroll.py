#!/usr/bin/env python3
"""Enroll a user's face using the IR camera.

Usage:
  ir_enroll.py <username> [label]

  username  - system username to enroll
  label     - optional label for this model (default: "Default")
"""

import sys
import os
import time
import warnings
import configparser
import cv2
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

CONFIG_PATH     = "/etc/ir-face/config.ini"
MODEL_DIR       = "/etc/ir-face/models"
INSIGHTFACE_DIR = "/etc/ir-face/insightface/models"


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
    d = os.path.join(INSIGHTFACE_DIR, pack)
    for f in sorted(os.listdir(d)):
        if f.startswith(prefix) and f.endswith(".onnx"):
            return os.path.join(d, f)
    return None


def load_models(det_pack, rec_pack, providers):
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


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <username> [label]")
        sys.exit(1)

    username = sys.argv[1]
    label    = sys.argv[2] if len(sys.argv) > 2 else "Default"

    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)

    model_name        = config.get(    "core",  "recognition_model",  fallback="buffalo_m")
    det_pack          = config.get(    "core",  "det_pack",            fallback=model_name)
    rec_pack          = config.get(    "core",  "rec_pack",            fallback=model_name)
    max_models        = config.getint( "core",  "max_models",          fallback=5)
    device            = config.get(    "video", "device",              fallback="/dev/video2")
    frames_to_capture = config.getint( "video", "enrollment_frames",   fallback=60)
    frame_interval    = config.getfloat("video", "enrollment_interval", fallback=0.08)
    clahe_clip        = config.getfloat("video", "clahe_clip",          fallback=2.0)
    min_good_frames   = max(20, frames_to_capture // 3)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"{username}.npy")

    existing = []
    if os.path.exists(model_path):
        data = np.load(model_path, allow_pickle=True).item()
        existing = data.get("models", [])
        labels = [m["label"] for m in existing]
        print(f"Existing models for '{username}': {labels}")
        existing = [m for m in existing if m["label"] != label]
        if len(existing) >= max_models:
            print(f"Maximum of {max_models} models reached. Remove one first:")
            for m in existing:
                print(f"  ir-face remove {m['label']}")
            sys.exit(1)

    print(f"Loading models (det={det_pack}, rec={rec_pack})...")
    _patch_ort()

    # GPU is worthwhile for enrollment: CUDA init (~3.5s) is amortized across 60+ frames
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved2     = os.dup(2)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)
    devnull    = open(os.devnull, "w")
    old_stdout, sys.stdout = sys.stdout, devnull

    t0 = time.time()
    det, rec, norm_crop = load_models(det_pack, rec_pack, providers)
    load_ms = int((time.time() - t0) * 1000)

    sys.stdout = old_stdout
    devnull.close()
    os.dup2(saved2, 2)
    os.close(saved2)

    print(f"Models loaded in {load_ms}ms")

    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))

    print(f"Opening IR camera ({device})...")
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {device}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    headless = not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if headless:
        print(f"\nEnrolling '{label}' for user '{username}' (headless mode).")
    else:
        print(f"\nEnrolling '{label}' for user '{username}'.")
    print(f"Look directly at the camera. Collecting {frames_to_capture} frames..."
          + ("" if headless else " (Q to cancel)") + "\n")

    embeddings     = []
    no_face_streak = 0
    last_accepted  = 0.0

    while len(embeddings) < frames_to_capture:
        ret, frame = cap.read()
        if not ret:
            continue

        now       = time.time()
        gray      = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        enhanced  = clahe.apply(gray)
        frame_3ch = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        bboxes, kpss = det.detect(frame_3ch, max_num=1)
        has_face = bboxes is not None and len(bboxes) > 0 and kpss is not None

        if has_face and (now - last_accepted) >= frame_interval:
            aimg = norm_crop(frame_3ch, landmark=kpss[0], image_size=112)
            feat = rec.get_feat(aimg).flatten()
            emb  = feat / np.linalg.norm(feat)
            embeddings.append(emb)
            last_accepted  = now
            no_face_streak = 0
        elif not has_face:
            no_face_streak += 1
            if no_face_streak == 30:
                print("\n  [!] No face detected — move closer or check IR emitter")
                no_face_streak = 0

        eta = max(0, (frames_to_capture - len(embeddings)) * frame_interval)
        sys.stdout.write(f"\r  Captured: {len(embeddings):3d}/{frames_to_capture}  ETA: {eta:.1f}s")
        sys.stdout.flush()

        if not headless:
            if has_face:
                preview = frame_3ch.copy()
                x1, y1, x2, y2 = [int(v) for v in bboxes[0][:4]]
                color = (0, 255, 0) if (now - last_accepted) < frame_interval else (0, 200, 255)
                cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
                cv2.putText(preview, f"{label}: {len(embeddings)}/{frames_to_capture}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                preview = frame_3ch.copy()
                cv2.putText(preview, "No face detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("IR Enrollment", preview)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nCancelled.")
                cap.release()
                cv2.destroyAllWindows()
                sys.exit(1)

    cap.release()
    if not headless:
        cv2.destroyAllWindows()
    print()

    if len(embeddings) < min_good_frames:
        print(f"\nOnly got {len(embeddings)} usable frames (need {min_good_frames}).")
        print("Make sure the IR emitter is active and you are in range.")
        sys.exit(1)

    emb_array = np.array(embeddings)

    # Filter outliers: discard frames whose embedding deviates far from the group mean.
    mean_emb = np.mean(emb_array, axis=0)
    mean_emb /= np.linalg.norm(mean_emb)
    sims = emb_array @ mean_emb
    keep = emb_array[sims >= 0.60]
    if len(keep) >= min_good_frames:
        dropped = len(emb_array) - len(keep)
        if dropped:
            print(f"Filtered {dropped} low-quality frames (kept {len(keep)})")
        emb_array = keep
        mean_emb  = np.mean(emb_array, axis=0)
        mean_emb /= np.linalg.norm(mean_emb)

    existing.append({
        "label":       label,
        "embeddings":  emb_array,
        "mean":        mean_emb,
        "created":     time.time(),
        "frame_count": len(emb_array),
        "device":      device,
        "det_pack":    det_pack,
        "rec_pack":    rec_pack,
    })

    payload = {
        "username":   username,
        "models":     existing,
        "embeddings": np.vstack([m["embeddings"] for m in existing]),
    }
    np.save(model_path, payload, allow_pickle=True)

    print(f"Enrolled {len(emb_array)} frames as '{label}' for '{username}'")
    print(f"Total models for '{username}': {[m['label'] for m in existing]}")
    print(f"Saved -> {model_path}")


if __name__ == "__main__":
    main()
