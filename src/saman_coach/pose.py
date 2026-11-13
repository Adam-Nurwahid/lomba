"""
Ekstraksi landmark pose MediaPipe dan transformasi koordinat Shoulder-Axis Body Frame.
"""

import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

from .constants import KEY_LANDMARKS, BONE_IDX, MODEL_URL


def unit(v: np.ndarray) -> np.ndarray:
    """Normalisasi array vektor menjadi vektor satuan (unit vector)."""
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def ensure_model(variant: str = "lite", path: Optional[str] = None) -> str:
    """Pastikan model biner MediaPipe Pose Landmarker tersedia di disk."""
    p = Path(path or f"pose_landmarker_{variant}.task")
    if not p.exists():
        url = MODEL_URL.format(v=variant)
        print(f"Mengunduh model MediaPipe '{variant}' -> {p}...")
        try:
            urllib.request.urlretrieve(url, p)
            print(f"[✓] Model '{variant}' berhasil diunduh.")
        except Exception as e:
            sys.exit(f"Gagal mengunduh model ({e}). Unduh manual dari:\n  {url}\nlalu simpan sebagai {p}")
    return str(p)


def extract_landmarks(
    video: str,
    model: str,
    start: float = 0.0,
    end: Optional[float] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Ekstraksi pose landmark frame per frame dari file video menggunakan MediaPipe."""
    # Redam log verbose dari C++/LiteRT
    os.environ["GLOG_minloglevel"] = "2"

    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f"Tidak bisa membuka video: {video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_all = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = n_all / fps if n_all > 0 else None
    meta = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": fps,
        "duration": duration,
    }

    opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )

    frames = []
    total = 0
    i = 0
    last_t = 0.0

    with vision.PoseLandmarker.create_from_options(opts) as landmarker:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            t = i / fps
            i += 1
            last_t = t

            if t < start:
                continue
            if end is not None and t > end:
                break

            total += 1
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect_for_video(mp_image, int(t * 1000))

            if res.pose_landmarks and res.pose_world_landmarks:
                frames.append({
                    "t": t,
                    "lm": [[p.x, p.y, p.z, p.visibility] for p in res.pose_landmarks[0]],
                    "world": [[p.x, p.y, p.z] for p in res.pose_world_landmarks[0]],
                })

    cap.release()
    if meta["duration"] is None:
        meta["duration"] = last_t
    meta["frames_total"] = total
    meta["range"] = [start, end if end is not None else meta["duration"]]
    return frames, meta


def load_frames(cfg: Any, out: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Muat cache jika --reuse aktif dan rentang video sesuai, atau lakukan ekstraksi baru."""
    cache = Path(f"{out}.landmarks.json")
    if getattr(cfg, "reuse", False) and cache.exists():
        try:
            with open(cache, "r", encoding="utf-8") as f:
                d = json.load(f)
            meta = d["meta"]
            rng = meta.get("range")
            if not rng and meta.get("fps") and meta.get("frames_total"):
                meta = dict(meta)
                meta["duration"] = meta["frames_total"] / meta["fps"]
                rng = [0.0, meta["duration"]]

            if rng:
                req_end = cfg.end if cfg.end is not None else meta["duration"]
                if cfg.start >= rng[0] - 1e-6 and req_end <= rng[1] + 1e-6:
                    frames = [f for f in d["frames"] if cfg.start <= f["t"] <= req_end]
                    meta = dict(meta)
                    meta["frames_total"] = max(1, round((req_end - cfg.start) * meta["fps"]))
                    meta["range"] = [cfg.start, req_end]
                    print(f"Cache landmark digunakan: {cache} ({len(frames)} frame, {cfg.start:.1f}s - {req_end:.1f}s)")
                    return frames, meta
        except Exception:
            pass

    model_path = ensure_model(cfg.variant, getattr(cfg, "model", None))
    frames, meta = extract_landmarks(cfg.video, model_path, cfg.start, cfg.end)
    with open(cache, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "frames": frames}, f)
    print(f"Ekstraksi selesai: {len(frames)}/{meta['frames_total']} frame tersimpan di {cache}")
    return frames, meta


def detection_gaps(frames: List[Dict[str, Any]], fps: float, min_gap: float = 1.5) -> List[Tuple[float, float]]:
    """Deteksi celah waktu antar-frame di mana pose hilang / tidak terdeteksi."""
    if not frames:
        return []
    t = np.array([f["t"] for f in frames])
    dt = np.diff(t)
    indices = np.where(dt > min_gap / fps)[0]
    return [(float(t[idx]), float(t[idx + 1])) for idx in indices]


def check_quality(
    frames: List[Dict[str, Any]],
    meta: Dict[str, Any],
    max_sw: float = 0.45
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Evaluasi kualitas deteksi video: visibilitas sendi, batas frame, dan rasio bahu."""
    t = np.array([f["t"] for f in frames])
    LM = np.array([f["lm"] for f in frames])  # Shape: (n, 33, 4)
    key = LM[:, KEY_LANDMARKS]
    vis = key[:, :, 3]

    inframe = ((key[:, :, 0] > -0.02) & (key[:, :, 0] < 1.02) &
               (key[:, :, 1] > -0.02) & (key[:, :, 1] < 1.02)).all(1)

    W, H = meta["width"], meta["height"]
    sw = np.hypot((LM[:, 11, 0] - LM[:, 12, 0]) * W, (LM[:, 11, 1] - LM[:, 12, 1]) * H) / H
    good = (vis.min(1) >= 0.5) & inframe & (sw <= max_sw)

    q = {
        "detectRate": round(len(frames) / max(1, meta["frames_total"]), 3),
        "goodRate": round(float(good.mean()), 3),
        "inFrameRate": round(float(inframe.mean()), 3),
        "visMedian": {
            n: round(float(np.median(vis[:, k])), 2)
            for k, n in enumerate(["L_sh", "R_sh", "L_el", "R_el", "L_wr", "R_wr"])
        },
        "shoulderWidthPctHeight_5_50_95": np.percentile(sw, [5, 50, 95]).round(2).tolist(),
        "warnings": [],
    }

    w = q["warnings"]
    if q["detectRate"] < 0.8:
        w.append(f"Deteksi hanya {q['detectRate']:.0%} frame")
    if min(q["visMedian"].values()) < 0.6:
        w.append("Ada sendi lengan dengan visibilitas median < 0.6")
    if q["inFrameRate"] < 0.9:
        w.append(f"Sendi lengan keluar frame di {1 - q['inFrameRate']:.0%} frame")
    if q["goodRate"] < 0.7:
        w.append(f"Hanya {q['goodRate']:.0%} frame lolos filter kualitas")
    q["usable"] = len(w) == 0

    return q, t, good


def bones_body(frames: List[Dict[str, Any]], use_z: bool = False) -> np.ndarray:
    """
    Transformasi koordinat 3D ke Shoulder-Axis Body Frame:
    - ex: Sumbu garis bahu kanan -> bahu kiri (+X gambar)
    - ey: Sumbu tegak lurus mengarah ke bawah (+Y gambar)
    - ez: Sumbu kedalaman (+Z kedalaman, dinonaktifkan secara default untuk stabilitas)
    """
    Wd = np.array([f["world"] for f in frames], dtype=float)  # (n, 33, 3)
    if not use_z:
        Wd[:, :, 2] = 0.0

    ex = unit(Wd[:, 11] - Wd[:, 12])  # Bahu kanan -> bahu kiri
    down = np.array([0.0, 1.0, 0.0])
    ey = unit(down - (ex @ down)[:, None] * ex)
    ez = np.cross(ex, ey)

    out = np.zeros((len(frames), 4, 3))
    for k, (a, b) in enumerate(BONE_IDX):
        v = unit(Wd[:, b] - Wd[:, a])
        out[:, k, 0] = (v * ex).sum(1)
        out[:, k, 1] = (v * ey).sum(1)
        out[:, k, 2] = (v * ez).sum(1)
    return out
