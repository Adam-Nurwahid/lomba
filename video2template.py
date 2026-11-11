"""
video2template.py v0.2.1.1  (default = mode "tanpa z", yang terbukti paling bagus)

Video tari -> template gerakan (pose unik + timeline note) + laporan kualitas.

Pakai:
    python video2template.py videoku.mp4
    python video2template.py videoku.mp4 --start 5 --end 17
    python video2template.py videoku.mp4 --reuse            # pakai cache landmark
    python video2template.py videoku.mp4 --variant full     # model lebih akurat (coba & bandingkan)
    python video2template.py videoku.mp4 --use-z            # aktifkan kedalaman (tidak disarankan)

Install:
    pip install mediapipe opencv-python numpy scipy matplotlib

Perubahan dari v0.1:
  - z dimatikan secara default (--use-z untuk menyalakan). --no-z tetap diterima (no-op).
  - Model MediaPipe diunduh otomatis kalau belum ada (--variant lite|full|heavy).
  - --reuse sekarang menghormati --start/--end (dulu cache lama dipakai apa adanya).
  - Cache menyimpan rentang & durasi, jadi kalau rentang baru tidak tercakup -> ekstraksi ulang.
  - Output JSON mencatat variant model & parameter yang dipakai (reproducible).
  - Pesan saran konkret saat video belum layak (usable = False).
  - Membaca cache v0.1 (videoku_landmarks.json) tanpa ekstraksi ulang.
  - Laporan celah deteksi (dari angles.py) dan --snapshots / contact sheet (dari snapshots.py).

Fitur utama = ARAH TULANG lengan (vektor satuan) dalam bingkai badan dari garis bahu.
Tidak memakai pinggul, tidak bergantung ukuran badan / jarak kamera.
Catatan: bingkai ini meniadakan putaran badan terhadap kamera, bukan kemiringan kamera.
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, savgol_filter

FPS = 30
KEY = [11, 12, 13, 14, 15, 16]  # bahu, siku, pergelangan
BONE_NAMES = ["L_upper", "L_fore", "R_upper", "R_fore"]
BONE_IDX = [(11, 13), (13, 15), (12, 14), (14, 16)]
ACCENT_MAX_SPEED = 100  # deg/s
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task")


# ----------------------------------------------------------------- ekstraksi
def ensure_model(variant, path=None):
    p = Path(path or f"pose_landmarker_{variant}.task")
    if not p.exists():
        url = MODEL_URL.format(v=variant)
        print(f"Mengunduh model {variant} -> {p}")
        try:
            urllib.request.urlretrieve(url, p)
        except Exception as e:  # noqa: BLE001
            sys.exit(f"Gagal unduh model ({e}). Unduh manual dari:\n  {url}\nlalu simpan sebagai {p}")
    return str(p)


def extract(video, model, start=0.0, end=None):
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        sys.exit(f"Tidak bisa membuka video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n_all = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = n_all / fps if n_all > 0 else None
    meta = {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": fps, "duration": duration}

    opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model),
        running_mode=vision.RunningMode.VIDEO, num_poses=1)
    frames, total, i, last_t = [], 0, 0, 0.0
    with vision.PoseLandmarker.create_from_options(opts) as lm:
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
            res = lm.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(t * 1000))
            if res.pose_landmarks:
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


def load_frames(cfg, out):
    """Pakai cache bila --reuse dan rentangnya mencukupi; kalau tidak, ekstraksi ulang."""
    cache = Path(f"{out}.landmarks.json")
    if cfg.reuse and cache.exists():
        d = json.load(open(cache))
        meta = d["meta"]
        rng = meta.get("range")
        if not rng and meta.get("fps") and meta.get("frames_total"):
            # cache v0.1: tidak menyimpan rentang. Asumsi: video penuh dari detik 0 (trim default).
            meta = dict(meta)
            meta["duration"] = meta["frames_total"] / meta["fps"]
            rng = [0.0, meta["duration"]]
            print("cache v0.1 terdeteksi -> dianggap video penuh (0 s - akhir)")
        if rng:
            req_end = cfg.end if cfg.end is not None else meta["duration"]
            if cfg.start >= rng[0] - 1e-6 and req_end <= rng[1] + 1e-6:
                frames = [f for f in d["frames"] if cfg.start <= f["t"] <= req_end]
                meta = dict(meta)
                meta["frames_total"] = max(1, round((req_end - cfg.start) * meta["fps"]))
                meta["range"] = [cfg.start, req_end]
                print(f"cache dipakai: {cache} ({len(frames)} frame, {cfg.start:.1f}-{req_end:.1f}s)")
                return frames, meta
            print("cache tidak mencakup rentang yang diminta -> ekstraksi ulang")
        else:
            print("cache tidak valid (tanpa meta fps/frames_total) -> ekstraksi ulang")
    frames, meta = extract(cfg.video, ensure_model(cfg.variant, cfg.model), cfg.start, cfg.end)
    json.dump({"meta": meta, "frames": frames}, open(cache, "w"))
    print(f"ekstraksi: {len(frames)}/{meta['frames_total']} frame terdeteksi -> {cache}")
    return frames, meta


def detection_gaps(frames, fps, min_gap=1.5):
    """Celah waktu antar frame terdeteksi (detik) yang lebih panjang dari min_gap frame."""
    t = np.array([f["t"] for f in frames])
    dt = np.diff(t)
    return [(float(t[i]), float(t[i + 1])) for i in np.where(dt > min_gap / fps)[0]]


# ------------------------------------------------------------------- kualitas
def check_quality(frames, meta, max_sw):
    t = np.array([f["t"] for f in frames])
    LM = np.array([f["lm"] for f in frames])  # (n, 33, 4)
    key = LM[:, KEY]
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
        "visMedian": {n: round(float(np.median(vis[:, k])), 2)
                      for k, n in enumerate(["L_sh", "R_sh", "L_el", "R_el", "L_wr", "R_wr"])},
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


# -------------------------------------------------------------------- fitur
def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def bones_body(frames, use_z):
    Wd = np.array([f["world"] for f in frames], dtype=float)  # (n, 33, 3)
    if not use_z:
        Wd[:, :, 2] = 0
    ex = unit(Wd[:, 11] - Wd[:, 12])  # bahu kanan -> kiri ~ +x gambar
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


def dist(A, B):
    """Rata-rata sudut (derajat) antar-tulang dua pose (4,3)."""
    return float(np.degrees(np.arccos(np.clip((A * B).sum(1), -1, 1))).mean())


def runs(mask, grid, min_len):
    m = np.r_[mask, False]
    out, start = [], None
    for i, v in enumerate(m):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if grid[i - 1] - grid[start] >= min_len:
                out.append((start, i - 1))
            start = None
    return out


def seg_pose(B, s, e):
    return unit(np.median(B[s:e + 1], axis=0))


# ------------------------------------------------------------------- proses
def process(frames, meta, cfg):
    q, t, good = check_quality(frames, meta, cfg.max_shoulder)
    if good.sum() < 15:
        return {"quality": q, "error": "Terlalu sedikit frame berkualitas baik"}

    Bg = bones_body([f for f, g in zip(frames, good) if g], cfg.use_z)
    tg = t[good]
    grid = np.arange(tg[0], tg[-1], 1 / FPS)
    B = np.stack([np.stack([np.interp(grid, tg, Bg[:, k, c]) for c in range(3)], 1)
                  for k in range(4)], 1)  # (n, 4, 3)
    B = unit(savgol_filter(B, 15, 2, axis=0))

    # tandai titik grid yang jauh dari sampel bagus (hasil interpolasi di celah) = tidak dipercaya
    idx = np.searchsorted(tg, grid)
    near = np.minimum(np.abs(grid - tg[np.clip(idx - 1, 0, len(tg) - 1)]),
                      np.abs(grid - tg[np.clip(idx, 0, len(tg) - 1)]))
    bad = near > 0.1
    k = int(0.2 * FPS)
    bad = np.convolve(bad.astype(float), np.ones(2 * k + 1), "same") > 0

    ang = np.degrees(np.arccos(np.clip((B[:-1] * B[1:]).sum(-1), -1, 1)))  # (n-1, 4)
    speed = ang.sum(1) * FPS
    speed = np.r_[speed[0], speed]

    segs = runs((speed < cfg.thresh) & ~bad, grid, cfg.min_hold)
    merged = []
    for s, e in segs:
        if merged:
            ps, pe = merged[-1]
            if grid[s] - grid[pe] <= 0.5 and dist(seg_pose(B, s, e), seg_pose(B, ps, pe)) <= 20:
                merged[-1] = (ps, e)
                continue
        merged.append((s, e))
    segs = merged

    pk, _ = find_peaks(-speed, distance=int(0.4 * FPS), prominence=80)
    acc = [i for i in pk if speed[i] < ACCENT_MAX_SPEED and not bad[i]
           and not any(grid[s] - 0.5 <= grid[i] <= grid[e] + 0.5 for s, e in segs)]

    notes = [{"type": "hold", "tStart": float(grid[s]), "tEnd": float(grid[e]),
              "bones": seg_pose(B, s, e)} for s, e in segs]
    notes += [{"type": "accent", "t": float(grid[i]), "bones": B[i]} for i in acc]
    notes.sort(key=lambda n: n.get("tStart", n.get("t")))

    # klaster pose: pose baru jika jaraknya > cfg.cluster dari semua centroid
    cent, members, assign = [], [], []
    for n in notes:
        best, bd = None, cfg.cluster
        for j, c in enumerate(cent):
            d = dist(n["bones"], c)
            if d <= bd:
                best, bd = j, d
        if best is None:
            cent.append(n["bones"].copy())
            members.append([n])
            assign.append(len(cent) - 1)
        else:
            members[best].append(n)
            cent[best] = unit(np.mean([m["bones"] for m in members[best]], axis=0))
            assign.append(best)

    poses = [{"id": f"P{j + 1}",
              "bones": {nm: np.round(cent[j][i], 4).tolist() for i, nm in enumerate(BONE_NAMES)},
              "spreadDeg": (round(float(np.mean([dist(m["bones"], cent[j]) for m in ms])), 1)
                            if len(ms) > 1 else None),  # 1 sampel = belum ada ukuran sebaran
              "count": len(ms)} for j, ms in enumerate(members)]

    timeline = []
    for n, a in zip(notes, assign):
        e = {"type": n["type"], "pose": f"P{a + 1}"}
        if n["type"] == "hold":
            e["tStart"], e["tEnd"] = round(n["tStart"], 2), round(n["tEnd"], 2)
        else:
            e["t"] = round(n["t"], 2)
        timeline.append(e)

    return {"quality": q, "poses": poses, "timeline": timeline, "centroids": cent,
            "grid": grid, "speed": speed, "bad": bad, "segs": segs, "acc": acc}


# --------------------------------------------------------------------- gambar
def draw_stick(ax, bones, lw=3):
    import matplotlib.pyplot as plt
    UP, FO = 0.8, 0.7  # satuan lebar bahu

    def arm(sx, bu, bf):
        s = np.array([sx, 0.0])
        e = s + UP * bu[:2]
        w = e + FO * bf[:2]
        return s, e, w

    Ls, Le, Lw = arm(0.5, bones[0], bones[1])
    Rs, Re, Rw = arm(-0.5, bones[2], bones[3])
    hl, hr = np.array([0.35, 1.7]), np.array([-0.35, 1.7])
    for a, b in [(Rs, Ls), (Ls, Le), (Le, Lw), (Rs, Re), (Re, Rw), (Ls, hl), (Rs, hr), (hr, hl),
                 (np.array([0, 0.0]), np.array([0, -0.25]))]:
        ax.plot([a[0], b[0]], [a[1], b[1]], "k-", lw=lw, solid_capstyle="round")
    ax.add_patch(plt.Circle((0, -0.55), 0.3, fill=False, lw=lw))
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(2.0, -1.7)
    ax.set_aspect("equal")
    ax.axis("off")


def save_plots(res, out, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    poses, cent = res["poses"], res["centroids"]
    cols = min(5, len(poses))
    rows = int(np.ceil(len(poses) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(2.6 * cols, 3.2 * rows), squeeze=False)
    for ax in axs.flat:
        ax.axis("off")
    for ax, p, c in zip(axs.flat, poses, cent):
        draw_stick(ax, c)
        sp_txt = f", ±{p['spreadDeg']:.0f}°" if p["spreadDeg"] is not None else ""
        ax.set_title(f"{p['id']} (x{p['count']}{sp_txt})", fontsize=9)
    plt.tight_layout()
    fig.savefig(f"{out}_poses.png", dpi=110)

    g, sp = res["grid"], res["speed"]
    fig2, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(g, sp, color="gray", lw=1)
    for s, e in res["segs"]:
        ax.axvspan(g[s], g[e], alpha=0.25, color="green")
    for i in np.where(res["bad"])[0]:
        ax.axvspan(g[i] - 0.5 / FPS, g[i] + 0.5 / FPS, alpha=0.15, color="red", lw=0)
    for e in res["timeline"]:
        x = e["tStart"] if e["type"] == "hold" else e["t"]
        if e["type"] == "accent":
            ax.axvline(x, color="k", ls="--", lw=0.8)
            ax.text(x, ax.get_ylim()[1] * 0.95, e["pose"], fontsize=8, va="top")
    ax.set_xlabel("detik")
    ax.set_ylabel("kecepatan tulang (°/s)")
    plt.tight_layout()
    fig2.savefig(f"{out}_timeline.png", dpi=110)
    if show:
        plt.show()


def save_contact_sheet(video, timeline, out, cols=4):
    """Satu gambar berisi frame video asli di tiap note -> cek visual apakah pose terbaca benar."""
    import cv2
    cap = cv2.VideoCapture(video)
    tiles = []
    for e in timeline:
        t = (e["tStart"] + e["tEnd"]) / 2 if e["type"] == "hold" else e["t"]
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (320, int(320 * h / w)))
        cv2.putText(frame, f"{e['pose']} {e['type']} @{t:.2f}s", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        tiles.append(frame)
    cap.release()
    if not tiles:
        print("contact sheet: tidak ada frame yang bisa dibaca")
        return
    H, W = tiles[0].shape[:2]
    while len(tiles) % cols:
        tiles.append(np.zeros((H, W, 3), np.uint8))
    sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(f"{out}_contact.png", sheet)
    print(f"{out}_contact.png tersimpan ({len(timeline)} note)")


# ----------------------------------------------------------------------- main
def build_parser():
    ap = argparse.ArgumentParser(description="Video tari -> template gerakan")
    ap.add_argument("video")
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default=None, help="path file .task (default: unduh otomatis sesuai --variant)")
    ap.add_argument("--variant", choices=["lite", "full", "heavy"], default="lite",
                    help="lite = sama dgn hasil tanpa_z; full/heavy lebih akurat tapi lebih lambat")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--use-z", action="store_true", help="pakai kedalaman z (default: mati)")
    ap.add_argument("--no-z", action="store_true", help="(deprecated) z sudah mati secara default")
    ap.add_argument("--thresh", type=float, default=120)
    ap.add_argument("--min-hold", type=float, default=0.4)
    ap.add_argument("--cluster", type=float, default=18)
    ap.add_argument("--max-shoulder", type=float, default=0.45,
                    help="lebar bahu maks (fraksi tinggi gambar); lebih besar = terlalu dekat/condong")
    ap.add_argument("--snapshots", action="store_true",
                    help="simpan contact sheet: frame video asli di tiap note (butuh file video)")
    ap.add_argument("--show", action="store_true")
    return ap


def main():
    cfg = build_parser().parse_args()
    if cfg.use_z and cfg.no_z:
        sys.exit("--use-z dan --no-z tidak bisa dipakai bersamaan.")
    out = cfg.out or Path(cfg.video).stem

    frames, meta = load_frames(cfg, out)
    if not frames:
        sys.exit("Tidak ada pose terdeteksi.")

    res = process(frames, meta, cfg)
    q = res["quality"]

    print("\n== KUALITAS ==")
    for k_, v_ in q.items():
        if k_ != "warnings":
            print(f"  {k_}: {v_}")
    for w in q["warnings"]:
        print(f"  [!] {w}")
    if not q["usable"]:
        print("\n  >>> VIDEO BELUM LAYAK jadi template resmi. Saran:")
        print("      - Rekam ulang: seluruh lengan (bahu sampai ujung jari) harus selalu di dalam frame")
        print("      - Mundur / zoom out, kamera sejajar dada, tripod, pencahayaan terang, baju kontras dgn latar")
        print("      - Atau potong video ke bagian yang bagus: --start X --end Y")
        print("      Hasil di bawah hanya untuk uji pipeline.")
    if "error" in res:
        sys.exit(res["error"])

    gaps = detection_gaps(frames, meta["fps"])
    if gaps:
        print(f"\n== {len(gaps)} celah deteksi (pose tidak terbaca) ==")
        for a, b in gaps[:10]:
            print(f"  {a:.2f}s -> {b:.2f}s ({b - a:.2f}s)")
        if len(gaps) > 10:
            print(f"  ... +{len(gaps) - 10} lagi")

    print(f"\n== {len(res['timeline'])} note -> {len(res['poses'])} pose unik ==")
    for p in res["poses"]:
        sp_txt = f"spread ±{p['spreadDeg']}°" if p["spreadDeg"] is not None else "spread n/a (1 sampel)"
        print(f"  {p['id']} x{p['count']}  {sp_txt}")
    for e in res["timeline"]:
        tt = f"{e['tStart']:.2f}-{e['tEnd']:.2f}s" if e["type"] == "hold" else f"{e['t']:.2f}s"
        print(f"  {tt:>13}  {e['pose']:<4} {e['type']}")

    doc = {
        "schema": "dance-template/0.2",
        "source": {"video": Path(cfg.video).name, "trim": [cfg.start, cfg.end]},
        "features": {"bones": BONE_NAMES,
                     "frame": "shoulder-axis body frame, unit vectors [x kanan-gambar, y bawah, z kedalaman]",
                     "z": cfg.use_z},
        "poses": res["poses"], "timeline": res["timeline"], "quality": q,
        "params": {"thresh": cfg.thresh, "min_hold": cfg.min_hold, "cluster": cfg.cluster,
                   "max_shoulder": cfg.max_shoulder, "variant": cfg.variant},
    }
    json.dump(doc, open(f"{out}.template.json", "w"), indent=1)
    save_plots(res, out, cfg.show)
    if cfg.snapshots:
        save_contact_sheet(cfg.video, res["timeline"], out)
    print(f"\nTersimpan: {out}.template.json, {out}_poses.png, {out}_timeline.png")


if __name__ == "__main__":
    main()