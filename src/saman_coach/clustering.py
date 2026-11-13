"""
Algoritma segmentasi kecepatan, deteksi titik aksen (accent) & tahanan (hold),
serta pengelompokan pose unik (pose clustering).
"""

from typing import Dict, List, Tuple, Any
import numpy as np
from scipy.signal import find_peaks, savgol_filter

from .constants import FPS, BONE_NAMES, ACCENT_MAX_SPEED
from .pose import check_quality, bones_body, unit


def angular_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Rata-rata sudut (derajat) antar-tulang pada dua pose (4 tulang, 3 koordinat)."""
    dot_products = (a * b).sum(1)
    clipped = np.clip(dot_products, -1.0, 1.0)
    return float(np.degrees(np.arccos(clipped)).mean())


def find_runs(mask: np.ndarray, grid: np.ndarray, min_len: float) -> List[Tuple[int, int]]:
    """Cari rentang indeks di mana mask bernilai True dengan durasi >= min_len detik."""
    m = np.r_[mask, False]
    out = []
    start = None
    for i, v in enumerate(m):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if grid[i - 1] - grid[start] >= min_len:
                out.append((start, i - 1))
            start = None
    return out


def segment_pose(b: np.ndarray, s: int, e: int) -> np.ndarray:
    """Hitung pose representatif segmen menggunakan nilai median dari frame s s/d e."""
    return unit(np.median(b[s : e + 1], axis=0))


def process_poses(frames: List[Dict[str, Any]], meta: Dict[str, Any], cfg: Any) -> Dict[str, Any]:
    """
    Pipeline analisis pose:
    1. Filter frame berkualitas baik.
    2. Hitung vektor tulang dalam kerangka bahu.
    3. Interpolasi grid reguler dan Savitzky-Golay smoothing.
    4. Hitung kecepatan perubahan sudut sendi (°/s).
    5. Segmentasikan segmen diam (hold) dan titik balik kecepatan (accent).
    6. Klaster pose-pose serupa menjadi pose unik P1, P2, dst.
    """
    q, t, good = check_quality(frames, meta, getattr(cfg, "max_shoulder", 0.45))
    if good.sum() < 15:
        return {"quality": q, "error": "Terlalu sedikit frame berkualitas baik untuk dianalisis."}

    use_z = getattr(cfg, "use_z", False)
    thresh = getattr(cfg, "thresh", 120.0)
    min_hold = getattr(cfg, "min_hold", 0.4)
    cluster_threshold = getattr(cfg, "cluster", 18.0)

    bg = bones_body([f for f, g in zip(frames, good) if g], use_z)
    tg = t[good]
    grid = np.arange(tg[0], tg[-1], 1.0 / FPS)

    # Interpolasi ke grid waktu seragam
    b_interp = np.stack(
        [
            np.stack([np.interp(grid, tg, bg[:, k, c]) for c in range(3)], 1)
            for k in range(4)
        ],
        1,
    )
    b_smooth = unit(savgol_filter(b_interp, 15, 2, axis=0))

    # Tandai titik hasil interpolasi celah deteksi yang tidak boleh dipercaya
    idx = np.searchsorted(tg, grid)
    near = np.minimum(
        np.abs(grid - tg[np.clip(idx - 1, 0, len(tg) - 1)]),
        np.abs(grid - tg[np.clip(idx, 0, len(tg) - 1)])
    )
    bad = near > 0.1
    k = int(0.2 * FPS)
    bad = np.convolve(bad.astype(float), np.ones(2 * k + 1), "same") > 0

    # Kecepatan perubahan sudut antar frame (°/s)
    ang = np.degrees(np.arccos(np.clip((b_smooth[:-1] * b_smooth[1:]).sum(-1), -1.0, 1.0)))
    speed = ang.sum(1) * FPS
    speed = np.r_[speed[0], speed]

    # Segmentasi rentang pose diam (Hold)
    segs = find_runs((speed < thresh) & ~bad, grid, min_hold)
    merged = []
    for s, e in segs:
        if merged:
            ps, pe = merged[-1]
            if grid[s] - grid[pe] <= 0.5 and angular_dist(segment_pose(b_smooth, s, e), segment_pose(b_smooth, ps, pe)) <= 20:
                merged[-1] = (ps, e)
                continue
        merged.append((s, e))
    segs = merged

    # Deteksi titik aksen / ketukan gerak cepat (Accent)
    pk, _ = find_peaks(-speed, distance=int(0.4 * FPS), prominence=80)
    acc = [
        i for i in pk
        if speed[i] < ACCENT_MAX_SPEED
        and not bad[i]
        and not any(grid[s] - 0.5 <= grid[i] <= grid[e] + 0.5 for s, e in segs)
    ]

    notes = [
        {"type": "hold", "tStart": float(grid[s]), "tEnd": float(grid[e]), "bones": segment_pose(b_smooth, s, e)}
        for s, e in segs
    ]
    notes += [{"type": "accent", "t": float(grid[i]), "bones": b_smooth[i]} for i in acc]
    notes.sort(key=lambda n: n.get("tStart", n.get("t", 0.0)))

    # Pengelompokan (clustering) pose
    cent = []
    members = []
    assign = []

    for n in notes:
        best_idx, best_dist = None, cluster_threshold
        for j, c in enumerate(cent):
            d = angular_dist(n["bones"], c)
            if d <= best_dist:
                best_idx, best_dist = j, d

        if best_idx is None:
            cent.append(n["bones"].copy())
            members.append([n])
            assign.append(len(cent) - 1)
        else:
            members[best_idx].append(n)
            cent[best_idx] = unit(np.mean([m["bones"] for m in members[best_idx]], axis=0))
            assign.append(best_idx)

    poses = [
        {
            "id": f"P{j + 1}",
            "bones": {nm: np.round(cent[j][i], 4).tolist() for i, nm in enumerate(BONE_NAMES)},
            "spreadDeg": (
                round(float(np.mean([angular_dist(m["bones"], cent[j]) for m in ms])), 1)
                if len(ms) > 1 else None
            ),
            "count": len(ms),
        }
        for j, ms in enumerate(members)
    ]

    timeline = []
    for n, a in zip(notes, assign):
        item = {"type": n["type"], "pose": f"P{a + 1}"}
        if n["type"] == "hold":
            item["tStart"] = round(n["tStart"], 2)
            item["tEnd"] = round(n["tEnd"], 2)
        else:
            item["t"] = round(n["t"], 2)
        timeline.append(item)

    return {
        "quality": q,
        "poses": poses,
        "timeline": timeline,
        "centroids": cent,
        "grid": grid,
        "speed": speed,
        "bad": bad,
        "segs": segs,
        "acc": acc,
    }
