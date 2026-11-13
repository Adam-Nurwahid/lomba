"""
Visualisasi pose stickman, profil kecepatan timeline, dan snapshot contact sheet.
"""

from typing import Dict, List, Any
from pathlib import Path
import numpy as np

from .constants import FPS


def draw_stick(ax: Any, bones: np.ndarray, lw: float = 3.0) -> None:
    """Gambar stickman Tari Saman 2D pada objek matplotlib Axes."""
    import matplotlib.pyplot as plt

    UP, FO = 0.8, 0.7  # Satuan skala panjang lengan terhadap lebar bahu

    def arm(sx: float, bu: np.ndarray, bf: np.ndarray):
        s = np.array([sx, 0.0])
        e = s + UP * bu[:2]
        w = e + FO * bf[:2]
        return s, e, w

    ls, le, lw_pt = arm(0.5, bones[0], bones[1])
    rs, re, rw_pt = arm(-0.5, bones[2], bones[3])
    hl, hr = np.array([0.35, 1.7]), np.array([-0.35, 1.7])

    lines = [
        (rs, ls), (ls, le), (le, lw_pt),
        (rs, re), (re, rw_pt),
        (ls, hl), (rs, hr), (hr, hl),
        (np.array([0.0, 0.0]), np.array([0.0, -0.25])),
    ]

    for a, b in lines:
        ax.plot([a[0], b[0]], [a[1], b[1]], "k-", lw=lw, solid_capstyle="round")

    ax.add_patch(plt.Circle((0.0, -0.55), 0.3, fill=False, lw=lw, color="black"))
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(2.0, -1.7)
    ax.set_aspect("equal")
    ax.axis("off")


def save_plots(res: Dict[str, Any], out: str, show: bool = False) -> None:
    """Simpan grafik diagram stickman pose unik dan grafik kecepatan gerak timeline."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    poses, cent = res["poses"], res["centroids"]
    cols = min(5, max(1, len(poses)))
    rows = int(np.ceil(len(poses) / cols))

    # 1. Gambar diagram stickman pose unik
    fig, axs = plt.subplots(rows, cols, figsize=(2.6 * cols, 3.2 * rows), squeeze=False)
    for ax in axs.flat:
        ax.axis("off")

    for ax, p, c in zip(axs.flat, poses, cent):
        draw_stick(ax, c)
        sp_txt = f", ±{p['spreadDeg']:.0f}°" if p["spreadDeg"] is not None else ""
        ax.set_title(f"{p['id']} (x{p['count']}{sp_txt})", fontsize=9, fontweight="bold")

    plt.tight_layout()
    poses_path = f"{out}_poses.png"
    fig.savefig(poses_path, dpi=110)
    plt.close(fig)

    # 2. Gambar grafik kecepatan dan timeline note
    g, sp = res["grid"], res["speed"]
    fig2, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(g, sp, color="#555555", lw=1.2, label="Kecepatan Lengan")

    for s, e in res["segs"]:
        ax.axvspan(g[s], g[e], alpha=0.25, color="#2ecc71")
    for i in np.where(res["bad"])[0]:
        ax.axvspan(g[i] - 0.5 / FPS, g[i] + 0.5 / FPS, alpha=0.15, color="#e74c3c", lw=0)

    for e in res["timeline"]:
        x = e["tStart"] if e["type"] == "hold" else e["t"]
        if e["type"] == "accent":
            ax.axvline(x, color="black", ls="--", lw=0.8)
            ax.text(x, ax.get_ylim()[1] * 0.93, e["pose"], fontsize=8, va="top", ha="center")

    ax.set_xlabel("Waktu (detik)")
    ax.set_ylabel("Kecepatan Tulang (°/s)")
    ax.set_title("Analisis Profil Kecepatan Tari & Ketukan Gerak", fontsize=11, fontweight="bold")
    plt.tight_layout()
    timeline_path = f"{out}_timeline.png"
    fig2.savefig(timeline_path, dpi=110)
    plt.close(fig2)

    if show:
        plt.show()


def save_contact_sheet(video: str, timeline: List[Dict[str, Any]], out: str, cols: int = 4) -> None:
    """Buat satu gambar lembar kontak (contact sheet) berisi frame asli video pada setiap ketukan pose."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"[!] Tidak dapat membuka video untuk contact sheet: {video}")
        return

    tiles = []
    for e in timeline:
        t = (e["tStart"] + e["tEnd"]) / 2.0 if e["type"] == "hold" else e["t"]
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (320, int(320 * h / w)))
        cv2.putText(
            frame,
            f"{e['pose']} {e['type']} @{t:.2f}s",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        tiles.append(frame)

    cap.release()
    if not tiles:
        print("[!] Contact sheet: Tidak ada frame yang terbaca.")
        return

    h, w = tiles[0].shape[:2]
    while len(tiles) % cols:
        tiles.append(np.zeros((h, w, 3), np.uint8))

    sheet = np.vstack([np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)])
    contact_path = f"{out}_contact.png"
    cv2.imwrite(contact_path, sheet)
    print(f"[✓] Contact sheet tersimpan: {contact_path} ({len(timeline)} note)")
