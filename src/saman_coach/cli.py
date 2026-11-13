"""
Antarmuka Baris Perintah (CLI) untuk ekstraksi template pose Tari Saman.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .constants import BONE_NAMES
from .pose import load_frames, detection_gaps
from .clustering import process_poses
from .visualizer import save_plots, save_contact_sheet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="saman-extract",
        description="Ekstraksi pose video Tari Saman menjadi template JSON & visualisasi ritme.",
    )
    parser.add_argument("video", help="Path file video acuan (.mp4, .mov, dll.)")
    parser.add_argument("--out", default=None, help="Prefix nama file output (default: nama file video)")
    parser.add_argument("--model", default=None, help="Path file model .task lokal (default: unduh otomatis)")
    parser.add_argument(
        "--variant",
        choices=["lite", "full", "heavy"],
        default="lite",
        help="Varian model MediaPipe (default: lite)",
    )
    parser.add_argument("--start", type=float, default=0.0, help="Detik awal video yang dianalisis (default: 0.0)")
    parser.add_argument("--end", type=float, default=None, help="Detik akhir video yang dianalisis (default: selesai)")
    parser.add_argument("--reuse", action="store_true", help="Gunakan cache landmark JSON jika sudah ada")
    parser.add_argument("--use-z", action="store_true", help="Gunakan koordinat kedalaman z (default: false/dinonaktifkan)")
    parser.add_argument("--no-z", action="store_true", help="(Deprecated) z sudah nonaktif secara default")
    parser.add_argument("--thresh", type=float, default=120.0, help="Ambang batas kecepatan sudut untuk Hold (°/s)")
    parser.add_argument("--min-hold", type=float, default=0.4, help="Durasi minimal pose diam Hold (detik)")
    parser.add_argument("--cluster", type=float, default=18.0, help="Toleransi perbedaan sudut klaster pose (°)")
    parser.add_argument(
        "--max-shoulder",
        type=float,
        default=0.45,
        help="Lebar bahu maksimal (fraksi tinggi gambar)",
    )
    parser.add_argument(
        "--snapshots",
        action="store_true",
        help="Simpan contact sheet frame video penari asli di setiap pose",
    )
    parser.add_argument("--show", action="store_true", help="Tampilkan plot interaktif saat selesai")
    return parser


def run_extraction(cfg: argparse.Namespace) -> None:
    if cfg.use_z and cfg.no_z:
        sys.exit("Error: --use-z dan --no-z tidak bisa digunakan bersamaan.")

    out = cfg.out or Path(cfg.video).stem
    frames, meta = load_frames(cfg, out)

    if not frames:
        sys.exit("Error: Tidak ada pose yang terdeteksi pada video.")

    res = process_poses(frames, meta, cfg)
    q = res["quality"]

    print("\n" + "=" * 40)
    print("📋 LAPORAN KUALITAS VIDEO")
    print("=" * 40)
    for k, v in q.items():
        if k != "warnings":
            print(f"  {k}: {v}")
    for w in q["warnings"]:
        print(f"  ⚠️  [PERINGATAN] {w}")

    if not q["usable"]:
        print("\n  >>> Catatan: Kualitas video belum optimal untuk template resmi sanggar.")
    if "error" in res:
        sys.exit(f"\nError: {res['error']}")

    gaps = detection_gaps(frames, meta["fps"])
    if gaps:
        print(f"\nℹ️  Ditemukan {len(gaps)} celah deteksi:")
        for a, b in gaps[:5]:
            print(f"   {a:.2f}s -> {b:.2f}s ({b - a:.2f}s)")
        if len(gaps) > 5:
            print(f"   ... dan {len(gaps) - 5} lainnya.")

    print(f"\n✨ HASIL: {len(res['timeline'])} Ketukan Note -> {len(res['poses'])} Pose Unik")
    for p in res["poses"]:
        sp_txt = f"variasi ±{p['spreadDeg']}°" if p["spreadDeg"] is not None else "1 sampel"
        print(f"  • {p['id']} (x{p['count']}): {sp_txt}")

    # Simpan Template JSON
    doc = {
        "schema": "dance-template/0.2",
        "source": {"video": Path(cfg.video).name, "trim": [cfg.start, cfg.end]},
        "features": {
            "bones": BONE_NAMES,
            "frame": "shoulder-axis body frame, unit vectors [x kanan-gambar, y bawah, z kedalaman]",
            "z": cfg.use_z,
        },
        "poses": res["poses"],
        "timeline": res["timeline"],
        "quality": q,
        "params": {
            "thresh": cfg.thresh,
            "min_hold": cfg.min_hold,
            "cluster": cfg.cluster,
            "max_shoulder": cfg.max_shoulder,
            "variant": cfg.variant,
        },
    }

    json_path = f"{out}.template.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)

    save_plots(res, out, cfg.show)
    if cfg.snapshots:
        save_contact_sheet(cfg.video, res["timeline"], out)

    print(f"\n[✓] Berhasil disimpan:\n    - {json_path}\n    - {out}_poses.png\n    - {out}_timeline.png\n")


def main(args: Optional[List[str]] = None) -> None:
    parser = build_parser()
    cfg = parser.parse_args(args)
    run_extraction(cfg)


def batch_main() -> None:
    """Entry point untuk mengekstrak seluruh video sample secara otomatis."""
    candidate_dirs = [
        Path("videos/saman"),
        Path("../sanggar-digital/public/videos/saman"),
        Path("/Users/wedjaw/project/sanggar-digital/public/videos/saman"),
    ]

    target_dir = None
    for d in candidate_dirs:
        if d.exists() and list(d.glob("*.mp4")):
            target_dir = d
            break

    if not target_dir:
        print("[!] Direktori video tidak ditemukan di path default.")
        sys.exit(1)

    videos = sorted(list(target_dir.glob("*.mp4")))
    print(f"Menemukan {len(videos)} video di {target_dir}:")
    for v in videos:
        print(f" - {v.name}")
    print()

    for v in videos:
        print(f"▶ Memproses: {v.name}")
        parser = build_parser()
        cfg = parser.parse_args([str(v), "--snapshots", "--reuse"])
        run_extraction(cfg)


if __name__ == "__main__":
    main()
