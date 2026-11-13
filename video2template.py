#!/usr/bin/env python3
"""
video2template.py - CLI wrapper kompatibel ke belakang untuk Saman Coach.
"""

import sys
from pathlib import Path

# Tambahkan src ke sys.path jika belum terinstal sebagai package
src_path = Path(__file__).resolve().parent / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from saman_coach.cli import main

if __name__ == "__main__":
    main()
