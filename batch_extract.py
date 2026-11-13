#!/usr/bin/env python3
"""
batch_extract.py - Ekstraksi otomatis seluruh video Tari Saman.
"""

import sys
from pathlib import Path

src_path = Path(__file__).resolve().parent / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from saman_coach.cli import batch_main

if __name__ == "__main__":
    batch_main()
