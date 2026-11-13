"""
Saman Coach - Computer Vision & Rhythm Training Toolkit for Tari Saman.
"""

__version__ = "0.3.0"
__author__ = "Adam Nurwahid & Contributors"

from .constants import BONE_NAMES, BONE_IDX, FPS
from .pose import ensure_model, extract_landmarks, bones_body, check_quality
from .clustering import process_poses
from .visualizer import save_plots, save_contact_sheet

__all__ = [
    "BONE_NAMES",
    "BONE_IDX",
    "FPS",
    "ensure_model",
    "extract_landmarks",
    "bones_body",
    "check_quality",
    "process_poses",
    "save_plots",
    "save_contact_sheet",
]
