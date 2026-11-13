"""
Konstanta dan konfigurasi model untuk Saman Coach.
"""

FPS = 30

# Landmark MediaPipe Pose:
# 11: Bahu Kiri (Left Shoulder)
# 12: Bahu Kanan (Right Shoulder)
# 13: Siku Kiri (Left Elbow)
# 14: Siku Kanan (Right Elbow)
# 15: Pergelangan Tangan Kiri (Left Wrist)
# 16: Pergelangan Tangan Kanan (Right Wrist)
KEY_LANDMARKS = [11, 12, 13, 14, 15, 16]

BONE_NAMES = ["L_upper", "L_fore", "R_upper", "R_fore"]
BONE_IDX = [(11, 13), (13, 15), (12, 14), (14, 16)]

ACCENT_MAX_SPEED = 100  # Derajat/detik maksimum untuk aksen titik balik gerak

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task"
)
