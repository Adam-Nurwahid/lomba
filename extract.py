# extract.py — belum saya jalankan, uji di mesinmu dan kirim error kalau ada
import cv2, json, time, sys
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

video_path = sys.argv[1]
opts = vision.PoseLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="pose_landmarker_lite.task"),
    running_mode=vision.RunningMode.VIDEO, num_poses=1)

frames, t0 = [], time.time()
with vision.PoseLandmarker.create_from_options(opts) as lm:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok: break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = lm.detect_for_video(img, int(i * 1000 / fps))
        if res.pose_landmarks:
            frames.append({
                "t": i / fps,
                "lm": [[p.x, p.y, p.z, p.visibility] for p in res.pose_landmarks[0]],
                "world": [[p.x, p.y, p.z] for p in res.pose_world_landmarks[0]],
            })
        i += 1

print(f"{i} frame, {i / (time.time() - t0):.1f} FPS proses, {len(frames)} frame terdeteksi")
json.dump(frames, open("landmarks.json", "w"))