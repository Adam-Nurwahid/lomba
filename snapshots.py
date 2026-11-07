import sys, json
import cv2
import numpy as np

video = sys.argv[1]
tpl = json.load(open("template_v0.json"))
cap = cv2.VideoCapture(video)

tiles = []
for e in tpl["timeline"]:
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

cols = 4
H, W = tiles[0].shape[:2]
while len(tiles) % cols:
    tiles.append(np.zeros((H, W, 3), np.uint8))
sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
cv2.imwrite("contact_sheet.png", sheet)
print("contact_sheet.png tersimpan,", len(tiles), "tile")