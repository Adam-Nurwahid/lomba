import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks

data = json.load(open("landmarks.json"))
t = np.array([f["t"] for f in data])
W = np.array([f["world"] for f in data])          # (n, 33, 3)

# --- laporan frame hilang ---
dt = np.diff(t)
gaps = [(t[i], t[i + 1]) for i in np.where(dt > 1.5 / 30)[0]]
print(f"{len(gaps)} celah deteksi:")
for a, b in gaps:
    print(f"  {a:.2f}s -> {b:.2f}s ({b - a:.2f}s)")

# --- sudut di titik b antara a-b dan c-b ---
def angle(a, b, c):
    ba, bc = a - b, c - b
    cos = (ba * bc).sum(-1) / (np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))

raw = {
    "L_elbow":    angle(W[:, 11], W[:, 13], W[:, 15]),
    "R_elbow":    angle(W[:, 12], W[:, 14], W[:, 16]),
    "L_shoulder": angle(W[:, 23], W[:, 11], W[:, 13]),
    "R_shoulder": angle(W[:, 24], W[:, 12], W[:, 14]),
}

# --- resample 30 Hz + smoothing ---
grid = np.arange(t[0], t[-1], 1 / 30)
J = {k: savgol_filter(np.interp(grid, t, v), 15, 2) for k, v in raw.items()}
speed = sum(np.abs(np.gradient(v, grid)) for v in J.values())
print("kecepatan (°/s) persentil 10/25/50/75:", np.percentile(speed, [10, 25, 50, 75]).round(0))

def med(a, b):
    m = (grid >= a) & (grid <= b)
    return np.array([np.median(J[k][m]) for k in J])

# --- pose ditahan ---
THRESH = 120      # °/s
MIN_HOLD = 0.4    # detik
MERGE_GAP = 0.5   # detik: gabung segmen yang berjarak <= ini
MERGE_DIFF = 12   # derajat: ...dan selisih median sudut terbesar <= ini

still = speed < THRESH
for a, b in gaps:
    still[(grid > a - 0.3) & (grid < b + 0.3)] = False

segs, start = [], None
for i, s in enumerate(still):
    if s and start is None:
        start = i
    if start is not None and (not s or i == len(still) - 1):
        if grid[i] - grid[start] >= MIN_HOLD:
            segs.append((grid[start], grid[i]))
        start = None

merged = []
for a, b in segs:
    if merged and a - merged[-1][1] <= MERGE_GAP and np.abs(med(a, b) - med(*merged[-1])).max() <= MERGE_DIFF:
        merged[-1] = (merged[-1][0], b)
    else:
        merged.append((a, b))
segs = merged

print(f"\n{len(segs)} pose ditahan:")
for a, b in segs:
    print(f"  {a:.2f}-{b:.2f}s | " + ", ".join(f"{k}={v:.0f}°" for k, v in zip(J, med(a, b))))

# --- lembah kecepatan = aksen (di luar pose ditahan dan celah) ---
valleys, _ = find_peaks(-speed, distance=int(0.4 * 30), prominence=80)
valleys = [i for i in valleys
           if speed[i] < 100
           and not any(a - 0.3 < grid[i] < b + 0.3 for a, b in gaps)
           and not any(a - 0.5 <= grid[i] <= b + 0.5 for a, b in segs)]
print(f"\n{len(valleys)} lembah (aksen):")
for i in valleys:
    print(f"  {grid[i]:.2f}s | {speed[i]:.0f}°/s | " + ", ".join(f"{k}={J[k][i]:.0f}°" for k in J))

# --- ekspor draft untuk template & stickman ---
LM_IDX = [0, 11, 12, 13, 14, 15, 16, 23, 24]       # hidung, bahu, siku, pergelangan, pinggul
Wg = np.stack([np.interp(grid, t, W[:, i, c]) for i in LM_IDX for c in range(3)], axis=1)
Wg = Wg.reshape(len(grid), len(LM_IDX), 3)

notes = []
for a, b in segs:
    m = (grid >= a) & (grid <= b)
    notes.append({
        "type": "hold", "tStart": round(float(a), 2), "tEnd": round(float(b), 2),
        "angles": {k: round(float(np.median(J[k][m])), 1) for k in J},
        "spread": {k: round(float(np.std(J[k][m])), 1) for k in J},
        "lm": np.median(Wg[m], axis=0).round(4).tolist(),
    })
for i in valleys:
    notes.append({
        "type": "accent", "t": round(float(grid[i]), 2),
        "angles": {k: round(float(J[k][i]), 1) for k in J},
        "lm": Wg[i].round(4).tolist(),
    })
notes.sort(key=lambda n: n.get("tStart", n.get("t")))
json.dump({"schema": "draft/0.1", "lmIndex": LM_IDX, "notes": notes},
          open("poses_draft.json", "w"), indent=1)
print(f"\nposes_draft.json: {len(notes)} note")

# --- plot ---
fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
for k, v in J.items():
    ax[0].plot(grid, v, label=k)
ax[1].plot(grid, speed, color="gray")
ax[1].axhline(THRESH, color="orange", ls=":", lw=1)
for a, b in segs:
    for x in ax:
        x.axvspan(a, b, alpha=0.2, color="green")
for a, b in gaps:
    for x in ax:
        x.axvspan(a, b, alpha=0.3, color="red")
for i in valleys:
    for x in ax:
        x.axvline(grid[i], color="black", ls="--", lw=0.8)
ax[0].set_ylabel("sudut (°)"); ax[0].legend()
ax[1].set_ylabel("kecepatan (°/s)"); ax[1].set_xlabel("detik")
plt.tight_layout(); plt.savefig("angles.png", dpi=120); plt.show()