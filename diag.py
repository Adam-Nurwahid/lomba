import json
import numpy as np

USE_Z = True    # coba False juga: kedalaman (z) adalah bagian yang paling tidak stabil

data = json.load(open("landmarks.json"))
t = np.array([f["t"] for f in data])
W = np.array([f["world"] for f in data])                       # (n, 33, 3)
V = np.array([[p[3] for p in f["lm"]] for f in data])          # (n, 33) visibilitas
if not USE_Z:
    W = W.copy(); W[:, :, 2] = 0

# 1) visibilitas median di bagian tarian (5-17 s)
m = (t >= 5) & (t <= 17)
names = {0: "hidung", 11: "bahu_kiri", 12: "bahu_kanan", 13: "siku_kiri", 14: "siku_kanan",
         15: "pergelangan_kiri", 16: "pergelangan_kanan", 23: "pinggul_kiri", 24: "pinggul_kanan"}
print("visibilitas median (0-1), 5-17 s:")
for i, n in names.items():
    print(f"  {n:18s} {np.median(V[m, i]):.2f}")

# 2) fitur v2: arah tulang (vektor satuan)
BONES = [(11, 13), (13, 15), (12, 14), (14, 16)]
def bones(L):
    out = []
    for a, b in BONES:
        v = L[b] - L[a]
        out.append(v / (np.linalg.norm(v) + 1e-9))
    return np.array(out)

def pose_at(n):
    if n["type"] == "hold":
        sel = (t >= n["tStart"]) & (t <= n["tEnd"])
    else:
        sel = np.abs(t - n["t"]) <= 0.05
    if not sel.any():
        sel = np.zeros(len(t), bool); sel[np.argmin(np.abs(t - n.get("t", n.get("tStart", 0))))] = True
    return bones(np.median(W[sel], axis=0))

def dist(A, B):   # rata-rata sudut antar-tulang, derajat
    return np.degrees(np.arccos(np.clip((A * B).sum(1), -1, 1))).mean()

notes = json.load(open("poses_draft.json"))["notes"]
tl = json.load(open("template_v0.json"))["timeline"]
P = [pose_at(n) for n in notes]
lab = [f"{e['pose']}@{(e['tStart'] + e['tEnd']) / 2 if e['type'] == 'hold' else e['t']:.1f}" for e in tl]

D = np.array([[dist(a, b) for b in P] for a in P])
print("\njarak arah-tulang antar note (derajat):")
print(" " * 10 + "".join(f"{i:>4d}" for i in range(len(lab))))
for i, l in enumerate(lab):
    print(f"{i:>2d} {l:<8s}" + "".join(f"{D[i, j]:>4.0f}" for j in range(len(lab))))

print("\nSATU pose (versi sudut) tapi berbeda > 35° (versi arah tulang):")
for i in range(len(lab)):
    for j in range(i + 1, len(lab)):
        if tl[i]["pose"] == tl[j]["pose"] and D[i, j] > 35:
            print(f"  {lab[i]} vs {lab[j]}: {D[i, j]:.0f}°")
print("\nBEDA pose (versi sudut) tapi mirip < 20° (versi arah tulang):")
for i in range(len(lab)):
    for j in range(i + 1, len(lab)):
        if tl[i]["pose"] != tl[j]["pose"] and D[i, j] < 20:
            print(f"  {lab[i]} vs {lab[j]}: {D[i, j]:.0f}°")
            