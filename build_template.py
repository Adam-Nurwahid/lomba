import json
import numpy as np
import matplotlib.pyplot as plt

JN = ["L_elbow", "R_elbow", "L_shoulder", "R_shoulder"]
CLUSTER_DIFF = 15   # derajat: pose dianggap sama bila semua sudut selisih <= ini

draft = json.load(open("poses_draft.json"))
notes = draft["notes"]

centroids, members, assign = [], [], []
for n in notes:
    vec = np.array([n["angles"][k] for k in JN])
    pid = next((j for j, c in enumerate(centroids) if np.abs(vec - c).max() <= CLUSTER_DIFF), None)
    if pid is None:
        centroids.append(vec)
        members.append([])
        pid = len(centroids) - 1
    members[pid].append(n)
    centroids[pid] = np.mean([[m["angles"][k] for k in JN] for m in members[pid]], axis=0)
    assign.append(pid)

poses = []
for j, ms in enumerate(members):
    poses.append({
        "id": f"P{j + 1}",
        "angles": dict(zip(JN, np.round(centroids[j], 1).tolist())),
        "spread": {k: round(float(np.std([m["angles"][k] for m in ms])), 1) for k in JN},
        "lm": np.mean([m["lm"] for m in ms], axis=0).round(4).tolist(),
        "count": len(ms),
    })

timeline = []
for n, pid in zip(notes, assign):
    e = {"type": n["type"], "pose": f"P{pid + 1}"}
    if n["type"] == "hold":
        e["tStart"], e["tEnd"] = n["tStart"], n["tEnd"]
    else:
        e["t"] = n["t"]
    timeline.append(e)

json.dump({"schema": "dance-template/0.1-draft", "lmIndex": draft["lmIndex"],
           "poses": poses, "timeline": timeline},
          open("template_v0.json", "w"), indent=1)

print(f"{len(notes)} note -> {len(poses)} pose unik")
for p in poses:
    print(f"  {p['id']} (x{p['count']}): " + ", ".join(f"{k}={v:.0f}" for k, v in p["angles"].items()))
print("\ntimeline:")
for e in timeline:
    t = f"{e['tStart']:.2f}-{e['tEnd']:.2f}s" if e["type"] == "hold" else f"{e['t']:.2f}s"
    print(f"  {t:>12}  {e['pose']}  ({e['type']})")

# --- preview stickman hitam-putih ---
# urutan lm: 0 hidung, 1 bahu kiri, 2 bahu kanan, 3 siku kiri, 4 siku kanan,
#            5 pergelangan kiri, 6 pergelangan kanan, 7 pinggul kiri, 8 pinggul kanan
EDGES = [(1, 2), (1, 3), (3, 5), (2, 4), (4, 6), (1, 7), (2, 8), (7, 8)]
cols = min(4, len(poses))
rows = int(np.ceil(len(poses) / cols))
fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3.6 * rows), squeeze=False)
for ax in axs.flat:
    ax.axis("off")
for ax, p in zip(axs.flat, poses):
    L = np.array(p["lm"])
    x, y = L[:, 0], L[:, 1]
    for a, b in EDGES:
        ax.plot([x[a], x[b]], [y[a], y[b]], "k-", lw=3, solid_capstyle="round")
    ax.plot([(x[1] + x[2]) / 2, x[0]], [(y[1] + y[2]) / 2, y[0]], "k-", lw=3)   # leher
    ax.add_patch(plt.Circle((x[0], y[0]), 0.09, fill=False, lw=3))              # kepala
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0.15, -0.7)       # y world MediaPipe ke bawah; kalau stickman terbalik, ganti jadi (-0.15, 0.7)
    ax.set_aspect("equal")
    ax.set_title(f"{p['id']}  (x{p['count']})")
plt.tight_layout(); plt.savefig("stickman_preview.png", dpi=120); plt.show()