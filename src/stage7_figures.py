"""Static example figure for the Stage 7 notes: query (with the kept window) + the learned
matcher's top-4 for a few settings. Probes are drawn with a fixed seed, NOT picked for success
(misses are shown as they come)."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import demo_core as D
from model_v2 import load_matcher

SETTINGS = [("socofing", D.SOCO_MODEL, 0.5),
            ("socofing", D.SOCO_MODEL, 0.25),
            ("fvc2002_db2", D.FVC_MODEL, 1.0),
            ("fvc2002_db2", D.FVC_MODEL, 0.5)]
K = 4
rng = random.Random(7)
fig, axes = plt.subplots(len(SETTINGS), K + 1, figsize=(2.4 * (K + 1), 2.5 * len(SETTINGS)))
cache = {}
for r, (dk, ml, area) in enumerate(SETTINGS):
    if (dk, ml) not in cache:
        ds = D.load_dataset(dk)
        cache[(dk, ml)] = (ds, D.Gallery(ds, load_matcher(D.model_path(ml))))
    ds, gal = cache[(dk, ml)]
    f = rng.choice(ds.fingers)
    key = rng.choice(ds.query_keys(f))
    q_full = ds.graph(f, key)
    qg, keep, rect = D.crop_window(q_full, area, rng.random(), rng.random())
    s, _ = gal.search_learned(qg)
    order = np.argsort(-s)
    z = D.zscores(s)
    ti = gal.fingers.index(f)
    D.draw_minutiae(axes[r, 0], ds.image(f, key), q_full, keep=keep if area < 1 else None, rect=rect,
                    arrows=False, title=f"query: {int(area * 100)}% of print\n{ds.label(f)} / {key}", )
    for c in range(K):
        gi = int(order[c])
        ax = axes[r, c + 1]
        ax.imshow(ds.image(gal.fingers[gi], ds.gallery_key), cmap="gray")
        ax.set_xticks([]); ax.set_yticks([])
        ok = gi == ti
        for sp in ax.spines.values():
            sp.set_edgecolor("#1b7f4b" if ok else "#cccccc"); sp.set_linewidth(4 if ok else 1)
        ax.set_title(f"#{c + 1} {ds.label(gal.fingers[gi])}\nz {z[gi]:+.1f}" + ("  TRUE" if ok else ""),
                     fontsize=8, color="#1b7f4b" if ok else "black")
    axes[r, 0].set_ylabel(("SOCOFing" if dk == "socofing" else dk.upper()), fontsize=9)
    print(f"{dk} {int(area * 100)}%: true rank {D.rank_of(s, ti)} of {len(gal.fingers)}")
fig.tight_layout()
os.makedirs("results/stage9e", exist_ok=True)
fig.savefig("results/stage9e/examples.png", dpi=110)
