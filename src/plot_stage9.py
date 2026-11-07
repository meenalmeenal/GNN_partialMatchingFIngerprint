"""Headline Stage 9 figure from results/stage9/sweep_metrics.json."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("results/stage9/sweep_metrics.json"))
areas = [100, 75, 50, 25, 10]
series = [  # key, label, colour, marker
    ("classical", "Classical Hough", "#2f7d5d", "o"),
    ("gnn_inv_local_cont", "Invariant GNN + MaxSim matching (Stage 9)", "#00798c", "h"),
    ("gnn_v1_global", "Stage 3 GCN + stronger loss (global vector)", "#7a4aa8", "v"),
    ("gnn_inv_global", "Invariant GNN, pooled to one vector", "#b8860b", "P"),
    ("gnn_trained", "Stage 3 GCN (as first trained)", "#8a8f99", "s"),
]
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
for key, label, col, mk in series:
    r1 = [d["levels"][str(a)][key]["rank1"] * 100 for a in areas]
    ax[0].plot(areas, r1, marker=mk, color=col, lw=2.2 if "inv_local" in key or key == "classical" else 1.5,
               label=label)
    m = lambda e: (e["eer_tnorm"] if "eer_tnorm" in e else e["eer"]) * 100
    ax[1].plot(areas, [m(d["levels"][str(a)][key]) for a in areas], marker=mk, color=col,
               lw=2.2 if "inv_local" in key or key == "classical" else 1.5)
for a_ in ax:
    a_.invert_xaxis(); a_.grid(alpha=.3); a_.set_xlabel("Query print area retained (%)")
ax[0].set_ylabel("Rank-1 identification (%)"); ax[0].set_title("Identification (200-finger gallery)")
ax[1].set_ylabel("Equal Error Rate (%)  - lower is better"); ax[1].set_title("Verification (test-normalised)")
fig.legend(loc="lower center", ncol=2, frameon=False, fontsize=9)
fig.tight_layout(rect=(0, 0.14, 1, 1))
fig.savefig("results/stage9/headline.png", dpi=130)
print("wrote results/stage9/headline.png")
