"""
Stage 6 - partiality sweep: how do the GCN and the classical Hough matcher degrade as
the query print shrinks?

For each altered probe, keep only a contiguous square window covering a fraction of the
print area (100/75/50/25/10 %), rebuild the k-NN graph from the surviving minutiae, and
score it against the FULL gallery templates with (a) the trained GCN, (b) an untrained
GCN (floor), and (c) the classical Hough matcher. Same crops for every method.

    python src/partiality_sweep.py --limit-fingers 200
"""
import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from evaluate import (collect, embed_paths, identification, verification,
                      build_untrained_model)
from match import load_model
from graph_build import minutiae_to_graph
from minutiae import Minutia
from classical_match import graph_to_minutiae, evaluate_classical

IMG_PX = 160          # extractor image size (true pixel space)
GRAPH_NORM = 256      # graph_build normalises coordinates by 256, not 160
LEVELS = [1.0, 0.75, 0.5, 0.25, 0.10]
MIN_MINUTIAE = 3


def graph_to_pixels(data):
    x = data.x.numpy()
    return (x[:, 0] * GRAPH_NORM, x[:, 1] * GRAPH_NORM,
            np.arctan2(x[:, 2], x[:, 3]), x[:, 4])


def crop_graph(data, frac, rng, k=5):
    """Keep a square window covering `frac` of the print; returns (graph, n_kept, fallback)."""
    px, py, th, ty = graph_to_pixels(data)
    n = len(px)
    side = math.sqrt(frac) * IMG_PX
    best_idx, best_cnt = None, -1
    for _ in range(25):
        c = rng.randrange(n)
        x0 = min(max(px[c] - side / 2, 0), IMG_PX - side)
        y0 = min(max(py[c] - side / 2, 0), IMG_PX - side)
        idx = np.where((px >= x0) & (px <= x0 + side) & (py >= y0) & (py <= y0 + side))[0]
        if len(idx) > best_cnt:
            best_idx, best_cnt = idx, len(idx)
        if best_cnt >= MIN_MINUTIAE:
            break
    fallback = False
    if best_cnt < 2:   # window too sparse to form a graph: keep the 2 nearest to the centre
        fallback = True
        c = rng.randrange(n)
        d = (px - px[c]) ** 2 + (py - py[c]) ** 2
        best_idx = np.argsort(d)[:2]
    mins = [Minutia(float(px[i]), float(py[i]), float(th[i]), int(ty[i])) for i in best_idx]
    return minutiae_to_graph(mins, k=k), len(best_idx), fallback


def gnn_metrics(model, gallery_graphs, probe_graphs, pf_idx, device):
    ge = embed_paths(model, gallery_graphs, device)
    pe = embed_paths(model, probe_graphs, device)
    gf = torch.arange(len(gallery_graphs))
    pf = torch.tensor(pf_idx)
    ident = identification(pe, pf, ge, gf)
    ver, _ = verification(pe, pf, ge, gf)
    return {"rank1": ident["rank1"], "rank5": ident["rank5"],
            "eer": ver["eer"], "auc": ver["auc"]}


def make_plots(results, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    areas = [int(l * 100) for l in LEVELS]
    series = [("classical", "Classical Hough", "#2f7d5d", "o"),
              ("gnn_trained", "GCN (trained)", "#2f6ea5", "s"),
              ("gnn_untrained", "GCN (untrained floor)", "#8a8f99", "^")]
    for metric, ylabel, fname, pct in [("rank1", "Rank-1 identification (%)", "sweep_rank1.png", True),
                                       ("eer", "Equal Error Rate (%)", "sweep_eer.png", True)]:
        plt.figure(figsize=(6.2, 4.4))
        for key, label, color, mk in series:
            ys = [results["levels"][str(a)][key][metric] * 100 for a in areas]
            plt.plot(areas, ys, marker=mk, color=color, lw=2, label=label)
        plt.gca().invert_xaxis()
        plt.xlabel("Query print area retained (%)")
        plt.ylabel(ylabel)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.title(ylabel + " vs. available print area")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, fname), dpi=120)
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="models/best_model.pt")
    ap.add_argument("--splits", default="models/splits.json")
    ap.add_argument("--packed", default="data/packed/test.pt")
    ap.add_argument("--graph_dir", default="data/graphs")
    ap.add_argument("--limit-fingers", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="results/stage6")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cpu")

    model, cfg = load_model(args.checkpoint, device)
    untrained, _ = build_untrained_model("configs/default.yaml", device)
    torch.manual_seed(0)  # untrained floor reproducible

    sp = json.load(open(args.splits))
    fingers = sp["test"][:args.limit_fingers]
    packed = args.packed if os.path.exists(args.packed) else None
    gallery, probes = collect(args.graph_dir, fingers, packed=packed)
    fids = list(gallery)
    fidx = {f: i for i, f in enumerate(fids)}
    gallery_graphs = [gallery[f] for f in fids]
    gallery_m = [graph_to_minutiae(g) for g in gallery_graphs]
    pf_idx = [fidx[f] for _, f, _ in probes]
    print(f"{len(fids)} fingers, {len(probes)} probes, levels={LEVELS}")

    results = {"n_fingers": len(fids), "n_probes": len(probes), "seed": args.seed,
              "min_minutiae": MIN_MINUTIAE, "levels": {}}
    for frac in LEVELS:
        t0 = time.time()
        rng = random.Random(args.seed)
        if frac >= 1.0:
            graphs, counts, fb = [g for g, _, _ in probes], [g.num_nodes for g, _, _ in probes], 0
        else:
            graphs, counts, fb = [], [], 0
            for g, _, _ in probes:
                cg, n, fell = crop_graph(g, frac, rng)
                graphs.append(cg); counts.append(n); fb += int(fell)
        probes_m = [graph_to_minutiae(g) for g in graphs]
        cm, _ = evaluate_classical(gallery_m, probes_m, pf_idx, fids)
        entry = {
            "classical": {k: cm[k] for k in ("rank1", "rank5", "eer", "auc")},
            "gnn_trained": gnn_metrics(model, gallery_graphs, graphs, pf_idx, device),
            "gnn_untrained": gnn_metrics(untrained, gallery_graphs, graphs, pf_idx, device),
            "minutiae_per_probe": {"mean": float(np.mean(counts)), "median": float(np.median(counts)),
                                   "min": int(np.min(counts))},
            "fallback_probes": fb,
        }
        results["levels"][str(int(frac * 100))] = entry
        print(f"[{int(frac*100):3d}%] minutiae/probe mean={entry['minutiae_per_probe']['mean']:.1f}  "
              f"R1 classical={cm['rank1']*100:.1f}  gcn={entry['gnn_trained']['rank1']*100:.1f}  "
              f"untrained={entry['gnn_untrained']['rank1']*100:.1f}   ({time.time()-t0:.0f}s)")
        json.dump(results, open(os.path.join(args.outdir, "sweep_metrics.json"), "w"), indent=2)

    make_plots(results, args.outdir)
    print("done ->", args.outdir)


if __name__ == "__main__":
    main()
