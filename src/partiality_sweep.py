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
                      identification_sim, verification_sim, build_untrained_model)
from model_v2 import load_matcher, score_matrix
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


def v2_metrics(model, gallery_graphs, probe_graphs, pf_idx):
    sim = score_matrix(model, probe_graphs, gallery_graphs)
    pf, gf = torch.tensor(pf_idx), torch.arange(len(gallery_graphs))
    ident = identification_sim(sim, pf, gf)
    ver, _ = verification_sim(sim, pf, gf)
    # test-normalisation: z-score each probe's scores over the gallery, so one global
    # threshold is meaningful across probes (does not change any ranking / Rank-k)
    sim_n = (sim - sim.mean(1, keepdim=True)) / sim.std(1, keepdim=True).clamp_min(1e-6)
    ver_n, _ = verification_sim(sim_n, pf, gf)
    return {"rank1": ident["rank1"], "rank5": ident["rank5"], "eer": ver["eer"], "auc": ver["auc"],
            "eer_tnorm": ver_n["eer"], "auc_tnorm": ver_n["auc"]}


def make_plots(results, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    areas = [int(l * 100) for l in LEVELS]
    first = results["levels"][str(areas[0])]
    palette = ["#2f6ea5", "#c0562b", "#7a4aa8", "#b8860b", "#d1495b", "#00798c", "#6a994e"]
    markers = ["s", "D", "v", "P", "X", "h", "<"]
    series = [("classical", "Classical Hough", "#2f7d5d", "o")] if "classical" in first else []
    extras = [k for k in first if k.startswith("gnn_") and k not in ("gnn_trained", "gnn_untrained")]
    series.append(("gnn_trained", "GCN (Stage 3, full prints only)", palette[0], markers[0]))
    for i, key in enumerate(extras):
        series.append((key, key[4:], palette[(i + 1) % 7], markers[(i + 1) % 7]))
    series.append(("gnn_untrained", "GCN (untrained floor)", "#8a8f99", "^"))
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
    ap.add_argument("--models", nargs="*", default=[],
                    help="extra GCN checkpoints to score, as name=path")
    ap.add_argument("--models-v2", nargs="*", default=[],
                    help="Stage 9 Matcher checkpoints (model_v2), as name=path")
    ap.add_argument("--skip-classical", action="store_true",
                    help="omit the classical matcher (it is O(fingers^2); needed for 900 fingers)")
    ap.add_argument("--reuse-classical", default=None,
                    help="sweep_metrics.json from an earlier run: reuse its classical results "
                         "(valid because crops are deterministic for a given seed)")
    ap.add_argument("--outdir", default="results/stage6")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cpu")

    model, cfg = load_model(args.checkpoint, device)
    extra_models = {}
    for spec in args.models:
        name, path = spec.split("=", 1)
        extra_models["gnn_" + name], _ = load_model(path, device)
    v2_models = {}
    for spec in args.models_v2:
        name, path = spec.split("=", 1)
        v2_models["gnn_" + name] = load_matcher(path, device)
    reused = json.load(open(args.reuse_classical)) if args.reuse_classical else None
    torch.manual_seed(0)  # seed BEFORE building so the untrained floor is reproducible
    untrained, _ = build_untrained_model("configs/default.yaml", device)

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
        if args.skip_classical:
            cm = None
        elif reused is not None:
            old = reused["levels"][str(int(frac * 100))]
            assert reused["n_probes"] == len(probes) and reused["seed"] == args.seed
            assert abs(old["minutiae_per_probe"]["mean"] - float(np.mean(counts))) < 1e-9, "crops differ"
            cm = old["classical"]
        else:
            probes_m = [graph_to_minutiae(g) for g in graphs]
            cm, cscores = evaluate_classical(gallery_m, probes_m, pf_idx, fids)
            cs = torch.as_tensor(np.asarray(cscores), dtype=torch.float32)
            cs = (cs - cs.mean(1, keepdim=True)) / cs.std(1, keepdim=True).clamp_min(1e-6)
            vn, _ = verification_sim(cs, torch.tensor(pf_idx), torch.arange(len(gallery_m)))
            cm = {**cm, "eer_tnorm": vn["eer"], "auc_tnorm": vn["auc"]}
        entry = {
            **({"classical": {k: cm[k] for k in ("rank1", "rank5", "eer", "auc", "eer_tnorm", "auc_tnorm")
                              if k in cm}} if cm else {}),
            "gnn_trained": gnn_metrics(model, gallery_graphs, graphs, pf_idx, device),
            "gnn_untrained": gnn_metrics(untrained, gallery_graphs, graphs, pf_idx, device),
            **{name: gnn_metrics(m, gallery_graphs, graphs, pf_idx, device)
               for name, m in extra_models.items()},
            **{name: v2_metrics(m, gallery_graphs, graphs, pf_idx) for name, m in v2_models.items()},
            "minutiae_per_probe": {"mean": float(np.mean(counts)), "median": float(np.median(counts)),
                                   "min": int(np.min(counts))},
            "fallback_probes": fb,
        }
        results["levels"][str(int(frac * 100))] = entry
        line = "  ".join(f"{k[4:] if k.startswith('gnn_') else k}={v['rank1']*100:.1f}"
                         for k, v in entry.items() if isinstance(v, dict) and "rank1" in v)
        print(f"[{int(frac*100):3d}%] minutiae/probe={entry['minutiae_per_probe']['mean']:.1f}  "
              f"R1: {line}   ({time.time()-t0:.0f}s)")
        json.dump(results, open(os.path.join(args.outdir, "sweep_metrics.json"), "w"), indent=2)

    make_plots(results, args.outdir)
    print("done ->", args.outdir)


if __name__ == "__main__":
    main()
