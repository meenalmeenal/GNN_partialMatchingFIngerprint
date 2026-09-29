"""
Stage 1 sanity report over the built graph dataset.

    python src/stage1_report.py --organized data/SOCOFing_organized --graphs data/graphs

Writes results/stage1_report.json and prints a summary:
  - #identities, #images, images-per-identity distribution
  - #graphs built vs expected (extraction failure rate)
  - node/edge count distribution (== minutiae per print)
  - split sizes
  - shape check on a few random graphs
"""
import argparse
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch


def _counts_per_dir(root, ext):
    out = {}
    if not os.path.isdir(root):
        return out
    for d in os.listdir(root):
        p = os.path.join(root, d)
        if os.path.isdir(p):
            out[d] = len(glob.glob(os.path.join(p, f"*.{ext}")))
    return out


def _dist(xs):
    if not xs:
        return {}
    a = np.array(xs, dtype=float)
    return {"min": float(a.min()), "p25": float(np.percentile(a, 25)),
            "median": float(np.median(a)), "mean": round(float(a.mean()), 2),
            "p75": float(np.percentile(a, 75)), "max": float(a.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--organized", default="data/SOCOFing_organized")
    ap.add_argument("--graphs", default="data/graphs")
    ap.add_argument("--splits", default="models/splits.json")
    ap.add_argument("--out", default="results/stage1_report.json")
    args = ap.parse_args()

    img_exts = ("bmp", "png", "jpg", "tif")
    img_counts = {}
    for e in img_exts:
        for k, v in _counts_per_dir(args.organized, e).items():
            img_counts[k] = img_counts.get(k, 0) + v
    graph_counts = _counts_per_dir(args.graphs, "pt")

    n_images = sum(img_counts.values())
    n_graphs = sum(graph_counts.values())
    fail = n_images - n_graphs
    fail_rate = round(100 * fail / n_images, 2) if n_images else None

    # node / edge distribution over a sample of graphs
    all_pt = glob.glob(os.path.join(args.graphs, "**", "*.pt"), recursive=True)
    sample = random.Random(0).sample(all_pt, min(2000, len(all_pt))) if all_pt else []
    nodes, edges, bad = [], [], []
    for p in sample:
        try:
            g = torch.load(p, weights_only=False)
            nodes.append(int(g.num_nodes))
            edges.append(int(g.edge_index.shape[1]))
            if g.x.shape[1] != 5:
                bad.append((p, tuple(g.x.shape)))
        except Exception as e:
            bad.append((p, str(e)))

    splits = {}
    if os.path.exists(args.splits):
        s = json.load(open(args.splits))
        splits = {k: len(s[k]) for k in ("train", "val", "test") if k in s}

    report = {
        "identities_organized": len(img_counts),
        "identities_with_graphs": len(graph_counts),
        "identities_ge2_samples": sum(1 for v in graph_counts.values() if v >= 2),
        "images_total": n_images,
        "graphs_total": n_graphs,
        "extraction_failures": fail,
        "extraction_failure_rate_pct": fail_rate,
        "images_per_identity": _dist(list(img_counts.values())),
        "graphs_per_identity": _dist(list(graph_counts.values())),
        "nodes_per_graph": _dist(nodes),
        "edges_per_graph": _dist(edges),
        "sampled_graphs_checked": len(sample),
        "bad_graphs": bad[:20],
        "splits": splits,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)

    print(json.dumps({k: v for k, v in report.items() if k != "bad_graphs"}, indent=2))
    if bad:
        print(f"\n{len(bad)} problem graphs (showing up to 20):")
        for b in bad[:20]:
            print("  ", b)
    print(f"\nFull report -> {args.out}")


if __name__ == "__main__":
    main()
