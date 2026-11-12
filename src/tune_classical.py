"""
Tune the classical Hough matcher's tolerances on FVC SET B (fingers 101-110 of every database,
the part FVC provides for tuning). Set A (the evaluation fingers) is never touched.

Objective: pooled verification AUC over the 8 databases on full prints (genuine = every ordered
pair of different impressions of one finger; impostor = same-index impressions of different
fingers); Rank-1 (gallery = impression 1, queries = impressions 2-8) is reported alongside.

    python src/tune_classical.py --out results/stage9c/classical_tuned.json
"""
import argparse
import functools
import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from classical_match import graph_to_minutiae, hough_score

DBS = [f"fvc{y}_db{d}" for y in (2002, 2004) for d in (1, 2, 3, 4)]
DEFAULT = dict(xy_bin=8.0, ang_bin=0.35, xy_tol=12.0, ang_tol=0.5)


def load(dirname):
    out = {}
    for db in DBS:
        p = torch.load(os.path.join(dirname, db + ".pt"), weights_only=False)
        out[db] = {f: {i: graph_to_minutiae(g) for i, g in p[f].items()} for f in p}
    return out


def build_pairs(data):
    """-> list of (db, query minutiae, gallery minutiae, kind) with kind in {gen, imp, id:<finger>}"""
    gen, imp, ident = [], [], []
    for db, d in data.items():
        fingers = sorted(d)
        for f in fingers:
            for j in sorted(d[f]):
                for i in sorted(d[f]):
                    if i < j:
                        gen.append((d[f][j], d[f][i]))
        for a in fingers:
            for b in fingers:
                if a != b:
                    for k in (1, 2):
                        if k in d[a] and k in d[b]:
                            imp.append((d[a][k], d[b][k]))
        for a in fingers:
            for j in sorted(d[a]):
                if j > 1:
                    ident.append((db, a, d[a][j]))
    return gen, imp, ident


def score(params, gen, imp, ident, data):
    fn = functools.partial(hough_score, **params)
    g = np.array([fn(q, t) for q, t in gen])
    i = np.array([fn(q, t) for q, t in imp])
    auc = roc_auc_score(np.r_[np.ones(len(g)), np.zeros(len(i))], np.r_[g, i])
    hit = 0
    for db, f, q in ident:
        fingers = sorted(data[db])
        s = [fn(q, data[db][h][1]) for h in fingers]
        hit += fingers[int(np.argmax(s))] == f
    return float(auc), hit / len(ident)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/fvc_packed_b")
    ap.add_argument("--out", default="results/stage9c/classical_tuned.json")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    data = load(args.dir)
    gen, imp, ident = build_pairs(data)
    print(f"{len(gen)} genuine, {len(imp)} impostor pairs, {len(ident)} identification queries", flush=True)

    base = score(DEFAULT, gen, imp, ident, data)
    print(f"default {DEFAULT}: AUC {base[0]:.4f}  Rank-1 {base[1]:.3f}", flush=True)
    grid = dict(xy_tol=[8, 12, 16, 20, 26], ang_tol=[0.35, 0.5, 0.7, 0.9],
                xy_bin=[6.0, 10.0, 14.0], ang_bin=[0.25, 0.4])
    rows, t0 = [], time.time()
    for vals in itertools.product(*grid.values()):
        p = dict(zip(grid, vals))
        auc, r1 = score(p, gen, imp, ident, data)
        rows.append({**p, "auc": auc, "rank1": r1})
        print(f"  {p}  AUC {auc:.4f}  R1 {r1:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    best = max(rows, key=lambda r: (round(r["auc"], 4), r["rank1"]))
    params = {k: best[k] for k in grid}
    json.dump({"default": {"params": DEFAULT, "auc": base[0], "rank1": base[1]},
               "tuned": {"params": params, "auc": best["auc"], "rank1": best["rank1"]},
               "grid": rows}, open(args.out, "w"), indent=1)
    print(f"best {params}: AUC {best['auc']:.4f}  Rank-1 {best['rank1']:.3f}")


if __name__ == "__main__":
    main()
