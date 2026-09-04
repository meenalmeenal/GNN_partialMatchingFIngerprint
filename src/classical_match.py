"""
Classical minutiae matcher - the baseline the GNN has to beat.

Algorithm (generalized Hough transform alignment, standard for minutiae matching):
  1. For every same-type minutia pair (probe i, gallery j), compute the rigid
     transform (dx, dy, dtheta) that would map i onto j.
  2. Vote: bin all candidate transforms: the bin with the most votes is the
     consensus alignment between the two prints.
  3. Apply that transform to every probe minutia; greedily match to the nearest
     gallery minutia within tolerance (one-to-one).
  4. Score = Dice coefficient = 2*matched / (n_probe + n_gallery).

Minutiae are reconstructed directly from the already-built PyG graphs (node features
are [x/size, y/size, sin, cos, type]) - no need to re-run image extraction.

    python src/classical_match.py --splits models/splits.json --limit-fingers 150
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from evaluate import collect, GALLERY_TAG
from minutiae import EXTRACT_CFG

IMG_SIZE = EXTRACT_CFG["size"]


def graph_to_minutiae(data) -> np.ndarray:
    """-> array [n, 4] columns (x, y, theta, type)"""
    x = data.x.numpy()
    px = x[:, 0] * IMG_SIZE
    py = x[:, 1] * IMG_SIZE
    theta = np.arctan2(x[:, 2], x[:, 3])
    typ = x[:, 4]
    return np.stack([px, py, theta, typ], axis=1)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def hough_score(P: np.ndarray, G: np.ndarray, xy_bin=8.0, ang_bin=0.35,
               xy_tol=12.0, ang_tol=0.5) -> float:
    """Dice-coefficient match score between two minutiae sets, rotation+translation invariant."""
    n, m = len(P), len(G)
    if n == 0 or m == 0:
        return 0.0

    # candidate transforms from same-type pairs only
    same_type = P[:, 3][:, None] == G[:, 3][None, :]
    if not same_type.any():
        pi, gj = np.repeat(np.arange(n), m), np.tile(np.arange(m), n)
    else:
        pi, gj = np.where(same_type)

    dtheta = _wrap(G[gj, 2] - P[pi, 2])
    c, s = np.cos(dtheta), np.sin(dtheta)
    rx = P[pi, 0] * c - P[pi, 1] * s
    ry = P[pi, 0] * s + P[pi, 1] * c
    dx = G[gj, 0] - rx
    dy = G[gj, 1] - ry

    bx = np.round(dx / xy_bin).astype(np.int64)
    by = np.round(dy / xy_bin).astype(np.int64)
    ba = np.round(dtheta / ang_bin).astype(np.int64)
    key = (bx + 4096) * (8192 * 64) + (by + 4096) * 64 + (ba + 32)
    uniq, counts = np.unique(key, return_counts=True)
    best = uniq[np.argmax(counts)]
    sel = key == best
    tx, ty, tth = float(np.median(dx[sel])), float(np.median(dy[sel])), float(np.median(dtheta[sel]))

    # apply the consensus transform to ALL probe points, greedily match to gallery
    c, s = np.cos(tth), np.sin(tth)
    qx = P[:, 0] * c - P[:, 1] * s + tx
    qy = P[:, 0] * s + P[:, 1] * c + ty
    qth = _wrap(P[:, 2] + tth)

    d2 = (qx[:, None] - G[None, :, 0]) ** 2 + (qy[:, None] - G[None, :, 1]) ** 2
    dang = np.abs(_wrap(qth[:, None] - G[None, :, 2]))
    type_ok = P[:, 3][:, None] == G[None, :, 3]
    ok = (d2 <= xy_tol ** 2) & (dang <= ang_tol) & type_ok
    d2masked = np.where(ok, d2, np.inf)

    matched, used_g = 0, set()
    order = np.argsort(d2masked.min(axis=1))
    for i in order:
        if not np.isfinite(d2masked[i]).any():
            continue
        j = int(np.argmin(d2masked[i]))
        if j in used_g:
            row = d2masked[i].copy()
            row[list(used_g)] = np.inf
            if not np.isfinite(row).any():
                continue
            j = int(np.argmin(row))
        used_g.add(j)
        matched += 1

    return 2.0 * matched / (n + m)


def evaluate_classical(gallery_m, probes_m, pf, fingers, max_rank=5, seed=0):
    """gallery_m: list of minutiae arrays aligned with `fingers`; probes_m: list aligned with pf."""
    n_g = len(gallery_m)
    scores = np.zeros((len(probes_m), n_g), dtype=np.float32)
    t0 = time.time()
    for i, P in enumerate(probes_m):
        for j, G in enumerate(gallery_m):
            scores[i, j] = hough_score(P, G)
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  probe {i + 1}/{len(probes_m)}  ({el:.0f}s, {el / (i + 1):.2f}s/probe)")

    order = np.argsort(-scores, axis=1)
    ranked = np.array(fingers)[order]
    hits = ranked == np.array(fingers)[np.array(pf)][:, None]
    rank_of = hits.argmax(axis=1)
    found = hits.any(axis=1)
    cmc = [float((found & (rank_of < r)).mean()) for r in range(1, max_rank + 1)]

    genuine = scores[np.arange(len(pf)), pf]
    rng = np.random.default_rng(seed)
    impostor = []
    for i in range(len(pf)):
        neg = [j for j in range(n_g) if j != pf[i]]
        for j in rng.choice(neg, size=min(10, len(neg)), replace=False):
            impostor.append(scores[i, j])
    from sklearn.metrics import roc_curve, auc
    y = np.array([1] * len(genuine) + [0] * len(impostor))
    s = np.concatenate([genuine, np.array(impostor)])
    fpr, tpr, thr = roc_curve(y, s)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[idx] + fnr[idx]) / 2)
    return {"rank1": cmc[0], "rank5": cmc[min(4, max_rank - 1)], "cmc": cmc,
            "eer": eer, "auc": float(auc(fpr, tpr))}, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="models/splits.json")
    ap.add_argument("--packed", default="data/packed/test.pt")
    ap.add_argument("--graph_dir", default="data/graphs")
    ap.add_argument("--limit-fingers", type=int, default=150,
                    help="classical matching is O(fingers^2); evaluate on a subset")
    ap.add_argument("--outdir", default="results/stage5")
    args = ap.parse_args()

    sp = json.load(open(args.splits))
    fingers = sp["test"][:args.limit_fingers]
    packed = args.packed if os.path.exists(args.packed) else None
    gallery, probes = collect(args.graph_dir, fingers, packed=packed)
    fingers = list(gallery.keys())
    fidx = {f: i for i, f in enumerate(fingers)}
    print(f"classical baseline on {len(fingers)} fingers, {len(probes)} probes")

    gallery_m = [graph_to_minutiae(gallery[f]) for f in fingers]
    probes_m = [graph_to_minutiae(g) for g, _, _ in probes]
    pf = [fidx[f] for _, f, _ in probes]
    tags = [t for _, _, t in probes]

    metrics, scores = evaluate_classical(gallery_m, probes_m, pf, fingers)

    per_tag = {}
    scores_arr = np.array(scores)
    top = np.array(fingers)[scores_arr.argmax(axis=1)]
    correct = top == np.array(fingers)[np.array(pf)]
    for t in sorted(set(tags)):
        m = np.array([x == t for x in tags])
        per_tag[t] = {"rank1": float(correct[m].mean()), "n": int(m.sum())}

    os.makedirs(args.outdir, exist_ok=True)
    out = {"n_fingers": len(fingers), "n_probes": len(probes), **metrics,
          "per_alteration_rank1": per_tag}
    json.dump(out, open(os.path.join(args.outdir, "classical_metrics.json"), "w"), indent=2)

    print(f"\nRank-1: {metrics['rank1']*100:.2f}%   Rank-5: {metrics['rank5']*100:.2f}%")
    print(f"EER:    {metrics['eer']*100:.2f}%   ROC AUC: {metrics['auc']:.4f}")
    for t, v in per_tag.items():
        print(f"  {t:20s} {v['rank1']*100:5.1f}%  (n={v['n']})")
    print(f"\nwrote {args.outdir}/classical_metrics.json")


if __name__ == "__main__":
    main()
