"""
Cross-dataset evaluation on FVC (zero-shot: models trained on SOCOFing only).

Protocol (per database, 100 fingers x 8 impressions, set A):
  gallery        = impression 1 of every finger (100 templates)
  identification = queries are impressions 2-8 (700), closed set -> Rank-1 / Rank-5
  verification   = FVC style: genuine = every ordered pair of different impressions of one
                   finger (query = later impression, 2,800 pairs); impostor = impression-1
                   queries vs impression-1 templates of other fingers (9,900 directed pairs)
  partial prints = the QUERY is cropped to a fraction of the print's own extent (bounding box
                   of its minutiae); templates stay whole. Same crops for every method.
Verification is reported raw and impostor-cohort normalised (each query's score minus the mean
of its scores against other fingers' templates, divided by their std) for every method.

    python src/eval_fvc.py --db fvc2002_db1 --models-v2 inv_local=models/stage9_inv_local_cont.pt
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
from sklearn.metrics import roc_curve, auc

from augment import build_graph_fast, GRAPH_NORM
from classical_match import graph_to_minutiae, hough_score, hough_score_line
from evaluate import identification_sim, embed_paths
from match import load_model
from model_v2 import load_matcher, encode_graphs, pair_scores

LEVELS = [1.0, 0.75, 0.5, 0.25, 0.10]
MIN_MINUTIAE = 3


def crop_bbox(data, frac, rng):
    """Square window covering `frac` of the print's own extent (bbox of its minutiae)."""
    x = data.x.numpy()
    px, py = x[:, 0] * GRAPH_NORM, x[:, 1] * GRAPH_NORM
    th, ty = np.arctan2(x[:, 2], x[:, 3]), x[:, 4]
    n = len(px)
    L = max(px.max() - px.min(), py.max() - py.min(), 1.0)
    side = math.sqrt(frac) * L
    best, best_n = None, -1
    for _ in range(25):
        c = rng.randrange(n)
        x0 = min(max(px[c] - side / 2, px.min()), max(px.max() - side, px.min()))
        y0 = min(max(py[c] - side / 2, py.min()), max(py.max() - side, py.min()))
        sel = np.where((px >= x0) & (px <= x0 + side) & (py >= y0) & (py <= y0 + side))[0]
        if len(sel) > best_n:
            best, best_n = sel, len(sel)
        if best_n >= MIN_MINUTIAE:
            break
    if best_n < 2:
        c = rng.randrange(n)
        best = np.argsort((px - px[c]) ** 2 + (py - py[c]) ** 2)[:2]
    return build_graph_fast(px[best], py[best], th[best], ty[best])


def eer_auc(genuine, impostor):
    y = np.r_[np.ones(len(genuine)), np.zeros(len(impostor))]
    s = np.r_[genuine, impostor]
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2), float(auc(fpr, tpr))


class Classical:
    name = "classical"
    fn = staticmethod(hough_score)

    def __init__(self, name=None, fn=None):
        if name:
            self.name = name
        if fn:
            self.fn = staticmethod(fn)

    def prepare(self, q_graphs, g_graphs):
        self.q = [graph_to_minutiae(g) for g in q_graphs]
        self.g = [graph_to_minutiae(g) for g in g_graphs]

    def score(self, qi, gi):
        return np.array([self.fn(self.q[a], self.g[b]) for a, b in zip(qi, gi)], dtype=np.float32)


class Learned:
    def __init__(self, name, model):
        self.name, self.model = name, model

    def prepare(self, q_graphs, g_graphs):
        self.eq = encode_graphs(self.model, q_graphs)
        self.eg = encode_graphs(self.model, g_graphs)

    def score(self, qi, gi):
        return pair_scores(self.model, self.eq, self.eg, qi, gi).numpy()


class OldGCN:
    """Stage 3 TripletGCN (absolute-coordinate features, one pooled vector)."""

    def __init__(self, name, path):
        self.name = name
        self.model, _ = load_model(path, "cpu")

    def prepare(self, q_graphs, g_graphs):
        self.eq = embed_paths(self.model, q_graphs, "cpu")
        self.eg = embed_paths(self.model, g_graphs, "cpu")

    def score(self, qi, gi):
        return (self.eq[torch.as_tensor(qi)] * self.eg[torch.as_tensor(gi)]).sum(1).numpy()


def evaluate_level(scorer, q_graphs, g_graphs, fid, imp, n_f):
    """fid/imp: finger and impression index of every print (same order as the graph lists)."""
    N = len(fid)
    idx = {(fid[k], imp[k]): k for k in range(N)}
    fingers = sorted(set(fid))
    g1 = [idx[(f, 1)] for f in fingers]                      # gallery templates (impression 1)
    f_pos = {f: i for i, f in enumerate(fingers)}

    # pair lists ------------------------------------------------------------------------
    id_q = [k for k in range(N) if imp[k] > 1]
    id_pairs = [(k, g1[j]) for k in id_q for j in range(n_f)]
    imp_q = [idx[(f, 1)] for f in fingers]
    imp_pairs = [(idx[(a, 1)], g1[j]) for a in fingers for j in range(n_f) if fingers[j] != a]
    gen_pairs = [(idx[(f, j)], idx[(f, i)]) for f in fingers for j in range(2, 9)
                 for i in range(1, j) if (f, j) in idx and (f, i) in idx]
    pairs = id_pairs + imp_pairs + gen_pairs
    scorer.prepare(q_graphs, g_graphs)
    t0 = time.time()
    sc = scorer.score([a for a, _ in pairs], [b for _, b in pairs])
    dt = time.time() - t0

    S_id = sc[:len(id_pairs)].reshape(len(id_q), n_f)
    imp_sc = sc[len(id_pairs):len(id_pairs) + len(imp_pairs)]
    gen_sc = sc[len(id_pairs) + len(imp_pairs):]

    pf = torch.tensor([f_pos[fid[k]] for k in id_q])
    ident = identification_sim(torch.from_numpy(S_id), pf, torch.arange(n_f))

    # raw verification
    eer, au = eer_auc(gen_sc, imp_sc)
    # impostor-cohort normalisation: stats of each query's scores vs OTHER fingers' templates
    coh = {}
    for k_row, k in enumerate(id_q):
        row = np.delete(S_id[k_row], f_pos[fid[k]])
        coh[k] = (row.mean(), row.std() + 1e-6)
    imp_by_q = {}
    for (a, _), s in zip(imp_pairs, imp_sc):
        imp_by_q.setdefault(a, []).append(s)
    for a, v in imp_by_q.items():
        coh[a] = (np.mean(v), np.std(v) + 1e-6)
    z_imp = np.array([(s - coh[a][0]) / coh[a][1] for (a, _), s in zip(imp_pairs, imp_sc)])
    z_gen = np.array([(s - coh[a][0]) / coh[a][1] for (a, _), s in zip(gen_pairs, gen_sc)])
    eer_z, auc_z = eer_auc(z_gen, z_imp)
    return {"rank1": ident["rank1"], "rank5": ident["rank5"], "eer": eer, "auc": au,
            "eer_z": eer_z, "auc_z": auc_z, "n_id_queries": len(id_q), "n_genuine": len(gen_pairs),
            "n_impostor": len(imp_pairs), "sec": dt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="name of data/fvc_packed/<db>.pt, e.g. fvc2002_db1")
    ap.add_argument("--packed-dir", default="data/fvc_packed")
    ap.add_argument("--models-v2", nargs="*", default=[])
    ap.add_argument("--old", nargs="*", default=[], help="Stage 3 TripletGCN checkpoints name=path")
    ap.add_argument("--no-classical", action="store_true")
    ap.add_argument("--classical-line", action="store_true",
                    help="also score the orientation-mod-pi variant of the classical matcher")
    ap.add_argument("--levels", nargs="*", type=float, default=LEVELS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--finger-range", default=None, help="e.g. 51-100: evaluate only these fingers "
                    "(the held-out fingers when fine-tuning on fingers 1-50)")
    ap.add_argument("--tag", default="", help="suffix for the output json name")
    ap.add_argument("--outdir", default="results/stage8")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    packed = torch.load(os.path.join(args.packed_dir, args.db + ".pt"), weights_only=False)
    fingers = sorted(f for f in packed if 1 in packed[f] and len(packed[f]) >= 2)
    if args.finger_range:
        lo, hi = (int(v) for v in args.finger_range.split("-"))
        fingers = [f for f in fingers if lo <= f <= hi]
    keys = [(f, i) for f in fingers for i in sorted(packed[f])]
    fid, imp = [k[0] for k in keys], [k[1] for k in keys]
    g_graphs = [packed[f][i] for f, i in keys]
    n_f = len(fingers)
    print(f"{args.db}: {n_f} fingers, {len(keys)} prints, minutiae/print median "
          f"{np.median([g.num_nodes for g in g_graphs]):.0f}", flush=True)

    scorers = [] if args.no_classical else [Classical()]
    if args.classical_line:
        scorers.append(Classical("classical_line", hough_score_line))
    scorers += [OldGCN(n, p) for n, p in (s.split("=", 1) for s in args.old)]
    scorers += [Learned(n, load_matcher(p)) for n, p in (s.split("=", 1) for s in args.models_v2)]

    res = {"db": args.db, "fingers": n_f, "prints": len(keys), "levels": {}}
    for frac in args.levels:
        rng = random.Random(args.seed)
        q_graphs = g_graphs if frac >= 1.0 else [crop_bbox(g, frac, rng) for g in g_graphs]
        entry = {"minutiae_per_query_mean": float(np.mean([g.num_nodes for g in q_graphs]))}
        for sc in scorers:
            entry[sc.name] = evaluate_level(sc, q_graphs, g_graphs, fid, imp, n_f)
        res["levels"][str(int(frac * 100))] = entry
        line = "  ".join(f"{sc.name}={entry[sc.name]['rank1'] * 100:.1f}" for sc in scorers)
        print(f"[{int(frac * 100):3d}%] minutiae/query={entry['minutiae_per_query_mean']:.1f}  R1: {line}",
              flush=True)
        json.dump(res, open(os.path.join(args.outdir, f"{args.db}{args.tag}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
