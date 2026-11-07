"""
Full-protocol evaluation (all test fingers, all altered probes) for Stage 9 Matcher models.

Reports Rank-1/5, CMC, EER/AUC raw and with per-probe test-normalisation, and the
per-alteration Rank-1 breakdown. The classical matcher is O(fingers^2) and is only run on
the 200-finger subset in partiality_sweep.py; this script is for the learned models.

    python src/evaluate_v2.py --checkpoint models/stage9_inv_local.pt --outdir results/stage9/full_inv_local
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from evaluate import (collect, identification_sim, verification_sim, per_tag_rank1_sim)
from model_v2 import load_matcher, score_matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--splits", default="models/splits.json")
    ap.add_argument("--packed", default="data/packed/test.pt")
    ap.add_argument("--limit-fingers", type=int, default=0)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    model = load_matcher(args.checkpoint)
    fingers = json.load(open(args.splits))["test"]
    if args.limit_fingers:
        fingers = fingers[:args.limit_fingers]
    gallery, probes = collect("data/graphs", fingers, packed=args.packed)
    ids = list(gallery)
    ix = {f: i for i, f in enumerate(ids)}
    t0 = time.time()
    sim = score_matrix(model, [g for g, _, _ in probes], [gallery[f] for f in ids])
    print(f"scored {sim.shape[0]} probes x {sim.shape[1]} gallery in {time.time() - t0:.0f}s")
    pf, gf = torch.tensor([ix[f] for _, f, _ in probes]), torch.arange(len(ids))
    tags = [t for _, _, t in probes]

    ident = identification_sim(sim, pf, gf)
    ver, _ = verification_sim(sim, pf, gf)
    sim_n = (sim - sim.mean(1, keepdim=True)) / sim.std(1, keepdim=True).clamp_min(1e-6)
    ver_n, _ = verification_sim(sim_n, pf, gf)
    tagr = per_tag_rank1_sim(sim, pf, gf, tags)
    out = {"checkpoint": args.checkpoint, "test_fingers": len(ids), "probes": len(probes),
           "rank1": ident["rank1"], "rank5": ident["rank5"], "cmc": ident["cmc"],
           "eer": ver["eer"], "auc": ver["auc"], "eer_tnorm": ver_n["eer"], "auc_tnorm": ver_n["auc"],
           "per_alteration_rank1": tagr}
    json.dump(out, open(os.path.join(args.outdir, "eval_metrics.json"), "w"), indent=2)
    print(f"Rank-1 {ident['rank1']*100:.2f}%  Rank-5 {ident['rank5']*100:.2f}%  "
          f"EER {ver['eer']*100:.2f}% (tnorm {ver_n['eer']*100:.2f}%)  "
          f"AUC {ver['auc']:.4f} (tnorm {ver_n['auc']:.4f})")
    for t, v in tagr.items():
        print(f"  {t:20s} {v['rank1']*100:5.1f}%  (n={v['n']})")


if __name__ == "__main__":
    main()
