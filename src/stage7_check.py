"""
Numbers behind the Stage 7 demo: Rank-1 / Rank-5 of exactly what the app does (partial window
cut from one probe, searched against the WHOLE gallery) for the learned and classical matchers.

  SOCOFing : gallery = all 900 held-out fingers; one random altered probe per query finger
  FVC      : gallery = the 50 held-out fingers (51-100) of each database; one random impression
             2-8 per finger (fine-tuned model)

    python src/stage7_check.py --n-socofing 300 --out results/stage7/demo_numbers.json
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import demo_core as D
from model_v2 import load_matcher

AREAS = [1.0, 0.5, 0.25]
SOCO_MODEL = "Stage 9 matcher (trained on SOCOFing only)"
FVC_MODEL = "Stage 8 matcher, fine-tuned on real FVC prints"


def run(ds, model, n, seed):
    gal = D.Gallery(ds, model)
    rng = random.Random(seed)
    fingers = rng.sample(gal.fingers, min(n, len(gal.fingers))) if n else list(gal.fingers)
    out = {}
    for area in AREAS:
        r = {"learned": [], "classical": []}
        rng = random.Random(seed + int(area * 100))
        for f in fingers:
            key = rng.choice(ds.query_keys(f))
            q, _, _ = D.crop_window(ds.graph(f, key), area, rng.random(), rng.random())
            i = gal.fingers.index(f)
            r["learned"].append(D.rank_of(gal.search_learned(q)[0], i))
            r["classical"].append(D.rank_of(gal.search_classical(q), i))
        out[str(int(area * 100))] = {m: {"rank1": float(np.mean(np.array(v) == 1)),
                                         "rank5": float(np.mean(np.array(v) <= 5)),
                                         "median_rank": float(np.median(v))}
                                     for m, v in r.items()}
        out[str(int(area * 100))]["n_queries"] = len(fingers)
        out[str(int(area * 100))]["gallery"] = len(gal.fingers)
        print(f"  {int(area * 100):3d}%  learned R1 {out[str(int(area * 100))]['learned']['rank1'] * 100:5.1f}  "
              f"classical R1 {out[str(int(area * 100))]['classical']['rank1'] * 100:5.1f}  "
              f"(n={len(fingers)}, gallery {len(gal.fingers)})", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-socofing", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/stage7/demo_numbers.json")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    res = {}
    print("SOCOFing", flush=True)
    res["socofing"] = run(D.load_dataset("socofing"), load_matcher(D.model_path(SOCO_MODEL)),
                          args.n_socofing, args.seed)
    ft = load_matcher(D.model_path(FVC_MODEL))
    for name in [k for k in D.dataset_choices() if k != "socofing"]:
        print(name, flush=True)
        res[name] = run(D.load_dataset(name), ft, 0, args.seed)
    json.dump(res, open(args.out, "w"), indent=2)

    real = [k for k in res if k != "socofing" and not k.endswith("db4")]
    lines = ["| setting | 100 % | 50 % | 25 % |", "|---|---|---|---|"]
    def row(label, keys, m):
        cells = [f"{np.mean([res[k][a][m]['rank1'] for k in keys]) * 100:.1f}" for a in ("100", "50", "25")]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    row("SOCOFing (900-print gallery), learned", ["socofing"], "learned")
    row("SOCOFing (900-print gallery), classical", ["socofing"], "classical")
    row(f"FVC real sensors (6 DBs, 50-print galleries), fine-tuned learned", real, "learned")
    row(f"FVC real sensors (6 DBs, 50-print galleries), classical", real, "classical")
    open(os.path.join(os.path.dirname(args.out), "demo_numbers.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
