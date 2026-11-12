"""
Stage 9c: evaluation fixes for the SOCOFing results.

  closed-set sweep at the FULL 900-finger gallery (Stage 6/9 used 200), one random altered probe
  per finger, 95 % bootstrap CIs over queries, learned model re-run with 3 probe/crop seeds, and two
  window-placement modes:
      centred : window centred on a random minutia (what partiality_sweep.py did)
      random  : window placed uniformly over the minutiae extent (what the Stage 7 demo did)
  open-set test : half of the test fingers are enrolled, the other half are *unknown* (not in the
      gallery). A query is accepted if its best score is above a threshold. Reports DIR@FAR (correct
      top-1 among enrolled queries vs false accepts among unknown queries) and the AUC of
      known-vs-unknown by top score.

Score matrices (queries x 900 gallery) are cached in data/cache/eval_fixes/ so the open-set test
reuses the closed-set scores.

    python src/eval_fixes.py --modes centred --out results/stage9c/sweep_centred.json
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
from sklearn.metrics import roc_auc_score

import demo_core as D
from model_v2 import load_matcher
from partiality_sweep import crop_graph

LEVELS = [1.0, 0.75, 0.5, 0.25, 0.10]
CACHE = "data/cache/eval_fixes"
MODEL = "Stage 9 matcher (trained on SOCOFing only)"
NOTE = ""


def make_queries(ds, fingers, mode, frac, seed):
    """One random altered probe per finger, cut to `frac` of the print."""
    rng = random.Random(seed)
    out = []
    for f in fingers:
        g = ds.graph(f, rng.choice(ds.query_keys(f)))
        if frac >= 1.0:
            out.append(g)
        elif mode == "centred":
            out.append(crop_graph(g, frac, rng)[0])
        else:
            out.append(D.crop_window(g, frac, rng.random(), rng.random())[0])
    return out


def score_matrix(gal, queries, method):
    if method == "learned":
        return np.stack([gal.search_learned(q)[0] for q in queries])
    return np.stack([gal.search_classical(q) for q in queries])


def cached_scores(gal, ds, mode, frac, seed, method):
    path = os.path.join(CACHE, f"{mode}_{int(frac * 100)}_s{seed}_{method}.npy")
    if os.path.exists(path):
        return np.load(path)
    os.makedirs(CACHE, exist_ok=True)
    t = time.time()
    S = score_matrix(gal, make_queries(ds, gal.fingers, mode, frac, seed), method)
    np.save(path, S)
    print(f"    {os.path.basename(path)}: {time.time() - t:.0f}s", flush=True)
    return S


def ranks_from(S):
    """1-based rank of the true finger (column i for query i)."""
    true = np.diag(S)
    return 1 + (S > true[:, None]).sum(1)


def boot_ci(x, n=1000, seed=0):
    rng = np.random.RandomState(seed)
    x = np.asarray(x, float)
    m = [x[rng.randint(0, len(x), len(x))].mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def closed_metrics(S):
    r = ranks_from(S)
    out = {}
    for k in (1, 5):
        v = (r <= k).astype(float)
        lo, hi = boot_ci(v)
        out[f"rank{k}"] = float(v.mean())
        out[f"rank{k}_ci"] = [lo, hi]
    out["median_rank"] = float(np.median(r))
    return out


def open_set(S, split_seed, znorm):
    """Enrol half the fingers; queries of the other half are unknown."""
    n = S.shape[0]
    perm = np.random.RandomState(split_seed).permutation(n)
    enrolled, unknown = np.sort(perm[: n // 2]), np.sort(perm[n // 2:])
    G = S[:, enrolled]                                   # scores against the enrolled gallery only
    top = G.max(1)
    arg = G.argmax(1)
    if znorm:
        top = (top - G.mean(1)) / (G.std(1) + 1e-6)
    known_correct = np.zeros(n, bool)
    known_correct[enrolled] = arg[enrolled] == np.arange(len(enrolled))
    kn, un = top[enrolled], top[unknown]
    out = {"rank1_known": float(known_correct[enrolled].mean())}
    for far in (0.01, 0.10):
        thr = np.quantile(un, 1 - far)
        out[f"DIR@FAR={int(far * 100)}%"] = float((known_correct[enrolled] & (kn >= thr)).mean())
    out["auc_known_vs_unknown"] = float(roc_auc_score(
        np.r_[np.ones(len(kn)), np.zeros(len(un))], np.r_[kn, un]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="*", default=["centred", "random"])
    ap.add_argument("--levels", nargs="*", type=float, default=LEVELS)
    ap.add_argument("--seeds", type=int, default=3, help="probe/crop seeds for the learned model")
    ap.add_argument("--no-classical", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--packed-dir", default=None, help="graphs from another extractor, e.g. data/packed_leader")
    ap.add_argument("--checkpoint", default=None, help="Matcher checkpoint (default: Stage 9 inv_local_cont)")
    ap.add_argument("--cache-tag", default="", help="suffix for the score-matrix cache (needed when the "
                    "extractor or model differs from the default)")
    args = ap.parse_args()
    global CACHE
    CACHE = CACHE + args.cache_tag
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    ds = D.load_dataset("socofing", args.packed_dir)
    gal = D.Gallery(ds, load_matcher(args.checkpoint or D.model_path(MODEL)))
    print(f"{len(gal.fingers)} fingers in gallery", flush=True)
    res = {"gallery": len(gal.fingers), "seeds": args.seeds, "modes": {}}
    for mode in args.modes:
        res["modes"][mode] = {}
        for frac in args.levels:
            if frac >= 1.0 and mode != args.modes[0]:
                res["modes"][mode][str(int(frac * 100))] = res["modes"][args.modes[0]][str(int(frac * 100))]
                continue
            print(f"[{mode} {int(frac * 100)}%]", flush=True)
            entry = {"learned_seeds": []}
            for s in range(args.seeds):
                entry["learned_seeds"].append(closed_metrics(cached_scores(gal, ds, mode, frac, s, "learned")))
            r1 = [m["rank1"] for m in entry["learned_seeds"]]
            entry["learned"] = {**entry["learned_seeds"][0], "rank1_mean_over_seeds": float(np.mean(r1)),
                                "rank1_sd_over_seeds": float(np.std(r1, ddof=1)) if len(r1) > 1 else 0.0}
            Sl = cached_scores(gal, ds, mode, frac, 0, "learned")
            entry["open_set"] = {"learned": {}, "classical": {}}
            if not args.no_classical:
                Sc = cached_scores(gal, ds, mode, frac, 0, "classical")
                entry["classical"] = closed_metrics(Sc)
            for name, S in (("learned", Sl),) + ((("classical", Sc),) if not args.no_classical else ()):
                for z in (False, True):
                    runs = [open_set(S, sp, z) for sp in (0, 1)]
                    entry["open_set"][name]["z" if z else "raw"] = {
                        k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
            res["modes"][mode][str(int(frac * 100))] = entry
            l = entry["learned"]
            c = entry.get("classical", {"rank1": float("nan")})
            print(f"    Rank-1 learned {l['rank1'] * 100:.1f} (seeds {np.mean(r1) * 100:.1f}+-"
                  f"{l['rank1_sd_over_seeds'] * 100:.1f}) classical {c['rank1'] * 100:.1f}", flush=True)
            json.dump(res, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
