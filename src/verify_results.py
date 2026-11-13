"""
Check the numbers documented in README.md / results/stage*_notes.md against the result files on disk.

  python src/verify_results.py                 # committed artefacts: must match the documented numbers (tol 0.15 points)
  python src/verify_results.py --tol 4.0       # after re-running the pipeline: retraining moves numbers by seed noise
                                               # (Stage 10: sd about 1-2 points on FVC Rank-1), so allow a few points

Exit code 0 = every check passed. Every check reads a json produced by the scripts, never a hard-coded copy.
"""
import argparse
import json
import os
import sys

import numpy as np

REAL = [f"fvc{y}_db{d}" for y in (2002, 2004) for d in (1, 2, 3)]


def fvc_mean(folder, key, area, metric="rank1"):
    return 100 * np.mean([json.load(open(f"{folder}/{db}.json"))["levels"][area][key][metric] for db in REAL])


def sweep(path, meth, area, metric="rank1"):
    return 100 * json.load(open(path))["modes"]["centred"][area][meth][metric]


# (description, function returning the measured value, documented value)
CHECKS = [
    ("Stage 8  FVC classical, old extractor, 100 %", lambda: fvc_mean("results/stage8/heldout", "classical", "100"), 17.1),
    ("Stage 8  FVC fine-tuned learned, old extractor, 100 %", lambda: fvc_mean("results/stage8/heldout", "inv_local_fvcft", "100"), 21.4),
    ("Stage 9c SOCOFing learned, old extractor, 50 % (900 gallery)", lambda: sweep("results/stage9c/sweep_centred.json", "learned", "50"), 42.7),
    ("Stage 9c SOCOFing classical, old extractor, 25 %", lambda: sweep("results/stage9c/sweep_centred.json", "classical", "25"), 13.2),
    ("Stage 9c FVC classical tuned, old extractor, 100 %", lambda: fvc_mean("results/stage9c/fvc_tuned", "classical_tuned", "100"), 21.4),
    ("Stage 9d FVC classical default, LEADER, 100 %", lambda: fvc_mean("results/stage9d/fvc_oldmodel", "classical", "100"), 74.7),
    ("Stage 9d FVC classical tuned, LEADER, 100 %", lambda: fvc_mean("results/stage9d/fvc_tuned", "classical_tuned", "100"), 78.8),
    ("Stage 9d FVC old model on LEADER graphs, 100 %", lambda: fvc_mean("results/stage9d/fvc_oldmodel", "inv_local_oldmodel", "100"), 54.4),
    ("Stage 9d FVC retrained learned, 100 %", lambda: fvc_mean("results/stage9d/fvc_newmodel", "inv_local_leader", "100"), 76.3),
    ("Stage 9d FVC retrained learned, 10 %", lambda: fvc_mean("results/stage9d/fvc_newmodel", "inv_local_leader", "10"), 41.1),
    ("Stage 9d FVC fine-tuned learned, 100 %", lambda: fvc_mean("results/stage9d/fvc_ft", "inv_local_leader_fvcft", "100"), 88.2),
    ("Stage 9d FVC fine-tuned learned EER, 100 %", lambda: fvc_mean("results/stage9d/fvc_ft", "inv_local_leader_fvcft", "100", "eer_z"), 7.4),
    ("Stage 9d SOCOFing learned, LEADER, 100 %", lambda: sweep("results/stage9d/sweep_centred_leader.json", "learned", "100"), 88.4),
    ("Stage 9d SOCOFing learned, LEADER, 25 %", lambda: sweep("results/stage9d/sweep_centred_leader.json", "learned", "25"), 60.8),
    ("Stage 9d SOCOFing classical, LEADER, 10 %", lambda: sweep("results/stage9d/sweep_centred_leader.json", "classical", "10"), 6.1),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=0.15, help="allowed |measured - documented| in percentage points")
    args = ap.parse_args()
    bad = 0
    for desc, fn, want in CHECKS:
        try:
            got = fn()
        except Exception as e:                       # missing file etc.
            print(f"  MISSING  {desc}: {type(e).__name__}: {e}")
            bad += 1
            continue
        ok = abs(got - want) <= args.tol
        bad += not ok
        print(f"  {'ok     ' if ok else 'DIFFERS'}  {desc}: measured {got:.1f}, documented {want:.1f}")
    print(f"{len(CHECKS) - bad}/{len(CHECKS)} checks passed (tolerance {args.tol} points)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
