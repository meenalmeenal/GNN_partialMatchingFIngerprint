"""
Calibrate the demo's "is this finger enrolled at all?" decision.

For each gallery and print area: enrol a random half of the fingers, treat the other half as
unknown, score every probe against the enrolled half, z-score its best score over the enrolled
gallery, and take the z thresholds that let 10 % / 1 % of unknown probes through. Averaged over 4
random enrolment splits. The app interpolates these thresholds at the chosen print area.

    python src/calibrate_reject.py          # -> results/stage9e/reject_thresholds.json
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import demo_core as D
from model_v2 import load_matcher

AREAS = [1.0, 0.75, 0.5, 0.25, 0.10]
SPLITS = 4


def queries(ds, fingers, area, seed):
    rng = random.Random(seed)
    qs, owner = [], []
    for f in fingers:
        keys = ds.query_keys(f) if ds.real else [rng.choice(ds.query_keys(f))]   # FVC: every impression
        for k in keys:
            q, _, _ = D.crop_window(ds.graph(f, k), area, rng.random(), rng.random())
            qs.append(q)
            owner.append(fingers.index(f))
    return qs, np.array(owner)


def calibrate(ds, gal):
    res = {}
    for area in AREAS:
        qs, owner = queries(ds, gal.fingers, area, 0)
        S = np.stack([gal.search_learned(q)[0] for q in qs])            # [Q, G]
        t10, t1, d10, d1 = [], [], [], []
        for sp in range(SPLITS):
            perm = np.random.RandomState(sp).permutation(len(gal.fingers))
            enrolled = np.sort(perm[: len(perm) // 2])
            known = np.isin(owner, enrolled)
            G = S[:, enrolled]
            z = (G.max(1) - G.mean(1)) / (G.std(1) + 1e-6)
            pos = {int(g): i for i, g in enumerate(enrolled)}
            correct = np.array([known[i] and int(G[i].argmax()) == pos[int(owner[i])] for i in range(len(owner))])
            zu = z[~known]
            for far, tl, dl in ((0.10, t10, d10), (0.01, t1, d1)):
                thr = np.quantile(zu, 1 - far)
                tl.append(thr)
                dl.append(float((correct & (z >= thr))[known].mean()))
        res[str(int(area * 100))] = {"z_far10": float(np.mean(t10)), "z_far1": float(np.mean(t1)),
                                     "dir_far10": float(np.mean(d10)), "dir_far1": float(np.mean(d1)),
                                     "n_queries": int(len(qs))}
        print(f"    {int(area * 100):3d}%  z@FAR10% {np.mean(t10):.2f}  z@FAR1% {np.mean(t1):.2f}  "
              f"DIR {np.mean(d10) * 100:.0f}% / {np.mean(d1) * 100:.0f}%", flush=True)
    return res


if __name__ == "__main__":
    out = {}
    for key in D.dataset_choices():
        model = load_matcher(D.model_path(D.SOCO_MODEL if key == "socofing" else D.FVC_MODEL))
        ds = D.load_dataset(key)
        print(key, flush=True)
        out[key] = calibrate(ds, D.Gallery(ds, model))
    os.makedirs("results/stage9e", exist_ok=True)
    json.dump(out, open("results/stage9e/reject_thresholds.json", "w"), indent=1)
