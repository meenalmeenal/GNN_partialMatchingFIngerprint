"""
Stage 9d: do pretrained extractors (pyfing LEADER / SBMEX) find minutiae that are more repeatable
across independent impressions than our own extractor?  Same harness as Stage 8 (`fvc_dev.py`):
FVC SET B only (fingers 101-110), orientation-free rigid registration AUC for genuine vs impostor
pairs of minutiae sets, per database, with ONE fixed ridge-period scale per database.
All coordinates are mapped to the same normalised image space (ridge period ~8.2 px) before scoring.

    KERAS_BACKEND=torch python src/extractor_dev.py --variants ours leader_norm leader_raw
"""
import argparse
import glob
import os
import random
import sys
import time

os.environ.setdefault("KERAS_BACKEND", "torch")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import fvc_pipeline as F
from fvc_dev import DEV_DBS, ROOT, pos_only_score, db_scale

torch.set_grad_enabled(False)
NATIVE_DPI = {("FVC2002", 2): 569, ("FVC2004", 3): 512}


def to_arr(mins, scale=1.0, min_quality=0.0):
    rows = [[m.x * scale, m.y * scale, m.direction, 0 if m.type == "E" else 1, m.quality]
            for m in mins if m.type in ("E", "B") and m.quality >= min_quality]
    return np.array(rows, dtype=np.float64).reshape(-1, 5)


def make_extractors():
    import pyfing as pf
    alg = {}

    def ours(gray, scale, key):
        mins = F.minutiae_from_gray(gray, F.calibrate_target_period(), scale)[0]
        return np.array([[m.x, m.y, m.angle, m.type] for m in mins], float).reshape(-1, 4)

    def norm(gray, scale):
        return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    def leader_norm(dpi, q):
        def f(gray, scale, key):
            return to_arr(pf.minutiae_extraction(norm(gray, scale), dpi=dpi, method="LEADER"), 1.0, q)
        return f

    def leader_raw(q):
        def f(gray, scale, key):
            dpi = NATIVE_DPI.get(key, 500)
            return to_arr(pf.minutiae_extraction(gray, dpi=dpi, method="LEADER"), scale, q)
        return f

    def sbmex_norm(gray, scale, key):
        img = norm(gray, scale)
        seg = pf.fingerprint_segmentation(img)
        ori = pf.orientation_field_estimation(img, seg)
        fre = pf.frequency_estimation(img, ori, seg)
        enh = pf.fingerprint_enhancement(img, ori, fre, seg, method="GBFEN")
        return to_arr(pf.minutiae_extraction(enh, method="SBMEX", segmentation_mask=seg), 1.0, 0.0)

    low = pf.Leader(pf.LeaderParameters(minutia_quality_threshold=0.2))

    def leader_low(gray, scale, key):          # low threshold; filtered by quality afterwards
        return to_arr(low.run(norm(gray, scale), 500), 1.0, 0.0)

    return {"ours": ours, "leader_low": leader_low,
            "leader_norm": leader_norm(500, 0.0), "leader_norm_dpi433": leader_norm(433, 0.0),
            "leader_norm_q5": leader_norm(500, 0.5), "leader_raw": leader_raw(0.0),
            "sbmex_norm": sbmex_norm}


def run_thresholds(variant, fn, target, dbs, thresholds, seed=0):
    """Extract once at a low quality threshold, then score each higher threshold."""
    cache = {}
    for y, db in dbs:
        paths = sorted(glob.glob(f"{ROOT}/{y}/Dbs/Db{db}_b/*.tif"))
        scale, _ = db_scale(paths, target, 4)
        cache[(y, db)] = {F.parse_name(p): fn(cv2.imread(p, cv2.IMREAD_GRAYSCALE), scale, (y, db)) for p in paths}
    for t in thresholds:
        aucs, cnts = [], []
        for (y, db), d in cache.items():
            by = {}
            for (f, i), a in d.items():
                by.setdefault(f, {})[i] = a[a[:, 4] >= t][:, :4] if len(a) else a[:, :4]
            rng = random.Random(seed)
            gen = [(by[f][i], by[f][j]) for f in by for i in by[f] for j in by[f] if i < j]
            fs = list(by)
            imp = []
            while len(imp) < len(gen):
                f, g = rng.sample(fs, 2)
                imp.append((by[f][rng.choice(list(by[f]))], by[g][rng.choice(list(by[g]))]))
            gs = [pos_only_score(a, b) for a, b in gen]
            is_ = [pos_only_score(a, b) for a, b in imp]
            aucs.append(roc_auc_score([1] * len(gs) + [0] * len(is_), gs + is_))
            cnts.append(np.median([len(v) for dd in by.values() for v in dd.values()]))
        print(f"{variant} quality>={t:.2f}: meanAUC {np.mean(aucs):.3f}  per-DB " +
              " ".join(f"{a:.2f}" for a in aucs) + f"  minutiae/print {np.mean(cnts):.0f}", flush=True)


def run(variant, fn, target, dbs, seed=0):
    aucs, cnts, gm, im = [], [], [], []
    for y, db in dbs:
        paths = sorted(glob.glob(f"{ROOT}/{y}/Dbs/Db{db}_b/*.tif"))
        scale, _ = db_scale(paths, target, 4)
        t0 = time.time()
        by = {}
        for p in paths:
            f, i = F.parse_name(p)
            by.setdefault(f, {})[i] = fn(cv2.imread(p, cv2.IMREAD_GRAYSCALE), scale, (y, db))
        rng = random.Random(seed)
        gen = [(by[f][i], by[f][j]) for f in by for i in by[f] for j in by[f] if i < j]
        fs = list(by)
        imp = []
        while len(imp) < len(gen):
            f, g = rng.sample(fs, 2)
            imp.append((by[f][rng.choice(list(by[f]))], by[g][rng.choice(list(by[g]))]))
        gs = [pos_only_score(a, b) for a, b in gen]
        is_ = [pos_only_score(a, b) for a, b in imp]
        auc = roc_auc_score([1] * len(gs) + [0] * len(is_), gs + is_)
        aucs.append(auc); gm.append(np.mean(gs)); im.append(np.mean(is_))
        cnts.append(np.median([len(v) for d in by.values() for v in d.values()]))
        print(f"    {variant:20s} {y[-2:]}DB{db}: AUC {auc:.3f}  genuine {gm[-1]:.2f}  impostor {im[-1]:.2f}  "
              f"minutiae {cnts[-1]:.0f}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"{variant:20s} meanAUC {np.mean(aucs):.3f}  genuine-match {np.mean(gm):.2f}  impostor {np.mean(im):.2f}  "
          f"minutiae/print {np.mean(cnts):.0f}", flush=True)
    return float(np.mean(aucs))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=["ours", "leader_norm"])
    ap.add_argument("--dbs", type=int, default=len(DEV_DBS))
    ap.add_argument("--thresholds", nargs="*", type=float, default=None,
                    help="with variant leader_low: score these quality thresholds")
    args = ap.parse_args()
    ex = make_extractors()
    target = F.calibrate_target_period()
    for v in args.variants:
        if args.thresholds:
            run_thresholds(v, ex[v], target, DEV_DBS[:args.dbs], args.thresholds)
        else:
            run(v, ex[v], target, DEV_DBS[:args.dbs])
