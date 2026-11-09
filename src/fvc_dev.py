"""
Extractor development on FVC SET B ONLY (fingers 101-110, provided by FVC for parameter
tuning). Measures how repeatable the extracted minutiae are across independent impressions of
the same finger, using orientation-free (position + type) rigid registration:

  AUC     separation of genuine vs impostor pairs by the registration score  (higher = better)
  match   mean fraction of minutiae consistent under the best rigid transform (genuine pairs)

The set-A benchmark databases are never touched here.

    python src/fvc_dev.py
"""
import argparse
import glob
import multiprocessing as mp
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

import minutiae as M
import fvc_pipeline as F

ROOT = "data/FVC/74034_3_En_4_MOESM1_ESM"
DEV_DBS = [("FVC2002", 1), ("FVC2002", 2), ("FVC2002", 3), ("FVC2004", 1), ("FVC2004", 2)]

BASE = dict(border=6, dedup=7, pair=14, spur=8, clean=0, coh=0.0, block=16)


def coherence_map(img, sigma):
    f = img.astype(np.float32)
    gx, gy = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    sxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma)
    syy = cv2.GaussianBlur(gy * gy, (0, 0), sigma)
    sxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma)
    return np.sqrt((sxx - syy) ** 2 + 4 * sxy ** 2) / (sxx + syy + 1e-6)


def gabor_multi(img, orient, lams):
    """Orientation-selective Gabor at several ridge wavelengths; per pixel keep the response of
    the wavelength with the strongest local energy (handles ridge spacing that varies across
    a print, which a single fixed wavelength cannot)."""
    c = M.EXTRACT_CFG
    n = c["n_orient"]
    f = img.astype(np.float32)
    idx = np.round((orient % np.pi) / (np.pi / n)).astype(int) % n
    best_e, best_r = None, None
    for lam in lams:
        sigma = 3.5 * lam / 9.0
        k = int(4.3 * sigma) | 1
        resp = np.stack([cv2.filter2D(f, cv2.CV_32F, cv2.getGaborKernel(
            (k, k), sigma, a + np.pi / 2, lam, c["gabor_gamma"], 0, ktype=cv2.CV_32F))
            for a in np.arange(n) * (np.pi / n)], axis=0)
        r = np.take_along_axis(resp, idx[None], axis=0)[0]
        e = cv2.GaussianBlur(np.abs(r), (0, 0), 6.0)
        if best_e is None:
            best_e, best_r = e, r
        else:
            take = e > best_e
            best_e = np.where(take, e, best_e)
            best_r = np.where(take, r, best_r)
    return cv2.normalize(best_r, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def extract_multi(gray, target, variants, fixed_scale=None):
    """One pass over the expensive stages, shared by every variant:
    (resize/orient/Gabor/binary) once, skeleton once per (clean, spur), post-filters per variant."""
    if fixed_scale is not None:
        scale = fixed_scale
    else:
        est = F.estimate_ridge_period(gray)
        scale = 1.0 if not est else float(np.clip(target / est, 0.4, 4.0))
    img = cv2.equalizeHist(cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC))
    mask = M.segment_mask(img)
    orient = M.orientation_field(img)
    from skimage.morphology import skeletonize
    binaries = {}
    coh = None
    skels, out = {}, {}
    for name, o in variants.items():
        lam = tuple(o.get("lam", ()))
        if lam not in binaries:
            enh = gabor_multi(img, orient, lam) if lam else M.gabor_enhance(img, orient)
            binaries[lam] = cv2.adaptiveThreshold(enh, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                  cv2.THRESH_BINARY, 2 * M.EXTRACT_CFG["block"] + 1, 5)
        key = (lam, o["clean"], o["spur"])
        if key not in skels:
            binary = binaries[lam].copy()
            if o["clean"]:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
                for pol in (255, 0):
                    n, lbl, st, _ = cv2.connectedComponentsWithStats((binary == pol).astype(np.uint8), 4)
                    bad = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] < int(o["clean"])]
                    binary[np.isin(lbl, bad)] = 255 - pol
            sk = skeletonize(binary > 0).astype(np.uint8) * mask
            skels[key] = M.prune_spurs(M._drop_specks(sk), o["spur"])
        M.EXTRACT_CFG.update(border=o["border"], dedup_dist=o["dedup"], pair_dist=o["pair"],
                             spur_len=o["spur"], max_minutiae=250)
        mins = M.extract_minutiae(skels[key], mask=mask)
        if o["coh"] > 0 and mins:
            if coh is None:
                coh = coherence_map(img, sigma=2 * target)
            mins = [m for m in mins if coh[int(m.y), int(m.x)] >= o["coh"]]
        out[name] = np.array([[m.x, m.y, m.angle, m.type] for m in mins], dtype=np.float64).reshape(-1, 4)
    return out


def _work(args):
    path, target, variants, fixed_scale = args
    return extract_multi(cv2.imread(path, cv2.IMREAD_GRAYSCALE), target, variants, fixed_scale)


def _est(path):
    return F.estimate_ridge_period(cv2.imread(path, cv2.IMREAD_GRAYSCALE))


def db_scale(paths, target, workers):
    with mp.Pool(workers) as p:
        ests = [e for e in p.map(_est, paths) if e]
    return float(np.clip(target / np.median(ests), 0.4, 4.0)), float(np.std(ests) / np.median(ests))


def pos_only_score(A, B, step_deg=6, bin_px=7.0):
    if len(A) < 3 or len(B) < 3:
        return 0.0
    a, b = A[:, :2], B[:, :2]
    a0 = a - a.mean(0)
    same = A[:, 3][:, None] == B[:, 3][None, :]
    if not same.any():
        return 0.0
    best = 0
    for deg in range(-180, 180, step_deg):
        c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        ra = a0 @ np.array([[c, s], [-s, c]])
        d = (b[None, :, :] - ra[:, None, :])[same]
        k = np.round(d / bin_px).astype(np.int64)
        key = (k[:, 0] + 8192) * 16384 + (k[:, 1] + 8192)
        best = max(best, np.unique(key, return_counts=True)[1].max())
    return best / min(len(A), len(B))


def evaluate_grid(variants, target, workers, dbs=DEV_DBS, seed=0, per_db_scale=False):
    res = {n: [] for n in variants}
    for y, db in dbs:
        paths = sorted(glob.glob(f"{ROOT}/{y}/Dbs/Db{db}_b/*.tif"))
        fixed = None
        if per_db_scale:
            fixed, cv = db_scale(paths, target, workers)
            print(f"  {y} DB{db}: fixed scale {fixed:.3f} (per-image period estimate CV {cv:.1%})", flush=True)
        with mp.Pool(workers) as p:
            per_img = p.map(_work, [(q, target, variants, fixed) for q in paths], chunksize=2)
        for name in variants:
            by = {}
            for q, d in zip(paths, per_img):
                f, i = F.parse_name(q)
                by.setdefault(f, {})[i] = d[name]
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
            res[name].append((f"{y[-2:]}DB{db}", auc, float(np.mean(gs)), float(np.mean(is_)),
                              float(np.median([len(d[name]) for d in per_img]))))
    return res


def show(tag, res):
    aucs = [r[1] for r in res]
    print(f"{tag:42s} meanAUC {np.mean(aucs):.3f} | " +
          " ".join(f"{r[0]}:{r[1]:.2f}" for r in res) +
          f" | genuine-match {np.mean([r[2] for r in res]):.2f} impostor {np.mean([r[3] for r in res]):.2f}"
          f" | minutiae/print {np.mean([r[4] for r in res]):.0f}", flush=True)
    return float(np.mean(aucs))


if __name__ == "__main__":
    mp.freeze_support()
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    args = ap.parse_args()
    T = F.calibrate_target_period()
    V = {"baseline": BASE}
    for name, r in evaluate_grid(V, T, args.workers).items():
        show(name, r)
