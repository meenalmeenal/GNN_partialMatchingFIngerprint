"""
Cross-dataset (FVC) pipeline: images -> resolution-normalised minutiae -> graphs.

The SOCOFing extractor resizes every image to 160x160, which silently assumes SOCOFing's
ridge spacing. FVC sensors differ in dpi and image size, so here every image is rescaled so
that its *ridge period* equals the period SOCOFing prints have inside the extractor, then
the unchanged extractor (segmentation, orientation field, Gabor, skeleton, spur pruning,
crossing numbers) runs on it. Graphs use the same features and the same /256 coordinate
normalisation as the SOCOFing graphs, so models trained on SOCOFing apply unchanged.

    python src/fvc_pipeline.py --root data/FVC/<db folder> --out data/fvc_packed/<name>.pt
"""
import argparse
import glob
import multiprocessing as mp
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import torch

import minutiae as M
from augment import build_graph_fast

TARGET_PERIOD = None   # set from SOCOFing by calibrate_target_period(); cached below
_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "stage8",
                      "target_period.txt")


def estimate_ridge_period(gray, lo=4.0, hi=24.0):
    """Dominant ridge period (pixels) from the radial power spectrum of the central region."""
    h, w = gray.shape
    ch, cw = int(h * 0.7), int(w * 0.7)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    img = gray[y0:y0 + ch, x0:x0 + cw].astype(np.float32)
    img = img - cv2.GaussianBlur(img, (0, 0), 12)          # remove low-frequency illumination
    n = 256
    win = np.outer(np.hanning(ch), np.hanning(cw)).astype(np.float32)
    f = np.abs(np.fft.fftshift(np.fft.fft2(img * win, s=(n, n)))) ** 2
    yy, xx = np.indices(f.shape)
    r = np.hypot(yy - n // 2, xx - n // 2)
    # frequency (cycles per pixel) = r / n ; period = n / r
    radial = np.bincount(r.astype(int).ravel(), weights=f.ravel()) / np.maximum(
        np.bincount(r.astype(int).ravel()), 1)
    periods = n / np.maximum(np.arange(len(radial)), 1e-6)
    ok = (periods >= lo) & (periods <= hi)
    if not ok.any():
        return None
    # weight by r to compensate the 1/f fall-off, then take the peak
    score = radial * np.arange(len(radial))
    idx = np.where(ok)[0][np.argmax(score[ok])]
    return float(periods[idx])


def skeleton_period(gray):
    """Ridge period as mask area / skeleton length, measured with the extractor itself. Used
    to calibrate the SOCOFing target: on SOCOFing's saturated, near-binary prints the spectral
    estimator locks onto a harmonic (4.8 px vs the true ~8.2 px), whereas on FVC's grayscale
    prints both estimators agree (checked on all 8 FVC2002/2004 sensors)."""
    img = cv2.equalizeHist(gray)
    mask = M.segment_mask(img)
    skel = M.binarize_and_skeletonize(M.gabor_enhance(img, M.orientation_field(img))) * mask
    skel = M.prune_spurs(M._drop_specks(skel))
    return float(mask.sum() / max(skel.sum(), 1))


def calibrate_target_period(socofing_dir="data/SOCOFing_organized", n=200):
    """Ridge period of SOCOFing prints as seen inside the extractor (160x160, equalised)."""
    if os.path.exists(_CACHE):
        return float(open(_CACHE).read())
    paths = sorted(glob.glob(os.path.join(socofing_dir, "*", "Real_real.bmp")))[:: max(1, 6000 // n)]
    ps = [skeleton_period(M.load_and_normalize(p)) for p in paths]
    val = float(np.median(ps))
    os.makedirs(os.path.dirname(_CACHE), exist_ok=True)
    open(_CACHE, "w").write(str(val))
    print(f"SOCOFing ridge period inside the extractor: median {val:.2f}px "
          f"(IQR {np.percentile(ps, 25):.2f}-{np.percentile(ps, 75):.2f}, n={len(ps)})")
    return val


def minutiae_from_gray(gray, target_period, fixed_scale=None):
    """Same extractor as minutiae.minutiae_from_image, after ridge-period normalisation.
    Pass `fixed_scale` (one factor per database/sensor): estimating the period per image
    jitters the scale by 11-25 % between impressions of the same finger and wrecks matching."""
    est = estimate_ridge_period(gray)
    if fixed_scale is not None:
        scale = fixed_scale
    else:
        scale = 1.0 if not est else float(np.clip(target_period / est, 0.4, 4.0))
    img = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    img = cv2.equalizeHist(img)
    mask = M.segment_mask(img)
    orient = M.orientation_field(img)
    enhanced = M.gabor_enhance(img, orient)
    skel = M.binarize_and_skeletonize(enhanced) * mask
    skel = M._drop_specks(skel)
    skel = M.prune_spurs(skel)
    return M.extract_minutiae(skel, mask=mask), est, scale, img, skel


def minutiae_to_data(mins):
    if len(mins) < 2:
        return None
    px = np.array([m.x for m in mins], dtype=np.float64)
    py = np.array([m.y for m in mins], dtype=np.float64)
    th = np.array([m.angle for m in mins], dtype=np.float64)
    ty = np.array([m.type for m in mins], dtype=np.float64)
    return build_graph_fast(px, py, th, ty)


def _est_path(path):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return estimate_ridge_period(g) if g is not None else None


def _work(args):
    path, target, fixed_scale = args
    M.EXTRACT_CFG["max_minutiae"] = 200          # FVC prints cover more area than SOCOFing's
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        return path, None, None
    mins, est, scale, _, _ = minutiae_from_gray(g, target, fixed_scale)
    return path, minutiae_to_data(mins), (est, scale, len(mins))


def parse_name(path):
    m = re.match(r"^(\d+)_(\d+)\.", os.path.basename(path))
    return (int(m.group(1)), int(m.group(2))) if m else None


def build_db(root, out, target, workers=7, set_filter="A"):
    """set A = fingers 1-100 (benchmark); set B = 101-110. Returns {finger: {impression: Data}}."""
    paths = [p for p in glob.glob(os.path.join(root, "*.tif")) if parse_name(p)]
    if set_filter == "A":
        paths = [p for p in paths if parse_name(p)[0] <= 100]
    elif set_filter == "B":
        paths = [p for p in paths if parse_name(p)[0] > 100]
    print(f"{root}: {len(paths)} images (set {set_filter})")
    with mp.Pool(workers) as pool:
        ests = [e for e in pool.map(_est_path, sorted(paths), chunksize=8) if e]
        fixed = float(np.clip(target / np.median(ests), 0.4, 4.0))
        print(f"  fixed database scale {fixed:.3f} (ridge period median {np.median(ests):.2f}px, "
              f"per-image CV {np.std(ests) / np.median(ests):.1%})")
        res = pool.map(_work, [(p, target, fixed) for p in sorted(paths)], chunksize=8)
    packed, stats, failed = {}, [], 0
    for p, data, st in res:
        if data is None:
            failed += 1
            continue
        f, i = parse_name(p)
        packed.setdefault(f, {})[i] = data
        stats.append(st)
    est = np.array([s[0] for s in stats if s[0]], dtype=float)
    cnt = np.array([s[2] for s in stats])
    print(f"  graphs built: {sum(len(v) for v in packed.values())}, failed: {failed}; "
          f"ridge period est median {np.median(est):.1f}px -> scale median "
          f"{np.median([s[1] for s in stats]):.2f}; minutiae/print median {np.median(cnt):.0f} "
          f"(p10 {np.percentile(cnt, 10):.0f}, p90 {np.percentile(cnt, 90):.0f})")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save(packed, out)
    return packed, {"graphs": len(stats), "failed": failed, "minutiae_median": float(np.median(cnt)),
                    "minutiae_p10": float(np.percentile(cnt, 10)),
                    "minutiae_p90": float(np.percentile(cnt, 90)),
                    "period_est_median": float(np.median(est))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder with <finger>_<impression>.tif files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", default="A", choices=["A", "B", "all"])
    ap.add_argument("--workers", type=int, default=7)
    args = ap.parse_args()
    target = calibrate_target_period()
    build_db(args.root, args.out, target, args.workers, None if args.set == "all" else args.set)


if __name__ == "__main__":
    mp.freeze_support()
    main()
