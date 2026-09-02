"""
Diagnostics for the minutiae extractor on a sample of the organized dataset.

    python src/tune_extraction.py --organized data/SOCOFing_organized --n 300

Reports the minutiae-count distribution, timing, and a rough stability proxy
(within-finger count variation - lower is better). Also dumps a side-by-side
visualization for a few prints to results/extraction_preview.png.
"""
import argparse
import glob
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

from minutiae import (EXTRACT_CFG, load_and_normalize, segment_mask, orientation_field,
                      gabor_enhance, binarize_and_skeletonize, prune_spurs,
                      extract_minutiae, minutiae_from_image)


def _dist(xs):
    a = np.array(xs, float)
    return dict(n=len(a), min=float(a.min()), p10=float(np.percentile(a, 10)),
               median=float(np.median(a)), mean=round(float(a.mean()), 1),
               p90=float(np.percentile(a, 90)), max=float(a.max()))


def preview(paths, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = len(paths)
    fig, ax = plt.subplots(rows, 4, figsize=(12, 3 * rows))
    for r, p in enumerate(paths):
        img = load_and_normalize(p)
        mask = segment_mask(img)
        enh = gabor_enhance(img, orientation_field(img))
        skel = binarize_and_skeletonize(enh) * mask
        skelp = prune_spurs(skel)
        m = extract_minutiae(skelp, mask=mask)
        for c, (title, canvas) in enumerate([
            ("input", img), ("mask", mask * 255), ("enhanced", enh), ("skeleton+minutiae", skelp * 255)]):
            ax[r, c].imshow(canvas, cmap="gray")
            ax[r, c].set_title(title if r == 0 else "", fontsize=9)
            ax[r, c].axis("off")
        for mm in m:
            ax[r, 3].plot(mm.x, mm.y, "r." if mm.type == 0 else "b+", ms=6)
        ax[r, 3].set_title(f"{len(m)} minutiae" if r else f"skeleton+minutiae ({len(m)})", fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=90)
    print(f"preview -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--organized", default="data/SOCOFing_organized")
    ap.add_argument("--n", type=int, default=300, help="random images to sample")
    ap.add_argument("--preview", type=int, default=5)
    args = ap.parse_args()

    print("EXTRACT_CFG:", EXTRACT_CFG)
    ids = [d for d in glob.glob(os.path.join(args.organized, "*")) if os.path.isdir(d)]
    rng = random.Random(0)
    rng.shuffle(ids)

    counts, per_finger, t0 = [], [], time.time()
    done = 0
    for d in ids:
        imgs = glob.glob(os.path.join(d, "*.*"))
        if len(imgs) < 2:
            continue
        fc = []
        for p in imgs:
            try:
                fc.append(len(minutiae_from_image(p)))
                done += 1
            except Exception as e:
                print("skip", p, e)
        counts += fc
        if len(fc) >= 2:
            per_finger.append(np.std(fc) / (np.mean(fc) + 1e-6))
        if done >= args.n:
            break
    dt = time.time() - t0

    print(f"\nsampled {done} images from {len(per_finger)} fingers in {dt:.1f}s "
          f"({dt / done * 1000:.0f} ms/img)")
    print("minutiae count :", _dist(counts))
    print(f"within-finger CV (std/mean): median={np.median(per_finger):.3f}  "
          f"mean={np.mean(per_finger):.3f}   (lower = more stable)")

    if args.preview:
        pv = []
        for d in ids[:50]:
            g = glob.glob(os.path.join(d, "*.*"))
            if g:
                pv.append(g[0])
            if len(pv) >= args.preview:
                break
        preview(pv, "results/extraction_preview.png")


if __name__ == "__main__":
    main()
