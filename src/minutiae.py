"""
Minutiae extraction from fingerprint images (classical, dependency-light).

Pipeline:
  grayscale -> resize -> normalize
  -> segmentation mask (block variance)      : restrict to the actual ridge area
  -> ridge orientation field (Rao gradient)  : local ridge direction per block
  -> orientation-selective Gabor enhancement : sharpen ridges, suppress noise
  -> adaptive binarize -> skeletonize
  -> spur pruning                            : kill short skeleton branches
  -> crossing-number minutiae detection      : CN==1 ending, CN==3 bifurcation
  -> spatial dedup + spur-pair removal + quality cap

Tunables live in EXTRACT_CFG so we can sweep them in tune_extraction.py.
For production accuracy, swap in NIST MINDTCT or a CNN detector (FingerNet).
"""
from dataclasses import dataclass

import cv2
import numpy as np
from skimage.morphology import skeletonize


@dataclass
class Minutia:
    x: int
    y: int
    angle: float      # local ridge orientation in radians
    type: int         # 0 = ridge ending, 1 = bifurcation


EXTRACT_CFG = dict(
    size=160,             # square resize target
    block=16,             # block size for segmentation + orientation
    seg_ratio=0.30,       # foreground if block std > seg_ratio * global std
    gabor_ksize=15,
    gabor_sigma=3.5,
    gabor_lambda=9.0,     # ridge wavelength at the resized scale
    gabor_gamma=0.6,
    n_orient=16,          # Gabor orientation bank size
    spur_len=8,           # skeleton branches shorter than this are pruned
    border=6,             # margin (px) eroded off the mask before detection
    dedup_dist=7,         # min spacing between kept minutiae
    pair_dist=14,         # an ending + bifurcation closer than this = spur artifact -> drop both
    max_minutiae=100,     # keep at most this many, by quality score
)


# ------------------------------------------------------------------ preprocessing

def load_and_normalize(path: str, size=None) -> np.ndarray:
    size = size or EXTRACT_CFG["size"]
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    img = cv2.equalizeHist(img)
    return img


def segment_mask(img: np.ndarray, block: int = None, ratio: float = None) -> np.ndarray:
    """Block-variance foreground segmentation -> clean boolean ROI mask."""
    block = block or EXTRACT_CFG["block"]
    ratio = EXTRACT_CFG["seg_ratio"] if ratio is None else ratio
    h, w = img.shape
    f = img.astype(np.float32)
    gstd = f.std() + 1e-6
    mask = np.zeros((h, w), np.uint8)
    for y in range(0, h, block):
        for x in range(0, w, block):
            blk = f[y:y + block, x:x + block]
            if blk.std() > ratio * gstd:
                mask[y:y + block, x:x + block] = 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (block, block))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (lbl == biggest).astype(np.uint8)
    return mask.astype(bool)


def orientation_field(img: np.ndarray, block: int = None) -> np.ndarray:
    """Rao gradient-based ridge orientation, one angle per pixel (block-constant)."""
    block = block or EXTRACT_CFG["block"]
    f = img.astype(np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    vx = cv2.boxFilter(2 * gx * gy, cv2.CV_32F, (block, block))
    vy = cv2.boxFilter(gx * gx - gy * gy, cv2.CV_32F, (block, block))
    # smooth the doubled-angle vector field, then halve
    vx = cv2.GaussianBlur(vx, (0, 0), block / 3.0)
    vy = cv2.GaussianBlur(vy, (0, 0), block / 3.0)
    theta = 0.5 * np.arctan2(vx, vy) + np.pi / 2.0
    return theta


def gabor_enhance(img: np.ndarray, orient: np.ndarray = None) -> np.ndarray:
    """Orientation-selective Gabor: filter with a bank, pick the response whose
    filter orientation is closest to the local ridge orientation."""
    c = EXTRACT_CFG
    if orient is None:
        orient = orientation_field(img)
    f = img.astype(np.float32)
    n = c["n_orient"]
    angles = np.arange(n) * (np.pi / n)
    responses = np.stack([
        cv2.filter2D(f, cv2.CV_32F, cv2.getGaborKernel(
            (c["gabor_ksize"], c["gabor_ksize"]), c["gabor_sigma"],
            # Gabor kernel is oriented perpendicular to ridge flow
            a + np.pi / 2, c["gabor_lambda"], c["gabor_gamma"], 0, ktype=cv2.CV_32F))
        for a in angles
    ], axis=0)
    idx = np.round((orient % np.pi) / (np.pi / n)).astype(int) % n
    out = np.take_along_axis(responses, idx[None], axis=0)[0]
    return cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def binarize_and_skeletonize(img: np.ndarray) -> np.ndarray:
    binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 2 * EXTRACT_CFG["block"] + 1, 5)
    skeleton = skeletonize(binary > 0)
    return skeleton.astype(np.uint8)


# ------------------------------------------------------------------ skeleton clean

_NB = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]


def _cn_map(skel: np.ndarray) -> np.ndarray:
    """Crossing number for every ridge pixel (vectorised)."""
    s = skel.astype(np.int32)
    p = [np.roll(np.roll(s, dy, 0), dx, 1) for dy, dx in _NB]
    cn = np.zeros_like(s)
    for i in range(8):
        cn += np.abs(p[i] - p[(i + 1) % 8])
    cn = cn // 2
    cn[s == 0] = 0
    return cn


def prune_spurs(skel: np.ndarray, max_len: int = None) -> np.ndarray:
    """Remove skeleton branches shorter than max_len that dangle off a junction."""
    max_len = max_len or EXTRACT_CFG["spur_len"]
    s = skel.copy().astype(np.uint8)
    h, w = s.shape
    for _ in range(2):
        cn = _cn_map(s)
        endpoints = list(zip(*np.where((cn == 1))))
        removed = False
        for (y, x) in endpoints:
            if s[y, x] == 0:
                continue
            path = [(y, x)]
            cy, cx = y, x
            prev = None
            for _step in range(max_len + 1):
                nbrs = [(cy + dy, cx + dx) for dy, dx in _NB
                        if 0 <= cy + dy < h and 0 <= cx + dx < w
                        and s[cy + dy, cx + dx] and (cy + dy, cx + dx) != prev]
                if len(nbrs) != 1:
                    break
                prev = (cy, cx)
                cy, cx = nbrs[0]
                if _count_nb(s, cy, cx) >= 3:
                    # hit a junction within max_len -> the collected path is a spur
                    for (py, px) in path:
                        s[py, px] = 0
                    removed = True
                    break
                path.append((cy, cx))
        if not removed:
            break
    return s


def _count_nb(s, y, x):
    h, w = s.shape
    return sum(1 for dy, dx in _NB
              if 0 <= y + dy < h and 0 <= x + dx < w and s[y + dy, x + dx])


# ------------------------------------------------------------------ minutiae

def _local_orientation(skeleton: np.ndarray, y: int, x: int, win: int = 8) -> float:
    y0, y1 = max(0, y - win), min(skeleton.shape[0], y + win)
    x0, x1 = max(0, x - win), min(skeleton.shape[1], x + win)
    ys, xs = np.nonzero(skeleton[y0:y1, x0:x1])
    if len(xs) < 2:
        return 0.0
    xs = xs.astype(np.float64) - xs.mean()
    ys = ys.astype(np.float64) - ys.mean()
    cov = np.array([[xs @ xs, xs @ ys], [xs @ ys, ys @ ys]])
    evals, evecs = np.linalg.eigh(cov)
    vx, vy = evecs[:, int(np.argmax(evals))]
    return float(np.arctan2(vy, vx))


def extract_minutiae(skeleton: np.ndarray, mask: np.ndarray = None,
                     border: int = None) -> list[Minutia]:
    border = EXTRACT_CFG["border"] if border is None else border
    h, w = skeleton.shape
    if mask is None:
        mask = np.ones((h, w), bool)
    roi = cv2.erode(mask.astype(np.uint8),
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * border + 1, 2 * border + 1)))
    # erosion with replicated borders doesn't shrink a mask that touches the image
    # edge - explicitly drop the outer frame (SOCOFing BMPs have a black border).
    roi[:border, :] = 0
    roi[-border:, :] = 0
    roi[:, :border] = 0
    roi[:, -border:] = 0
    cn = _cn_map(skeleton)
    out = []
    ys, xs = np.where((cn == 1) | (cn == 3))
    for y, x in zip(ys, xs):
        if y < 1 or x < 1 or y >= h - 1 or x >= w - 1 or not roi[y, x]:
            continue
        out.append(Minutia(int(x), int(y), _local_orientation(skeleton, y, x),
                           type=0 if cn[y, x] == 1 else 1))
    return _postfilter(out)


def _postfilter(minutiae: list[Minutia]) -> list[Minutia]:
    c = EXTRACT_CFG
    # drop ending/bifurcation pairs that sit on top of each other (spur signature)
    drop = set()
    for i in range(len(minutiae)):
        for j in range(i + 1, len(minutiae)):
            mi, mj = minutiae[i], minutiae[j]
            if mi.type != mj.type and (mi.x - mj.x) ** 2 + (mi.y - mj.y) ** 2 < c["pair_dist"] ** 2:
                drop.add(i)
                drop.add(j)
    minutiae = [m for i, m in enumerate(minutiae) if i not in drop]

    # spatial non-max suppression
    kept = []
    for m in minutiae:
        if all((m.x - k.x) ** 2 + (m.y - k.y) ** 2 >= c["dedup_dist"] ** 2 for k in kept):
            kept.append(m)

    # quality cap: prefer minutiae far from the ROI edge (i.e. later-added ones are
    # centre-ish) - here approximate quality by distance to image centre-of-mass
    if len(kept) > c["max_minutiae"]:
        cx = np.mean([m.x for m in kept])
        cy = np.mean([m.y for m in kept])
        kept.sort(key=lambda m: (m.x - cx) ** 2 + (m.y - cy) ** 2)
        kept = kept[:c["max_minutiae"]]
    return kept


def _suppress_duplicates(minutiae: list[Minutia], min_dist: int = 8) -> list[Minutia]:
    kept = []
    for m in minutiae:
        if all((m.x - k.x) ** 2 + (m.y - k.y) ** 2 >= min_dist ** 2 for k in kept):
            kept.append(m)
    return kept


def minutiae_from_image(path: str) -> list[Minutia]:
    img = load_and_normalize(path)
    mask = segment_mask(img)
    orient = orientation_field(img)
    enhanced = gabor_enhance(img, orient)
    skeleton = binarize_and_skeletonize(enhanced)
    skeleton = skeleton * mask
    skeleton = _drop_specks(skeleton)
    skeleton = prune_spurs(skeleton)
    return extract_minutiae(skeleton, mask=mask)


def _drop_specks(skel: np.ndarray, min_area: int = 10) -> np.ndarray:
    """Remove tiny disconnected skeleton fragments (each contributes 2 false endings)."""
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(skel.astype(np.uint8), 8)
    keep = np.zeros_like(skel)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[lbl == i] = 1
    return keep


if __name__ == "__main__":
    import sys
    pts = minutiae_from_image(sys.argv[1])
    print(f"Extracted {len(pts)} minutiae")
    for m in pts[:10]:
        print(m)
