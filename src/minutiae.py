"""
Minutiae extraction from fingerprint images.

Pipeline: grayscale -> normalize -> Gabor enhancement -> binarize -> thin (skeletonize)
-> crossing-number minutiae detection (ridge endings + bifurcations).

For production-grade extraction, consider swapping this out for NIST's MINDTCT
or a trained CNN minutiae detector (e.g. FingerNet). This module gives a
dependency-light, fully open-source baseline.
"""
import cv2
import numpy as np
from skimage.morphology import skeletonize
from dataclasses import dataclass


@dataclass
class Minutia:
    x: int
    y: int
    angle: float      # ridge orientation in radians
    type: int         # 0 = ending, 1 = bifurcation


def load_and_normalize(path: str, size=(256, 256)) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.resize(img, size)
    img = cv2.equalizeHist(img)
    return img


def gabor_enhance(img: np.ndarray) -> np.ndarray:
    """Enhance ridge structure with a bank of Gabor filters at multiple orientations."""
    accum = np.zeros_like(img, dtype=np.float32)
    for theta in np.arange(0, np.pi, np.pi / 8):
        kernel = cv2.getGaborKernel((21, 21), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(img.astype(np.float32), cv2.CV_32F, kernel)
        accum = np.maximum(accum, filtered)
    return cv2.normalize(accum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def binarize_and_skeletonize(img: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    skeleton = skeletonize(binary > 0)
    return skeleton.astype(np.uint8)


def _crossing_number(patch: np.ndarray) -> int:
    """8-neighborhood crossing number for a 3x3 patch centered on a ridge pixel."""
    p = [patch[0, 0], patch[0, 1], patch[0, 2], patch[1, 2],
         patch[2, 2], patch[2, 1], patch[2, 0], patch[1, 0]]
    return sum(abs(int(p[i]) - int(p[(i + 1) % 8])) for i in range(8)) // 2


def _local_orientation(skeleton: np.ndarray, y: int, x: int, win: int = 8) -> float:
    y0, y1 = max(0, y - win), min(skeleton.shape[0], y + win)
    x0, x1 = max(0, x - win), min(skeleton.shape[1], x + win)
    patch = skeleton[y0:y1, x0:x1]
    ys, xs = np.nonzero(patch)
    if len(xs) < 2:
        return 0.0
    vx, vy = np.polyfit(xs, ys, 1)[0], 1.0
    return float(np.arctan2(vy, vx))


def extract_minutiae(skeleton: np.ndarray, border: int = 10) -> list[Minutia]:
    """Crossing-number method: CN==1 -> ridge ending, CN==3 -> bifurcation."""
    h, w = skeleton.shape
    minutiae = []
    for y in range(border, h - border):
        for x in range(border, w - border):
            if skeleton[y, x] == 0:
                continue
            patch = skeleton[y - 1:y + 2, x - 1:x + 2]
            if patch.shape != (3, 3):
                continue
            cn = _crossing_number(patch)
            if cn == 1:
                minutiae.append(Minutia(x, y, _local_orientation(skeleton, y, x), type=0))
            elif cn == 3:
                minutiae.append(Minutia(x, y, _local_orientation(skeleton, y, x), type=1))
    return _suppress_duplicates(minutiae)


def _suppress_duplicates(minutiae: list[Minutia], min_dist: int = 8) -> list[Minutia]:
    """Non-max suppression: drop minutiae that are too close to an already-kept one."""
    kept = []
    for m in minutiae:
        if all((m.x - k.x) ** 2 + (m.y - k.y) ** 2 >= min_dist ** 2 for k in kept):
            kept.append(m)
    return kept


def minutiae_from_image(path: str) -> list[Minutia]:
    img = load_and_normalize(path)
    enhanced = gabor_enhance(img)
    skeleton = binarize_and_skeletonize(enhanced)
    return extract_minutiae(skeleton)


if __name__ == "__main__":
    import sys
    pts = minutiae_from_image(sys.argv[1])
    print(f"Extracted {len(pts)} minutiae")
    for m in pts[:10]:
        print(m)
