"""
Back-end of the Stage 7 demo (no Streamlit imports, so it can be tested on its own).

  dataset  -> where prints, images and minutiae graphs come from (SOCOFing test split or one
              FVC database, held-out fingers only)
  Gallery  -> the enrolled templates of a dataset, encoded once by a model
  search() -> rank the whole gallery for one (possibly partial) query graph, with the learned
              matcher and, optionally, with the classical Hough matcher for comparison
  crop_window / draw_* -> turn a query into a partial print and visualise everything

Graph coordinates are pixel / 256 in the extractor's own image space (160x160 for SOCOFing,
the ridge-period-normalised image for FVC), so minutiae are drawn on `dataset.image()`.
"""
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import torch

import minutiae as M
from augment import GRAPH_NORM, build_graph_fast
from classical_match import graph_to_minutiae, hough_score
from fvc_pipeline import calibrate_target_period, minutiae_from_gray, minutiae_to_data
from model_v2 import encode_graphs, load_matcher, sim_matrix

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
FVC_ROOT = os.path.join(ROOT, "data", "FVC", "74034_3_En_4_MOESM1_ESM")
GALLERY_TAG = "Real_real"
FVC_HELDOUT = (51, 100)      # fingers never used for fine-tuning or model selection (Stage 8)
MIN_MINUTIAE = 3

MODEL_FILES = {
    "Stage 9 matcher (trained on SOCOFing only)": ["models/stage9_inv_local_cont.pt",
                                                  "results/stage9_inv_local_cont/best_model.pt"],
    "Stage 8 matcher, fine-tuned on real FVC prints": ["models/stage8_inv_local_fvcft.pt",
                                                      "results/stage8_inv_local_fvcft/best_model.pt"],
}


def model_path(label):
    for rel in MODEL_FILES[label]:
        p = os.path.join(ROOT, rel)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no checkpoint found for '{label}': tried {MODEL_FILES[label]}")


def available_models():
    out = []
    for k in MODEL_FILES:
        try:
            model_path(k)
            out.append(k)
        except FileNotFoundError:
            pass
    return out


# --------------------------------------------------------------------------- datasets
class SocoDataset:
    """SOCOFing test split (900 held-out fingers). Gallery = the clean `Real` print, queries =
    its synthetically altered versions (CR / Obl / Zcut, Easy / Medium / Hard)."""
    key = "socofing"
    title = "SOCOFing - synthetic alterations (900 held-out fingers)"
    real = False
    gallery_key = GALLERY_TAG

    def __init__(self):
        self.packed = torch.load(os.path.join(ROOT, "data", "packed", "test.pt"), weights_only=False)
        self.fingers = sorted(self.packed)

    def query_keys(self, f):
        return sorted(k for k in self.packed[f] if k != GALLERY_TAG)

    def graph(self, f, key):
        return self.packed[f][key]

    def label(self, f):
        return str(f)

    def image(self, f, key):
        p = os.path.join(ROOT, "data", "SOCOFing_organized", str(f), key + ".BMP")
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        return None if img is None else cv2.resize(img, (M.EXTRACT_CFG["size"],) * 2,
                                                   interpolation=cv2.INTER_CUBIC)

    def extract(self, gray):
        """Uploaded image -> (graph, display image) with the exact SOCOFing recipe."""
        M.EXTRACT_CFG["max_minutiae"] = 100
        s = M.EXTRACT_CFG["size"]
        shown = cv2.resize(gray, (s, s), interpolation=cv2.INTER_CUBIC)
        img = cv2.equalizeHist(shown)
        mask = M.segment_mask(img)
        skel = M.binarize_and_skeletonize(M.gabor_enhance(img, M.orientation_field(img))) * mask
        skel = M.prune_spurs(M._drop_specks(skel))
        return minutiae_to_data(M.extract_minutiae(skel, mask=mask)), shown


def fvc_scales(target):
    """Fixed ridge-period scale factor of every FVC database, exactly as `fvc_pipeline.build_db`
    derived it (target period / median per-image estimate over set A). Cached in
    data/fvc_packed/scales.json; needed to process uploaded images and to draw the stored prints."""
    path = os.path.join(ROOT, "data", "fvc_packed", "scales.json")
    if os.path.exists(path):
        return json.load(open(path))
    import glob
    from fvc_pipeline import estimate_ridge_period, parse_name
    out = {}
    for y in (2002, 2004):
        for db in (1, 2, 3, 4):
            d = os.path.join(FVC_ROOT, f"FVC{y}", "Dbs", f"Db{db}_a")
            ests = [estimate_ridge_period(cv2.imread(p, cv2.IMREAD_GRAYSCALE))
                    for p in sorted(glob.glob(os.path.join(d, "*.tif"))) if parse_name(p)[0] <= 100]
            out[f"fvc{y}_db{db}"] = float(np.clip(target / np.median([e for e in ests if e]), 0.4, 4.0))
    json.dump(out, open(path, "w"), indent=1)
    return out


class FvcDataset:
    """One FVC2002/2004 database (set A), held-out fingers 51-100 only. Gallery = impression 1,
    queries = impressions 2-8 (independent captures of the same finger)."""
    real = True
    gallery_key = 1

    def __init__(self, name):
        self.key = name                                     # e.g. fvc2002_db1
        y, db = int(name[3:7]), int(name[-1])
        self.dir = os.path.join(FVC_ROOT, f"FVC{y}", "Dbs", f"Db{db}_a")
        kind = "synthetic SFinGe" if db == 4 else "real sensor"
        self.title = f"FVC{y} DB{db} - independent impressions ({kind}, 50 held-out fingers)"
        packed = torch.load(os.path.join(ROOT, "data", "fvc_packed", name + ".pt"), weights_only=False)
        self.fingers = [f for f in sorted(packed) if FVC_HELDOUT[0] <= f <= FVC_HELDOUT[1] and 1 in packed[f]]
        self.packed = packed
        self.target = calibrate_target_period()
        self.scale = fvc_scales(self.target)[name]

    def query_keys(self, f):
        return [i for i in sorted(self.packed[f]) if i != 1]

    def graph(self, f, key):
        return self.packed[f][key]

    def label(self, f):
        return f"finger {f}"

    def image(self, f, key):
        img = cv2.imread(os.path.join(self.dir, f"{f}_{key}.tif"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)
        return cv2.equalizeHist(img)

    def extract(self, gray):
        """Uploaded image -> (graph, display image); assumes the same sensor as this database."""
        M.EXTRACT_CFG["max_minutiae"] = 200
        mins, _, _, img, _ = minutiae_from_gray(gray, self.target, self.scale)
        return minutiae_to_data(mins), img


def dataset_choices():
    out = {"socofing": SocoDataset.title}
    for y in (2002, 2004):
        for db in (1, 2, 3, 4):
            n = f"fvc{y}_db{db}"
            if os.path.exists(os.path.join(ROOT, "data", "fvc_packed", n + ".pt")) and os.path.isdir(
                    os.path.join(FVC_ROOT, f"FVC{y}", "Dbs", f"Db{db}_a")):
                out[n] = f"FVC{y} DB{db}" + (" (synthetic)" if db == 4 else " (real sensor)")
    return out


def load_dataset(key):
    return SocoDataset() if key == "socofing" else FvcDataset(key)


# --------------------------------------------------------------------------- gallery + search
class Gallery:
    """Enrolled templates of a dataset, encoded once by one model."""

    def __init__(self, ds, model):
        self.ds, self.model = ds, model
        self.fingers = list(ds.fingers)
        self.graphs = [ds.graph(f, ds.gallery_key) for f in self.fingers]
        self.mins = [graph_to_minutiae(g) for g in self.graphs]
        t = time.time()
        self.enc = encode_graphs(model, self.graphs)
        self.encode_sec = time.time() - t

    def search_learned(self, q_graph):
        """-> (scores [G], query encoding). Score = MaxSim: mean over query minutiae of the best
        cosine match among the template's minutiae."""
        eq = encode_graphs(self.model, [q_graph])
        s = sim_matrix(self.model, eq, self.enc, chunk=1)[0]
        return s.detach().numpy(), eq

    def search_classical(self, q_graph):
        qm = graph_to_minutiae(q_graph)
        return np.array([hough_score(qm, g) for g in self.mins], dtype=np.float32)


def rank_of(scores, idx):
    """1-based rank of gallery entry `idx` (ties counted in its favour only if strictly greater)."""
    return int((scores > scores[idx]).sum()) + 1


def zscores(scores):
    return (scores - scores.mean()) / (scores.std() + 1e-6)


# --------------------------------------------------------------------------- partial prints
def crop_window(g, frac, cx=0.5, cy=0.5):
    """Square window covering `frac` of the print's own extent, placed by (cx, cy) in [0, 1]
    over the feasible positions (same family of crops as the Stage 6/8/9 evaluations).
    -> (cropped graph, kept indices, (x0, y0, side))."""
    x = g.x.numpy()
    px, py = x[:, 0] * GRAPH_NORM, x[:, 1] * GRAPH_NORM
    th, ty = np.arctan2(x[:, 2], x[:, 3]), x[:, 4]
    xmin, xmax, ymin, ymax = px.min(), px.max(), py.min(), py.max()
    if frac >= 0.999:
        return g, np.arange(len(px)), (xmin, ymin, max(xmax - xmin, ymax - ymin))
    side = math.sqrt(frac) * max(xmax - xmin, ymax - ymin, 1.0)
    x0 = xmin + cx * max(xmax - xmin - side, 0.0)
    y0 = ymin + cy * max(ymax - ymin - side, 0.0)
    keep = np.where((px >= x0) & (px <= x0 + side) & (py >= y0) & (py <= y0 + side))[0]
    if len(keep) < MIN_MINUTIAE:                          # window nearly empty: take the nearest few
        c = (x0 + side / 2, y0 + side / 2)
        keep = np.argsort((px - c[0]) ** 2 + (py - c[1]) ** 2)[:MIN_MINUTIAE]
    return build_graph_fast(px[keep], py[keep], th[keep], ty[keep]), keep, (x0, y0, side)


# --------------------------------------------------------------------------- drawing
END_COL, BIF_COL, DIM_COL, WIN_COL = "#d1495b", "#00798c", "#9aa0a6", "#edae49"


def _pts(g):
    x = g.x.numpy()
    return x[:, 0] * GRAPH_NORM, x[:, 1] * GRAPH_NORM, np.arctan2(x[:, 2], x[:, 3]), x[:, 4]


def draw_minutiae(ax, img, g, keep=None, rect=None, title=None, arrows=True):
    """Fingerprint with minutiae overlaid: red = ridge ending, teal = bifurcation, with a short tick
    for orientation. If `keep` is given, the other minutiae are drawn faintly and `rect` outlines
    the partial-print window."""
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    px, py, th, ty = _pts(g)
    on = np.ones(len(px), bool) if keep is None else np.isin(np.arange(len(px)), keep)
    for sel, alpha, dim in ((~on, 0.9, True), (on, 1.0, False)):
        for t, col in ((0, END_COL), (1, BIF_COL)):
            m = sel & (ty == t)
            if m.any():
                ax.scatter(px[m], py[m], s=26, facecolors="none", edgecolors=DIM_COL if dim else col,
                           linewidths=1.0 if dim else 1.6, alpha=alpha)
    if arrows:
        for i in np.where(on)[0]:
            ax.plot([px[i], px[i] + 9 * np.cos(th[i])], [py[i], py[i] + 9 * np.sin(th[i])],
                    color=END_COL if ty[i] == 0 else BIF_COL, lw=1.2)
    if rect is not None and keep is not None and len(keep) < len(px):
        x0, y0, s = rect
        ax.add_patch(__import__("matplotlib").patches.Rectangle((x0, y0), s, s, fill=False,
                                                                 edgecolor=WIN_COL, lw=2.2))
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


def draw_correspondence(fig, q_img, q_graph, g_img, g_graph, q_desc, g_desc, top=14, title=None):
    """The learned score is a mean over query minutiae of their best match in the template; draw
    the strongest of those (query minutia -> best template minutia) links."""
    import matplotlib.pyplot as plt  # noqa: F401
    ax1, ax2 = fig.subplots(1, 2)
    draw_minutiae(ax1, q_img, q_graph, title="query (partial)", arrows=False)
    draw_minutiae(ax2, g_img, g_graph, title=title or "enrolled template", arrows=False)
    sims = q_desc @ g_desc.T                          # [Kq, Kg] cosine
    best, arg = sims.max(1), sims.argmax(1)
    order = np.argsort(-best)[:top]
    qx, qy, _, _ = _pts(q_graph)
    gx, gy, _, _ = _pts(g_graph)
    from matplotlib.patches import ConnectionPatch
    cmap = __import__("matplotlib").colormaps["viridis"]
    lo, hi = float(best[order].min()), float(best[order].max())
    for i in order:
        c = cmap(0.15 + 0.8 * (best[i] - lo) / max(hi - lo, 1e-6))
        fig.add_artist(ConnectionPatch((qx[i], qy[i]), (gx[arg[i]], gy[arg[i]]), "data", "data",
                                       axesA=ax1, axesB=ax2, color=c, lw=1.3, alpha=0.9))
    return float(best.mean())
