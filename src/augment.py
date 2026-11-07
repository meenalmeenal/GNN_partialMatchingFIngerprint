"""
Training-time graph augmentation: simulate partial and rotated prints.

Works in true pixel space (graph x_norm * 256, image = 160 px, see Stage 6 notes), rebuilds
the k-NN graph from the surviving minutiae, and follows the same window-crop logic as
partiality_sweep.crop_graph so train-time crops resemble the evaluation crops (different
random draws).

Triplet roles: the ANCHOR (the query) is always augmented; the POSITIVE and NEGATIVE
(gallery-like prints) are augmented with probability p_other so the network cannot use
"small graph" as a shortcut.
"""
import math

import numpy as np
import torch
from torch_geometric.data import Data

IMG_PX = 160
GRAPH_NORM = 256
MIN_MINUTIAE = 3


def build_graph_fast(px, py, th, ty, k=5):
    """Vectorised equivalent of graph_build.minutiae_to_graph (same features, same edge
    order); ~10x faster, which matters because augmentation rebuilds graphs every step."""
    n = len(px)
    coords = np.stack([px, py], 1).astype(np.float32)
    feats = np.stack([px / GRAPH_NORM, py / GRAPH_NORM, np.sin(th), np.cos(th),
                      np.asarray(ty, dtype=np.float64)], 1).astype(np.float32)
    k_eff = min(k + 1, n)
    d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, -1.0)  # self is always the nearest neighbour, as in sklearn
    nbr = np.argsort(d2, axis=1, kind="stable")[:, 1:k_eff]
    adj = np.zeros((n, n), dtype=bool)
    adj[np.repeat(np.arange(n), nbr.shape[1]), nbr.ravel()] = True
    adj |= adj.T
    src, dst = np.nonzero(adj)
    dist = np.sqrt(((coords[src] - coords[dst]) ** 2).sum(-1)) / np.sqrt(2.0 * GRAPH_NORM ** 2)
    rel = th[src] - th[dst]
    eattr = np.stack([dist, np.sin(rel), np.cos(rel)], 1).astype(np.float32)
    return Data(x=torch.from_numpy(feats),
                edge_index=torch.from_numpy(np.stack([src, dst]).astype(np.int64)),
                edge_attr=torch.from_numpy(eattr), num_nodes=n)


class GraphAugmenter:
    def __init__(self, crop=True, max_rot_deg=0.0, p_full=0.15, min_frac=0.10,
                 p_anchor=1.0, p_other=0.3):
        self.crop = crop
        self.max_rot = math.radians(max_rot_deg)
        self.p_full = p_full
        self.min_frac = min_frac
        self.p = {"a": p_anchor, "p": p_other, "n": p_other}

    def __call__(self, data, role, rng):
        if rng.random() > self.p[role]:
            return data
        x = data.x.numpy()
        px, py = x[:, 0] * GRAPH_NORM, x[:, 1] * GRAPH_NORM
        th, ty = np.arctan2(x[:, 2], x[:, 3]), x[:, 4]

        if self.max_rot > 0:
            phi = rng.uniform(-self.max_rot, self.max_rot)
            c, s = math.cos(phi), math.sin(phi)
            cx = cy = IMG_PX / 2
            px, py = cx + (px - cx) * c - (py - cy) * s, cy + (px - cx) * s + (py - cy) * c
            th = th + phi

        idx = np.arange(len(px))
        if self.crop and rng.random() > self.p_full:
            frac = rng.uniform(self.min_frac, 1.0)
            side = math.sqrt(frac) * IMG_PX
            best, best_n = None, -1
            for _ in range(25):
                ci = rng.randrange(len(px))
                x0 = min(max(px[ci] - side / 2, 0), IMG_PX - side)
                y0 = min(max(py[ci] - side / 2, 0), IMG_PX - side)
                sel = np.where((px >= x0) & (px <= x0 + side) & (py >= y0) & (py <= y0 + side))[0]
                if len(sel) > best_n:
                    best, best_n = sel, len(sel)
                if best_n >= MIN_MINUTIAE:
                    break
            if best_n < 2:
                ci = rng.randrange(len(px))
                best = np.argsort((px - px[ci]) ** 2 + (py - py[ci]) ** 2)[:2]
            idx = best

        idx = np.asarray(idx)
        return build_graph_fast(px[idx], py[idx], th[idx], ty[idx])
