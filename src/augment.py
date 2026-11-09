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


def _impression_noise(px, py, th, ty, cfg, rng):
    """Simulate an independent impression of the same finger: smooth elastic warp, per-minutia
    position/orientation jitter, missed minutiae, spurious minutiae, occasional type flips.
    (SOCOFing probes are copies of the gallery impression, so a model trained on them has
    never seen any of this.)  Strengths in cfg are maxima; each print draws its own level."""
    n = len(px)
    amp = rng.uniform(0.0, cfg.get("warp_px", 4.0))
    if amp > 0:
        for _ in range(2):                                   # two random low-frequency waves
            w = rng.uniform(0.01, 0.04)
            ang = rng.uniform(0, math.pi)
            ph = rng.uniform(0, 2 * math.pi)
            kx, ky = w * math.cos(ang), w * math.sin(ang)
            wave = np.sin(2 * math.pi * (kx * px + ky * py) + ph)
            dx = amp * wave * rng.uniform(-1, 1)
            dy = amp * wave * rng.uniform(-1, 1)
            px, py = px + dx, py + dy
    jit = rng.uniform(0.0, cfg.get("jitter_px", 2.0))
    px = px + np.array([rng.gauss(0, jit) for _ in range(n)])
    py = py + np.array([rng.gauss(0, jit) for _ in range(n)])
    th = th + np.array([rng.gauss(0, cfg.get("angle_rad", 0.2)) for _ in range(n)])
    keep = np.array([rng.random() > rng.uniform(0.0, 1.0) * cfg.get("drop", 0.3) for _ in range(n)])
    if keep.sum() < 2:
        keep[:2] = True
    px, py, th, ty = px[keep], py[keep], th[keep], ty[keep].copy()
    flip = np.array([rng.random() < cfg.get("type_flip", 0.1) for _ in range(len(ty))])
    ty = np.where(flip, 1.0 - ty, ty)
    n_sp = int(len(px) * rng.uniform(0.0, 1.0) * cfg.get("spurious", 0.3))
    if n_sp > 0:
        lo_x, hi_x, lo_y, hi_y = px.min(), px.max(), py.min(), py.max()
        sx = np.array([rng.uniform(lo_x, hi_x) for _ in range(n_sp)])
        sy = np.array([rng.uniform(lo_y, hi_y) for _ in range(n_sp)])
        px, py = np.concatenate([px, sx]), np.concatenate([py, sy])
        th = np.concatenate([th, np.array([rng.uniform(-math.pi, math.pi) for _ in range(n_sp)])])
        ty = np.concatenate([ty, np.array([float(rng.random() < 0.5) for _ in range(n_sp)])])
    return px, py, th, ty


class GraphAugmenter:
    def __init__(self, crop=True, max_rot_deg=0.0, p_full=0.15, min_frac=0.10,
                 p_anchor=1.0, p_other=0.3, crop_roles=("a", "p", "n"), noise=None,
                 noise_roles=(), bbox_crop=False):
        """noise: dict of impression-noise strengths (see _impression_noise), applied to
        `noise_roles`; crop applies only to `crop_roles`."""
        self.crop = crop
        self.crop_roles = set(crop_roles)
        self.noise = noise
        self.noise_roles = set(noise_roles)
        self.bbox_crop = bbox_crop      # crop relative to the print's own extent (needed for FVC)
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
        if self.crop and role in self.crop_roles and rng.random() > self.p_full:
            frac = rng.uniform(self.min_frac, 1.0)
            if self.bbox_crop:
                bx0, bx1, by0, by1 = px.min(), px.max(), py.min(), py.max()
                side = math.sqrt(frac) * max(bx1 - bx0, by1 - by0, 1.0)
            else:
                bx0, bx1, by0, by1 = 0.0, float(IMG_PX), 0.0, float(IMG_PX)
                side = math.sqrt(frac) * IMG_PX
            best, best_n = None, -1
            for _ in range(25):
                ci = rng.randrange(len(px))
                x0 = min(max(px[ci] - side / 2, bx0), max(bx1 - side, bx0))
                y0 = min(max(py[ci] - side / 2, by0), max(by1 - side, by0))
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
        px, py, th, ty = px[idx], py[idx], th[idx], ty[idx]
        if self.noise and role in self.noise_roles:
            px, py, th, ty = _impression_noise(px, py, th, ty, self.noise, rng)
        return build_graph_fast(px, py, th, ty)
