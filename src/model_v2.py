"""
Stage 9 models. One `Matcher` class, three architectures, so each design decision can be
ablated on its own:

  v1_global   the Stage 3 GCN (absolute x/y/angle features, mean+max pooling -> one vector)
  inv_global  rotation/translation-INVARIANT features + edge-conditioned GNN, still pooled
              into one vector (isolates the effect of invariant features)
  inv_local   same invariant GNN, but NO pooling: every minutia keeps its own descriptor and
              two prints are compared by MaxSim (each query minutia is matched to its best
              gallery minutia, scores averaged). This is a learned analogue of the classical
              correspondence count, and it handles partial queries naturally because a
              query that covers part of a print only needs its own minutiae to find matches.

Invariant features are computed from the stored node features (x, y, sin, cos, type) inside
the model, so graph files, packed data and augmentation are unchanged. Orientation is
treated as an undirected line (doubled angles): the extractor's PCA orientation has an
arbitrary sign, and on matched minutiae it agrees as a direction 91.9 % of the time but as
a line 93.8 % of the time.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

from model import MinutiaeGCNEncoder

PX_PER_NORM = 256.0     # graph x_norm = pixel / 256
DIST_SCALE = 20.0       # typical neighbour spacing, pixels -> roughly unit scale
EDGE_DIM = 7


def relative_edge_features(x, edge_index):
    """Rotation/translation-invariant edge features, from the source node's point of view.
    [dist, cos2b_ij, sin2b_ij, cos2b_ji, sin2b_ji, cos2dth, sin2dth]  (b = bearing relative
    to the node's orientation line, dth = orientation difference; doubled angles)."""
    s, t = edge_index
    pos = x[:, :2] * PX_PER_NORM
    o = torch.stack([x[:, 3], x[:, 2]], dim=1)            # orientation vector (cos, sin)
    d = pos[t] - pos[s]
    dist = d.norm(dim=1, keepdim=True).clamp_min(1e-6)
    u = d / dist

    def dbl(a, b):  # a . b and a x b for 2-vectors -> cos2, sin2 of the angle between them
        c = (a * b).sum(1)
        sn = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        return c * c - sn * sn, 2 * c * sn

    c_ij, s_ij = dbl(o[s], u)           # neighbour direction in s's frame
    c_ji, s_ji = dbl(o[t], -u)          # reverse direction in t's frame
    c_dt, s_dt = dbl(o[s], o[t])
    return torch.cat([dist / DIST_SCALE, c_ij[:, None], s_ij[:, None], c_ji[:, None],
                      s_ji[:, None], c_dt[:, None], s_dt[:, None]], dim=1)


class RelLayer(nn.Module):
    """h_i' = h_i + MLP( [h_i, mean_j MLP_msg([h_i, h_j, e_ij])] )"""

    def __init__(self, h, dropout):
        super().__init__()
        self.msg = nn.Sequential(nn.Linear(2 * h + EDGE_DIM, h), nn.ReLU(), nn.Linear(h, h))
        self.upd = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU())
        self.norm = nn.LayerNorm(h)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, edge_index, e):
        s, t = edge_index
        m = self.msg(torch.cat([h[s], h[t], e], dim=1))
        agg = torch.zeros_like(h).index_add_(0, s, m)
        deg = torch.zeros(h.size(0), device=h.device).index_add_(0, s, torch.ones_like(s, dtype=h.dtype))
        agg = agg / deg.clamp_min(1).unsqueeze(1)
        return self.norm(h + self.drop(self.upd(torch.cat([h, agg], dim=1))))


class InvariantNet(nn.Module):
    def __init__(self, hidden=64, layers=2, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(2, hidden)                    # minutia type (one-hot) only
        self.layers = nn.ModuleList([RelLayer(hidden, dropout) for _ in range(layers)])

    def forward(self, x, edge_index):
        t = x[:, 4].long().clamp(0, 1)
        h = F.relu(self.inp(F.one_hot(t, 2).float()))
        e = relative_edge_features(x, edge_index)
        for layer in self.layers:
            h = layer(h, edge_index, e)
        return h


class Matcher(nn.Module):
    def __init__(self, arch="inv_local", hidden=64, out_dim=64, layers=2, dropout=0.1):
        super().__init__()
        self.cfg = dict(arch=arch, hidden=hidden, out_dim=out_dim, layers=layers, dropout=dropout)
        self.arch = arch
        self.local = arch == "inv_local"
        if arch == "v1_global":
            self.net = MinutiaeGCNEncoder(in_dim=5, hidden_dim=hidden, embedding_dim=out_dim,
                                          num_layers=max(layers, 3), dropout=0.3)
        else:
            self.net = InvariantNet(hidden, layers, dropout)
            if self.local:
                self.head = nn.Linear(hidden, out_dim)
            else:
                self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                                          nn.Dropout(dropout), nn.Linear(hidden, out_dim))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def encode(self, b):
        """global archs -> [B, D] unit vectors; local arch -> (dense [B, K, D], mask [B, K])."""
        if self.arch == "v1_global":
            return self.net(b.x, b.edge_index, b.batch)
        h = self.net(b.x, b.edge_index)
        if self.local:
            d = F.normalize(self.head(h), dim=1)
            return to_dense_batch(d, b.batch)
        from torch_geometric.nn import global_mean_pool, global_max_pool
        pooled = torch.cat([global_mean_pool(h, b.batch), global_max_pool(h, b.batch)], dim=1)
        return F.normalize(self.head(pooled), dim=1)


def sim_matrix(model, q, g, chunk=16):
    """Score matrix [Bq, Bg] between encoded queries and encoded gallery (differentiable)."""
    if not model.local:
        return q @ g.t()
    qd, qm = q
    gd, gm = g
    rows = []
    for i in range(0, qd.size(0), chunk):
        sims = torch.einsum("pqd,gkd->pgqk", qd[i:i + chunk], gd)
        sims = sims.masked_fill(~gm[None, :, None, :], -2.0)
        best = sims.max(dim=3).values                           # [p, g, q]
        m = qm[i:i + chunk][:, None, :].float()
        rows.append((best * m).sum(2) / m.sum(2).clamp_min(1))  # mean over real query nodes
    return torch.cat(rows, 0)


def similarity(model, q_batch, g_batch):
    return sim_matrix(model, model.encode(q_batch), model.encode(g_batch), chunk=64)


@torch.no_grad()
def score_matrix(model, probe_graphs, gallery_graphs, device="cpu", bs=128, chunk=8):
    """Probes x gallery score matrix for a list of Data graphs."""
    from torch_geometric.data import Batch
    model.eval()

    def enc(graphs):
        outs = [model.encode(Batch.from_data_list(graphs[i:i + bs]).to(device))
                for i in range(0, len(graphs), bs)]
        if not model.local:
            return torch.cat(outs, 0)
        k = max(o[0].size(1) for o in outs)
        pad = lambda t, v: F.pad(t, (0, 0, 0, k - t.size(1))) if t.dim() == 3 else F.pad(t, (0, k - t.size(1)))
        return (torch.cat([pad(o[0], 0) for o in outs], 0), torch.cat([pad(o[1], 0) for o in outs], 0))

    return sim_matrix(model, enc(probe_graphs), enc(gallery_graphs), chunk=chunk).cpu()


def save_matcher(path, model, **extra):
    torch.save({"model_state": model.state_dict(), "arch_cfg": model.cfg, **extra}, path)


def load_matcher(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = Matcher(**ck["arch_cfg"]).to(device)
    m.load_state_dict(ck["model_state"])
    m.eval()
    return m
