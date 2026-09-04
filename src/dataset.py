"""
PyG Dataset that samples (anchor, positive, negative) graph triplets.

Two data sources:
  - graph_dir: root laid out as graph_dir/<finger_id>/<sample>.pt  (loads per file)
  - packed:    a dict {finger_id: {tag: Data}} or a path to one (from pack_graphs.py);
               everything is held in RAM -> no per-step disk I/O.

Samples under the same finger_id are "same finger" (positives); any other finger
is a negative.
"""
import glob
import os
import random

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch


def load_packed(packed):
    """Accept a dict or a path; return the {finger: {tag: Data}} dict."""
    if isinstance(packed, str):
        return torch.load(packed, weights_only=False)
    return packed


class FingerprintTripletDataset(Dataset):
    def __init__(self, graph_dir: str = None, seed: int = 42, subjects: list | None = None,
                 packed=None, epoch_size: int | None = None):
        self._rng = random.Random(seed)
        self.epoch_size = epoch_size
        self._cache = {}          # key -> Data (for the file-based path)
        self.samples = {}         # finger_id -> list of (key, Data-or-None)

        if packed is not None:
            data = load_packed(packed)
            allowed = set(subjects) if subjects is not None else None
            for fid, tagmap in data.items():
                if allowed is not None and fid not in allowed:
                    continue
                if len(tagmap) >= 2:
                    self.samples[fid] = [(f"{fid}/{t}", g) for t, g in tagmap.items()]
        else:
            allowed = set(subjects) if subjects is not None else None
            for fid in sorted(os.listdir(graph_dir)):
                if allowed is not None and fid not in allowed:
                    continue
                d = os.path.join(graph_dir, fid)
                if not os.path.isdir(d):
                    continue
                files = glob.glob(os.path.join(d, "*.pt"))
                if len(files) >= 2:
                    self.samples[fid] = [(p, None) for p in files]

        self.finger_ids = list(self.samples)
        if len(self.finger_ids) < 2:
            raise ValueError(f"Need >=2 fingers with >=2 samples; found {len(self.finger_ids)}")

    def _get(self, key, graph):
        if graph is not None:
            return graph
        if key not in self._cache:
            self._cache[key] = torch.load(key, weights_only=False)
        return self._cache[key]

    def __len__(self):
        return self.epoch_size or sum(len(v) for v in self.samples.values())

    def __getitem__(self, idx):
        fid = self._rng.choice(self.finger_ids)
        (ak, ag), (pk, pg) = self._rng.sample(self.samples[fid], 2)
        neg_fid = self._rng.choice(self.finger_ids)
        while neg_fid == fid:
            neg_fid = self._rng.choice(self.finger_ids)
        nk, ng = self._rng.choice(self.samples[neg_fid])
        return self._get(ak, ag), self._get(pk, pg), self._get(nk, ng)


def triplet_collate(batch):
    a, p, n = zip(*batch)
    return Batch.from_data_list(a), Batch.from_data_list(p), Batch.from_data_list(n)


def split_subjects(graph_dir: str, val_split: float, test_split: float, seed: int = 42):
    """Partition finger ids (disjoint) into train/val/test lists."""
    subs = sorted(
        s for s in os.listdir(graph_dir)
        if os.path.isdir(os.path.join(graph_dir, s))
        and len(glob.glob(os.path.join(graph_dir, s, "*.pt"))) >= 2
    )
    random.Random(seed).shuffle(subs)
    n = len(subs)
    n_test = int(round(n * test_split))
    n_val = int(round(n * val_split))
    return subs[n_test + n_val:], subs[n_test:n_test + n_val], subs[:n_test]
