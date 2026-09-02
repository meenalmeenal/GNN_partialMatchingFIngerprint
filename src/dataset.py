"""
PyG Dataset that samples (anchor, positive, negative) graph triplets.

Assumes graph files are laid out as:
  graph_dir/<subject_id>/<sample>.pt
so that all files under the same subject_id are "same finger" (positives),
and files under any other subject_id are negatives.
"""
import os
import random
import glob
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch


class FingerprintTripletDataset(Dataset):
    def __init__(self, graph_dir: str, seed: int = 42, subjects: list | None = None):
        """
        graph_dir: root laid out as graph_dir/<subject_id>/<sample>.pt
        subjects:  optional whitelist of subject_ids to include (used for train/val/test splits).
        """
        self.graph_dir = graph_dir
        self.subjects = {}  # subject_id -> list of .pt paths
        allowed = set(subjects) if subjects is not None else None
        for subj in sorted(os.listdir(graph_dir)):
            if allowed is not None and subj not in allowed:
                continue
            subj_path = os.path.join(graph_dir, subj)
            if not os.path.isdir(subj_path):
                continue
            files = glob.glob(os.path.join(subj_path, "*.pt"))
            if len(files) >= 2:  # need at least 2 samples for a positive pair
                self.subjects[subj] = files
        self.subject_ids = list(self.subjects.keys())
        if len(self.subject_ids) < 2:
            raise ValueError(
                f"Need >=2 subjects with >=2 samples each in {graph_dir}; found {len(self.subject_ids)}"
            )
        self._rng = random.Random(seed)

    def __len__(self):
        return sum(len(v) for v in self.subjects.values())

    def __getitem__(self, idx):
        subj = self._rng.choice(self.subject_ids)
        files = self.subjects[subj]
        anchor_path, pos_path = self._rng.sample(files, 2)

        neg_subj = self._rng.choice([s for s in self.subject_ids if s != subj])
        neg_path = self._rng.choice(self.subjects[neg_subj])

        anchor = torch.load(anchor_path, weights_only=False)
        pos = torch.load(pos_path, weights_only=False)
        neg = torch.load(neg_path, weights_only=False)
        return anchor, pos, neg


def triplet_collate(batch):
    anchors, positives, negatives = zip(*batch)
    return Batch.from_data_list(anchors), Batch.from_data_list(positives), Batch.from_data_list(negatives)


def split_subjects(graph_dir: str, val_split: float, test_split: float, seed: int = 42):
    """Partition subject_ids (disjoint) into train/val/test lists."""
    subs = sorted(
        s for s in os.listdir(graph_dir)
        if os.path.isdir(os.path.join(graph_dir, s))
        and len(glob.glob(os.path.join(graph_dir, s, "*.pt"))) >= 2
    )
    random.Random(seed).shuffle(subs)
    n = len(subs)
    n_test = int(round(n * test_split))
    n_val = int(round(n * val_split))
    test = subs[:n_test]
    val = subs[n_test:n_test + n_val]
    train = subs[n_test + n_val:]
    return train, val, test
