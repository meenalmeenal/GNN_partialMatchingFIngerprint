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
    def __init__(self, graph_dir: str, seed: int = 42):
        self.graph_dir = graph_dir
        self.subjects = {}  # subject_id -> list of .pt paths
        for subj in sorted(os.listdir(graph_dir)):
            subj_path = os.path.join(graph_dir, subj)
            if not os.path.isdir(subj_path):
                continue
            files = glob.glob(os.path.join(subj_path, "*.pt"))
            if len(files) >= 2:  # need at least 2 samples for a positive pair
                self.subjects[subj] = files
        self.subject_ids = list(self.subjects.keys())
        random.seed(seed)

    def __len__(self):
        return sum(len(v) for v in self.subjects.values())

    def __getitem__(self, idx):
        subj = random.choice(self.subject_ids)
        files = self.subjects[subj]
        anchor_path, pos_path = random.sample(files, 2)

        neg_subj = random.choice([s for s in self.subject_ids if s != subj])
        neg_path = random.choice(self.subjects[neg_subj])

        anchor = torch.load(anchor_path, weights_only=False)
        pos = torch.load(pos_path, weights_only=False)
        neg = torch.load(neg_path, weights_only=False)
        return anchor, pos, neg


def triplet_collate(batch):
    anchors, positives, negatives = zip(*batch)
    return Batch.from_data_list(anchors), Batch.from_data_list(positives), Batch.from_data_list(negatives)
