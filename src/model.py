"""
GCN-based encoder for minutiae graphs + triplet loss training objective.

Architecture: stacked GCNConv layers -> global pooling (mean+max) -> MLP projection head.
Produces a fixed-size embedding per fingerprint graph, regardless of node count -
this is what lets a small partial-print graph be compared against a full-print graph.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool


class MinutiaeGCNEncoder(nn.Module):
    def __init__(self, in_dim=5, hidden_dim=64, embedding_dim=128, num_layers=3, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.dropout = dropout

        # mean-pool + max-pool concatenated -> projection head
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        pooled = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        emb = self.proj(pooled)
        return F.normalize(emb, p=2, dim=1)  # L2-normalize for cosine similarity matching


class TripletGCN(nn.Module):
    """Wraps the encoder to run on (anchor, positive, negative) graph batches."""

    def __init__(self, **encoder_kwargs):
        super().__init__()
        self.encoder = MinutiaeGCNEncoder(**encoder_kwargs)

    def forward(self, anchor, positive, negative):
        a = self.encoder(anchor.x, anchor.edge_index, anchor.batch)
        p = self.encoder(positive.x, positive.edge_index, positive.batch)
        n = self.encoder(negative.x, negative.edge_index, negative.batch)
        return a, p, n

    def embed(self, data):
        return self.encoder(data.x, data.edge_index, data.batch)
