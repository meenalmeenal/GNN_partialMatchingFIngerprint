import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import FingerprintTripletDataset, triplet_collate, split_subjects
from model import TripletGCN


@torch.no_grad()
def eval_loss(model, loader, margin, device):
    model.eval()
    total, n = 0.0, 0
    for anchor, pos, neg in loader:
        anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
        a, p, nn_ = model(anchor, pos, neg)
        total += F.triplet_margin_loss(a, p, nn_, margin=margin).item() * anchor.num_graphs
        n += anchor.num_graphs
    return total / max(n, 1)


def train(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["train"]["seed"])

    graph_dir = cfg["data"]["graph_dir"]
    train_subs, val_subs, test_subs = split_subjects(
        graph_dir, cfg["train"]["val_split"], cfg["train"]["test_split"], seed=cfg["train"]["seed"]
    )
    print(f"Subjects  train={len(train_subs)}  val={len(val_subs)}  test={len(test_subs)}")
    os.makedirs(cfg["train"]["checkpoint_dir"], exist_ok=True)
    with open(os.path.join(cfg["train"]["checkpoint_dir"], "splits.json"), "w") as f:
        json.dump({"train": train_subs, "val": val_subs, "test": test_subs}, f, indent=2)

    dataset = FingerprintTripletDataset(graph_dir, seed=cfg["train"]["seed"], subjects=train_subs)
    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        collate_fn=triplet_collate,
    )
    val_loader = None
    if val_subs:
        val_ds = FingerprintTripletDataset(graph_dir, seed=cfg["train"]["seed"] + 1, subjects=val_subs)
        val_loader = DataLoader(
            val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, collate_fn=triplet_collate
        )

    model = TripletGCN(
        in_dim=cfg["data"]["node_feat_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        embedding_dim=cfg["model"]["embedding_dim"],
        num_layers=cfg["model"]["num_gcn_layers"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )
    margin = cfg["train"]["margin"]

    os.makedirs(cfg["train"]["checkpoint_dir"], exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for anchor, pos, neg in tqdm(loader, desc=f"Epoch {epoch}"):
            anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
            optimizer.zero_grad()
            a, p, n = model(anchor, pos, neg)
            loss = F.triplet_margin_loss(a, p, n, margin=margin)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * anchor.num_graphs

        avg_loss = total_loss / max(len(dataset), 1)
        monitor = eval_loss(model, val_loader, margin, device) if val_loader else avg_loss
        tag = "val" if val_loader else "train"
        print(f"Epoch {epoch}: train loss = {avg_loss:.4f}  {tag} loss = {monitor:.4f}")

        if monitor < best_loss:
            best_loss = monitor
            ckpt_path = os.path.join(cfg["train"]["checkpoint_dir"], "best_model.pt")
            torch.save({"model_state": model.state_dict(), "config": cfg}, ckpt_path)
            print(f"  saved new best checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    train(args.config)
