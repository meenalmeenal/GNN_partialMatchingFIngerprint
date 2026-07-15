import argparse
import os
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import FingerprintTripletDataset, triplet_collate
from model import TripletGCN


def train(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["train"]["seed"])

    dataset = FingerprintTripletDataset(cfg["data"]["graph_dir"], seed=cfg["train"]["seed"])
    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        collate_fn=triplet_collate,
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

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch}: avg triplet loss = {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = os.path.join(cfg["train"]["checkpoint_dir"], "best_model.pt")
            torch.save({"model_state": model.state_dict(), "config": cfg}, ckpt_path)
            print(f"  saved new best checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    train(args.config)
