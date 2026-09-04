"""
Train the Siamese GCN encoder with triplet loss.

Reads models/splits.json if present (else creates it), loads packed graphs from
data/packed/{train,val}.pt for fast in-RAM sampling, logs train/val loss per epoch
to results/<tag>/train_log.csv, early-stops on val loss, and plots the curve.

    python src/train.py --config configs/default.yaml
    python src/train.py --limit-fingers 800 --epochs 15 --iters-per-epoch 200 --tag quick
"""
import argparse
import csv
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import FingerprintTripletDataset, triplet_collate, split_subjects
from model import TripletGCN


@torch.no_grad()
def eval_loss(model, ds, batch_size, margin, device, seed=0):
    model.eval()
    ds._rng = random.Random(seed)            # identical val triplets every epoch
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=triplet_collate)
    total, n = 0.0, 0
    pos_d, neg_d = 0.0, 0.0
    for a, p, ng in loader:
        a, p, ng = a.to(device), p.to(device), ng.to(device)
        ea, ep, en = model(a, p, ng)
        total += F.triplet_margin_loss(ea, ep, en, margin=margin).item() * a.num_graphs
        pos_d += (ea - ep).norm(dim=1).sum().item()
        neg_d += (ea - en).norm(dim=1).sum().item()
        n += a.num_graphs
    return total / max(n, 1), pos_d / max(n, 1), neg_d / max(n, 1)


def plot_curve(csv_path, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader(open(csv_path)))
    ep = [int(r["epoch"]) for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [float(r["train_loss"]) for r in rows], label="train")
    ax[0].plot(ep, [float(r["val_loss"]) for r in rows], label="val")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("triplet loss"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(ep, [float(r["val_pos_dist"]) for r in rows], label="val d(a,pos)")
    ax[1].plot(ep, [float(r["val_neg_dist"]) for r in rows], label="val d(a,neg)")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("embedding distance"); ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=100); plt.close()


def train(cfg, tag="run"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["train"]["seed"])
    random.seed(cfg["train"]["seed"])

    graph_dir = cfg["data"]["graph_dir"]
    ckpt_dir = cfg["train"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    outdir = os.path.join("results", tag)
    os.makedirs(outdir, exist_ok=True)

    splits_path = os.path.join(ckpt_dir, "splits.json")
    if os.path.exists(splits_path):
        sp = json.load(open(splits_path))
        train_subs, val_subs = sp["train"], sp["val"]
    else:
        train_subs, val_subs, test_subs = split_subjects(
            graph_dir, cfg["train"]["val_split"], cfg["train"]["test_split"], seed=cfg["train"]["seed"])
        json.dump({"graph_dir": graph_dir, "seed": cfg["train"]["seed"],
                   "train": train_subs, "val": val_subs, "test": test_subs},
                  open(splits_path, "w"), indent=2)

    lim = cfg["train"].get("limit_fingers", 0)
    if lim:
        train_subs, val_subs = train_subs[:lim], val_subs[:max(2, lim // 5)]

    packed_dir = cfg["data"].get("packed_dir", "data/packed")
    tr_src = os.path.join(packed_dir, "train.pt")
    va_src = os.path.join(packed_dir, "val.pt")
    use_packed = os.path.exists(tr_src)
    print(f"source: {'packed ' + packed_dir if use_packed else 'per-file ' + graph_dir}")
    t0 = time.time()
    train_ds = FingerprintTripletDataset(
        graph_dir=None if use_packed else graph_dir,
        packed=tr_src if use_packed else None,
        subjects=train_subs, seed=cfg["train"]["seed"],
        epoch_size=cfg["train"]["iters_per_epoch"] * cfg["train"]["batch_size"])
    val_ds = FingerprintTripletDataset(
        graph_dir=None if use_packed else graph_dir,
        packed=va_src if use_packed else None,
        subjects=val_subs, seed=cfg["train"]["seed"] + 1,
        epoch_size=cfg["train"].get("val_iters", 60) * cfg["train"]["batch_size"])
    print(f"train fingers={len(train_ds.finger_ids)}  val fingers={len(val_ds.finger_ids)}  "
          f"(loaded in {time.time() - t0:.0f}s)")

    loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                        collate_fn=triplet_collate)

    model = TripletGCN(
        in_dim=cfg["data"]["node_feat_dim"], hidden_dim=cfg["model"]["hidden_dim"],
        embedding_dim=cfg["model"]["embedding_dim"], num_layers=cfg["model"]["num_gcn_layers"],
        dropout=cfg["model"]["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"],
                           weight_decay=cfg["train"]["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    margin = cfg["train"]["margin"]

    csv_path = os.path.join(outdir, "train_log.csv")
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["epoch", "train_loss", "val_loss", "val_pos_dist", "val_neg_dist", "lr", "sec"])
    csv_file.flush()

    best, bad, patience = float("inf"), 0, cfg["train"].get("patience", 8)
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        te = time.time()
        tot, n = 0.0, 0
        for a, p, ng in tqdm(loader, desc=f"Epoch {epoch}", leave=False,
                             disable=not sys.stdout.isatty()):
            a, p, ng = a.to(device), p.to(device), ng.to(device)
            opt.zero_grad()
            ea, ep, en = model(a, p, ng)
            loss = F.triplet_margin_loss(ea, ep, en, margin=margin)
            loss.backward()
            opt.step()
            tot += loss.item() * a.num_graphs
            n += a.num_graphs
        tr_loss = tot / max(n, 1)
        vl, pd, nd = eval_loss(model, val_ds, cfg["train"]["batch_size"], margin, device,
                               seed=cfg["train"]["seed"])
        sched.step(vl)
        lr = opt.param_groups[0]["lr"]
        dt = time.time() - te
        writer.writerow([epoch, f"{tr_loss:.4f}", f"{vl:.4f}", f"{pd:.4f}", f"{nd:.4f}",
                         f"{lr:.2e}", f"{dt:.0f}"])
        csv_file.flush()
        print(f"epoch {epoch:2d}  train {tr_loss:.4f}  val {vl:.4f}  "
              f"d+ {pd:.3f}  d- {nd:.3f}  lr {lr:.1e}  {dt:.0f}s")

        if vl < best - 1e-4:
            best, bad = vl, 0
            for path in (os.path.join(ckpt_dir, "best_model.pt"), os.path.join(outdir, "best_model.pt")):
                torch.save({"model_state": model.state_dict(), "config": cfg,
                            "epoch": epoch, "val_loss": vl}, path)
            print(f"  * new best (val {vl:.4f}) -> saved")
        else:
            bad += 1
            if bad >= patience:
                print(f"early stop: no val improvement for {patience} epochs")
                break

    csv_file.close()
    plot_curve(csv_path, os.path.join(outdir, "train_curve.png"))
    print(f"done. best val loss {best:.4f}. log -> {csv_path}, curve -> {outdir}/train_curve.png")
    return os.path.join(ckpt_dir, "best_model.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--iters-per-epoch", type=int)
    ap.add_argument("--limit-fingers", type=int)
    ap.add_argument("--patience", type=int)
    ap.add_argument("--lr", type=float)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cfg["train"].setdefault("iters_per_epoch", 400)
    cfg["train"].setdefault("patience", 8)
    if args.epochs: cfg["train"]["epochs"] = args.epochs
    if args.iters_per_epoch: cfg["train"]["iters_per_epoch"] = args.iters_per_epoch
    if args.limit_fingers: cfg["train"]["limit_fingers"] = args.limit_fingers
    if args.patience: cfg["train"]["patience"] = args.patience
    if args.lr: cfg["train"]["lr"] = args.lr

    train(cfg, tag=args.tag)


if __name__ == "__main__":
    main()
