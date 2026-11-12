"""
Stage 9 training: in-batch contrastive (InfoNCE) loss for the Matcher architectures.

Each step draws B (query, gallery) pairs of DIFFERENT prints of the same finger. The query
is a randomly cropped print (partial-print augmentation, anchors only); the gallery print is
left whole, mirroring "partial query vs full gallery". Every other gallery print in the batch
is a negative, so each query sees B-1 negatives instead of one random one (this also acts as
cheap hard-negative mining). Same-finger collisions inside a batch are masked out.

    python src/train_v2.py --arch inv_local --tag stage9_inv_local
"""
import argparse
import csv
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from augment import GraphAugmenter
from dataset import FingerprintTripletDataset, triplet_collate
from model_v2 import Matcher, similarity, save_matcher


def contrastive_loss(model, a, p):
    S = similarity(model, a, p)                                   # [B, B] query x gallery
    scale = model.logit_scale.exp().clamp(max=100.0)
    logits = S * scale
    same = a.fid[:, None] == p.fid[None, :]
    eye = torch.eye(S.size(0), dtype=torch.bool, device=S.device)
    logits = logits.masked_fill(same & ~eye, float("-inf"))       # same finger != negative
    target = torch.arange(S.size(0), device=S.device)
    loss = F.cross_entropy(logits, target)
    acc = (logits.argmax(1) == target).float().mean().item()
    return loss, acc


@torch.no_grad()
def evaluate(model, ds, batch_size, device, seed):
    model.eval()
    ds._rng = random.Random(seed)
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=triplet_collate)
    tl, ta, n = 0.0, 0.0, 0
    for a, p, _ in loader:
        loss, acc = contrastive_loss(model, a.to(device), p.to(device))
        tl += loss.item(); ta += acc; n += 1
    return tl / n, ta / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["v1_global", "inv_global", "inv_local"], required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ckpt-name", default=None)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--out-dim", type=int, default=None)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--iters-per-epoch", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--splits", default="models/splits.json")
    ap.add_argument("--packed-dir", default="data/packed")
    ap.add_argument("--noise", action="store_true",
                    help="simulate independent impressions (warp, jitter, missed/spurious minutiae) "
                         "on query AND gallery prints")
    ap.add_argument("--fvc-dbs", nargs="*", default=[],
                    help="fine-tune on real FVC impressions from these data/fvc_packed/<db>.pt "
                         "(training fingers 1-40, validation 41-50; fingers 51-100 stay held out)")
    ap.add_argument("--device", default="cpu", help="cpu | cuda | auto")
    ap.add_argument("--workers", type=int, default=0, help="DataLoader worker processes (graph augmentation)")
    ap.add_argument("--fvc-dir", default="data/fvc_packed", help="packed FVC graphs for --fvc-dbs")
    ap.add_argument("--init", default=None, help="warm-start from a Matcher checkpoint "
                    "(same architecture); optimizer and LR schedule restart")
    args = ap.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))
    print(f"device: {device}", flush=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    out_dim = args.out_dim or (128 if args.arch == "v1_global" else 64)
    outdir = os.path.join("results", args.tag)
    os.makedirs(outdir, exist_ok=True)
    ckpt_name = args.ckpt_name or args.tag

    sp = json.load(open(args.splits))
    if args.noise:
        aug = GraphAugmenter(crop=True, max_rot_deg=0.0, p_anchor=1.0, p_other=1.0, crop_roles=("a",),
                             noise={"warp_px": 4.0, "jitter_px": 2.0, "angle_rad": 0.2, "drop": 0.3,
                                    "spurious": 0.3, "type_flip": 0.1}, noise_roles=("a", "p"))
    else:
        aug = GraphAugmenter(crop=True, max_rot_deg=0.0, p_anchor=1.0, p_other=0.0)
    t0 = time.time()
    if args.fvc_dbs:
        tr_pack, va_pack = {}, {}
        for db in args.fvc_dbs:
            d = torch.load(os.path.join(args.fvc_dir, db + ".pt"), weights_only=False)
            for f, imps in d.items():
                if len(imps) >= 2:
                    if f <= 40:
                        tr_pack[f"{db}_{f}"] = imps
                    elif f <= 50:
                        va_pack[f"{db}_{f}"] = imps
        # FVC prints vary in size: crop relative to each print's own extent
        aug.bbox_crop = True
        train_ds = FingerprintTripletDataset(packed=tr_pack, subjects=list(tr_pack), seed=args.seed,
            augment=aug, attach_fid=True, epoch_size=args.iters_per_epoch * args.batch_size)
        val_ds = FingerprintTripletDataset(packed=va_pack, subjects=list(va_pack), seed=args.seed + 1,
            augment=aug, attach_fid=True, epoch_size=40 * args.batch_size)
    else:
        train_ds = FingerprintTripletDataset(
            packed=os.path.join(args.packed_dir, "train.pt"), subjects=sp["train"], seed=args.seed,
            augment=aug, attach_fid=True, epoch_size=args.iters_per_epoch * args.batch_size)
        val_ds = FingerprintTripletDataset(
            packed=os.path.join(args.packed_dir, "val.pt"), subjects=sp["val"], seed=args.seed + 1,
            augment=aug, attach_fid=True, epoch_size=40 * args.batch_size)
    print(f"arch={args.arch} train fingers={len(train_ds.finger_ids)} val={len(val_ds.finger_ids)} "
          f"(loaded in {time.time() - t0:.0f}s)", flush=True)

    loader = DataLoader(train_ds, batch_size=args.batch_size, collate_fn=triplet_collate,
                        num_workers=args.workers, persistent_workers=args.workers > 0)
    model = Matcher(args.arch, hidden=args.hidden, out_dim=out_dim, layers=args.layers).to(device)
    if args.init:
        ck = torch.load(args.init, map_location=device, weights_only=False)
        assert ck["arch_cfg"] == model.cfg, f"architecture mismatch: {ck['arch_cfg']} vs {model.cfg}"
        model.load_state_dict(ck["model_state"])
        print(f"warm start from {args.init} (epoch {ck.get('epoch')}, val {ck.get('val_loss')})", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)

    log_path = os.path.join(outdir, "train_log.csv")
    log_f = open(log_path, "w", newline="")
    w = csv.writer(log_f)
    w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "sec"])
    best, bad = float("inf"), 0
    os.makedirs("models", exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        te = time.time()
        tl, ta, n = 0.0, 0.0, 0
        for a, p, _ in loader:
            a, p = a.to(device), p.to(device)
            opt.zero_grad()
            loss, acc = contrastive_loss(model, a, p)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tl += loss.item(); ta += acc; n += 1
        vl, va = evaluate(model, val_ds, args.batch_size, device, args.seed)
        sched.step(vl)
        lr = opt.param_groups[0]["lr"]
        w.writerow([epoch, f"{tl / n:.4f}", f"{ta / n:.4f}", f"{vl:.4f}", f"{va:.4f}",
                    f"{lr:.2e}", f"{time.time() - te:.0f}"])
        log_f.flush()
        print(f"epoch {epoch:2d}  train {tl / n:.4f} (acc {ta / n:.3f})  val {vl:.4f} "
              f"(acc {va:.3f})  lr {lr:.1e}  {time.time() - te:.0f}s", flush=True)
        if vl < best - 1e-4:
            best, bad = vl, 0
            for path in (os.path.join("models", ckpt_name + ".pt"), os.path.join(outdir, "best_model.pt")):
                save_matcher(path, model, epoch=epoch, val_loss=vl, val_acc=va)
            print(f"  * new best ({vl:.4f})", flush=True)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop: no improvement for {args.patience} epochs", flush=True)
                break
    log_f.close()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader(open(log_path)))
    ep = [int(r["epoch"]) for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(ep, [float(r["train_loss"]) for r in rows], label="train")
    ax[0].plot(ep, [float(r["val_loss"]) for r in rows], label="val")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("InfoNCE loss"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(ep, [float(r["train_acc"]) for r in rows], label="train")
    ax[1].plot(ep, [float(r["val_acc"]) for r in rows], label="val")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("in-batch top-1 (partial query)"); ax[1].legend()
    ax[1].grid(alpha=.3)
    fig.suptitle(f"{args.arch}  ({n_params:,} params)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "train_curve.png"), dpi=100)
    print(f"done. best val loss {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
