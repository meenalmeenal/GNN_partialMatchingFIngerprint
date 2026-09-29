"""
Evaluate a trained encoder on the held-out TEST fingers only (from models/splits.json).

Protocol (damaged / partial matching):
  gallery  = the clean "Real" print of each test finger      (one enrolled template per finger)
  probes   = every altered print of each test finger          (CR / Obl / Zcut x Easy/Medium/Hard)

Metrics:
  - Rank-1 / Rank-5 closed-set identification accuracy (+ CMC curve)
  - Verification EER + ROC AUC (genuine = probe vs own template, impostor = probe vs other templates)
  - per-alteration Rank-1 breakdown

Outputs: results/eval_metrics.json, results/eval_roc.png, results/eval_cmc.png,
         results/eval_scores.png

    python src/evaluate.py --checkpoint models/best_model.pt --splits models/splits.json
    python src/evaluate.py --untrained --config configs/default.yaml --splits models/splits.json
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import yaml
from torch_geometric.data import Batch
from sklearn.metrics import roc_curve, auc

from model import TripletGCN
from match import load_model

GALLERY_TAG = "Real_real"


def build_untrained_model(config_path, device):
    cfg = yaml.safe_load(open(config_path))
    model = TripletGCN(
        in_dim=cfg["data"]["node_feat_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        embedding_dim=cfg["model"]["embedding_dim"],
        num_layers=cfg["model"]["num_gcn_layers"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    model.eval()
    return model, cfg


def collect(graph_dir, fingers, packed=None):
    """-> gallery {finger: Data|path}, probes [(Data|path, finger, tag)]"""
    gallery, probes = {}, []
    if packed is not None:
        data = packed if isinstance(packed, dict) else torch.load(packed, weights_only=False)
        for f in fingers:
            tagmap = data.get(f)
            if not tagmap or len(tagmap) < 2:
                continue
            gtag = GALLERY_TAG if GALLERY_TAG in tagmap else sorted(tagmap)[0]
            gallery[f] = tagmap[gtag]
            for t, g in tagmap.items():
                if t != gtag:
                    probes.append((g, f, t))
        return gallery, probes
    for f in fingers:
        d = os.path.join(graph_dir, f)
        if not os.path.isdir(d):
            continue
        samples = sorted(glob.glob(os.path.join(d, "*.pt")))
        if len(samples) < 2:
            continue
        g = os.path.join(d, GALLERY_TAG + ".pt")
        if not os.path.exists(g):
            g = samples[0]          # fallback: first sample is the enrolled template
        gallery[f] = g
        for p in samples:
            if p == g:
                continue
            probes.append((p, f, os.path.basename(p)[:-3]))
    return gallery, probes


@torch.no_grad()
def embed_paths(model, items, device, batch_size=256):
    """items: list of Data objects or list of .pt paths."""
    out = []
    for i in range(0, len(items), batch_size):
        chunk = [x if not isinstance(x, str) else torch.load(x, weights_only=False)
                 for x in items[i:i + batch_size]]
        batch = Batch.from_data_list(chunk).to(device)
        out.append(model.embed(batch).cpu())
    return torch.cat(out, 0) if out else torch.empty(0)


def identification(pe, pf, ge, gf, max_rank=20):
    sim = pe @ ge.t()                       # [P, G] cosine (embeddings are L2-normalised)
    order = sim.argsort(dim=1, descending=True)
    ranked = gf[order]                      # [P, G] finger ids in ranked order
    hits = (ranked == pf[:, None])
    rank_of = hits.float().argmax(dim=1)    # first True column
    found = hits.any(dim=1)
    cmc = [(found & (rank_of < r)).float().mean().item() for r in range(1, max_rank + 1)]
    return {"rank1": cmc[0], "rank5": cmc[4], "cmc": cmc}


def verification(pe, pf, ge, gf, seed=0):
    sim = pe @ ge.t()
    gf_np = gf.numpy()
    pf_np = pf.numpy()
    genuine, impostor = [], []
    rng = np.random.default_rng(seed)
    for i in range(sim.shape[0]):
        gi = np.where(gf_np == pf_np[i])[0]
        if len(gi) == 0:
            continue
        genuine.append(sim[i, gi[0]].item())
        neg = np.where(gf_np != pf_np[i])[0]
        for j in rng.choice(neg, size=min(20, len(neg)), replace=False):
            impostor.append(sim[i, j].item())
    scores = np.array(genuine + impostor)
    labels = np.array([1] * len(genuine) + [0] * len(impostor))
    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[idx] + fnr[idx]) / 2)
    return {"eer": eer, "auc": float(auc(fpr, tpr)), "threshold": float(thr[idx]),
            "n_genuine": len(genuine), "n_impostor": len(impostor)}, \
           (np.array(genuine), np.array(impostor), fpr, tpr)


def per_tag_rank1(pe, pf, ge, gf, tags):
    sim = pe @ ge.t()
    top = gf[sim.argmax(dim=1)]
    correct = (top == pf).numpy()
    out = {}
    for t in sorted(set(tags)):
        m = np.array([x == t for x in tags])
        out[t] = {"rank1": float(correct[m].mean()), "n": int(m.sum())}
    return out


def make_plots(genuine, impostor, fpr, tpr, cmc, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, lw=2)
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Accept Rate"); plt.ylabel("True Accept Rate")
    plt.title(f"Verification ROC (AUC={auc(fpr, tpr):.3f})")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "eval_roc.png"), dpi=100); plt.close()

    plt.figure(figsize=(6, 4))
    lo = float(min(genuine.min(), impostor.min()))
    hi = float(max(genuine.max(), impostor.max()))
    pad = max((hi - lo) * 0.05, 1e-6)
    bins = np.linspace(lo - pad, hi + pad, 60)
    plt.hist(impostor, bins=bins, alpha=0.6, label=f"impostor (n={len(impostor)})", density=True)
    plt.hist(genuine, bins=bins, alpha=0.6, label=f"genuine (n={len(genuine)})", density=True)
    plt.xlabel("cosine similarity"); plt.ylabel("density"); plt.legend()
    plt.title(f"Genuine vs impostor scores  (range [{lo:.4f}, {hi:.4f}])")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "eval_scores.png"), dpi=100); plt.close()

    plt.figure(figsize=(5, 4))
    plt.plot(range(1, len(cmc) + 1), cmc, marker="o", ms=3)
    plt.xlabel("rank"); plt.ylabel("identification rate"); plt.ylim(0, 1.02)
    plt.title("CMC curve"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "eval_cmc.png"), dpi=100); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="models/best_model.pt")
    ap.add_argument("--untrained", action="store_true",
                    help="evaluate a randomly-initialised model (sanity check)")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--splits", default="models/splits.json")
    ap.add_argument("--graph_dir", default=None)
    ap.add_argument("--packed", default="data/packed/test.pt",
                    help="packed test graphs; falls back to per-file graph_dir if missing")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--limit", type=int, default=0, help="cap #test fingers (debug)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.untrained:
        model, cfg = build_untrained_model(args.config, device)
    else:
        model, cfg = load_model(args.checkpoint, device)

    sp = json.load(open(args.splits))
    graph_dir = args.graph_dir or sp.get("graph_dir") or cfg["data"]["graph_dir"]
    fingers = sp["test"]
    if args.limit:
        fingers = fingers[:args.limit]

    packed = args.packed if (args.packed and os.path.exists(args.packed)) else None
    gallery, probes = collect(graph_dir, fingers, packed=packed)
    print(f"source: {'packed ' + args.packed if packed else 'per-file ' + graph_dir}")
    print(f"test fingers={len(fingers)}  gallery templates={len(gallery)}  probes={len(probes)}")

    gf_list = list(gallery.keys())
    fidx = {f: i for i, f in enumerate(gf_list)}
    ge = embed_paths(model, [gallery[f] for f in gf_list], device)
    gf = torch.tensor([fidx[f] for f in gf_list])

    pe = embed_paths(model, [p for p, _, _ in probes], device)
    pf = torch.tensor([fidx[f] for _, f, _ in probes])
    tags = [t for _, _, t in probes]

    ident = identification(pe, pf, ge, gf)
    ver, (gen, imp, fpr, tpr) = verification(pe, pf, ge, gf)
    tagbreak = per_tag_rank1(pe, pf, ge, gf, tags)

    os.makedirs(args.outdir, exist_ok=True)
    make_plots(gen, imp, fpr, tpr, ident["cmc"], args.outdir)

    metrics = {
        "mode": "untrained" if args.untrained else "trained",
        "checkpoint": None if args.untrained else args.checkpoint,
        "test_fingers": len(gallery), "probes": len(probes),
        "rank1": ident["rank1"], "rank5": ident["rank5"],
        "eer": ver["eer"], "auc": ver["auc"], "threshold": ver["threshold"],
        "n_genuine": ver["n_genuine"], "n_impostor": ver["n_impostor"],
        "per_alteration_rank1": tagbreak,
        "cmc": ident["cmc"],
    }
    json.dump(metrics, open(os.path.join(args.outdir, "eval_metrics.json"), "w"), indent=2)

    print(f"\nRank-1: {ident['rank1'] * 100:.2f}%   Rank-5: {ident['rank5'] * 100:.2f}%")
    print(f"EER:    {ver['eer'] * 100:.2f}%   ROC AUC: {ver['auc']:.4f}")
    print("per-alteration Rank-1:")
    for t, v in tagbreak.items():
        print(f"  {t:20s} {v['rank1'] * 100:5.1f}%  (n={v['n']})")
    print(f"\nwrote {args.outdir}/eval_metrics.json + eval_roc.png, eval_cmc.png, eval_scores.png")


if __name__ == "__main__":
    main()
