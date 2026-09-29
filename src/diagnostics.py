"""
Stage 4 - embedding diagnostics on the trained encoder (test split only).

Answers: do same-finger embeddings actually cluster? is the embedding space spread
across identities, or collapsed? when Rank-1 fails, is it a near-miss or a blowout?

Outputs (results/stage4/):
  tsne_embeddings.png        - 2D projection of gallery + probe embeddings, colored by finger
  gallery_intra_similarity.png - pairwise cosine similarity among the 900 gallery templates
  score_hist_by_tag.png      - genuine-score distribution split by alteration tag
  rank_gap.png                - true-match score minus best-wrong-match score, correct vs incorrect
  stage4_metrics.json

    python src/diagnostics.py --checkpoint models/best_model.pt --splits models/splits.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from sklearn.manifold import TSNE

from match import load_model
from evaluate import collect, embed_paths, GALLERY_TAG


def gallery_intra_similarity(ge, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sim = (ge @ ge.t()).numpy()
    n = sim.shape[0]
    off_diag = sim[~np.eye(n, dtype=bool)]
    plt.figure(figsize=(6, 4))
    plt.hist(off_diag, bins=60, density=True, color="steelblue")
    plt.axvline(off_diag.mean(), color="k", ls="--", lw=1,
                label=f"mean={off_diag.mean():.3f}")
    plt.xlabel("cosine similarity between two different fingers' gallery templates")
    plt.ylabel("density"); plt.legend()
    plt.title("Gallery inter-identity similarity (lower/spread = more discriminative)")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "gallery_intra_similarity.png"), dpi=100)
    plt.close()
    return {"mean": float(off_diag.mean()), "std": float(off_diag.std()),
            "p90": float(np.percentile(off_diag, 90)), "max": float(off_diag.max())}


def tsne_plot(ge, gf, gf_ids, pe, pf, tags, outdir, n_fingers=35, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(gf_ids), size=min(n_fingers, len(gf_ids)), replace=False)
    chosen_set = set(chosen.tolist())

    g_mask = np.isin(gf.numpy(), list(chosen_set))
    p_mask = np.isin(pf.numpy(), list(chosen_set))

    emb = torch.cat([ge[g_mask], pe[p_mask]], 0).numpy()
    finger_id = np.concatenate([gf.numpy()[g_mask], pf.numpy()[p_mask]])
    kind = np.array(["gallery"] * int(g_mask.sum()) + ["probe"] * int(p_mask.sum()))

    proj = TSNE(n_components=2, perplexity=15, init="pca", random_state=seed).fit_transform(emb)

    cmap = plt.get_cmap("tab20")
    color_of = {fid: cmap(i % 20) for i, fid in enumerate(sorted(chosen_set))}
    colors = [color_of[f] for f in finger_id]

    plt.figure(figsize=(7, 7))
    is_g = kind == "gallery"
    plt.scatter(proj[is_g, 0], proj[is_g, 1], c=[colors[i] for i in np.where(is_g)[0]],
               marker="*", s=220, edgecolors="black", linewidths=0.6, label="gallery (Real)", zorder=3)
    plt.scatter(proj[~is_g, 0], proj[~is_g, 1], c=[colors[i] for i in np.where(~is_g)[0]],
               marker="o", s=28, alpha=0.75, label="probe (altered)", zorder=2)
    plt.legend(loc="upper right")
    plt.title(f"t-SNE of test embeddings, {len(chosen_set)} fingers "
             "(star=gallery, dot=probe, color=finger identity)")
    plt.xticks([]); plt.yticks([])
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "tsne_embeddings.png"), dpi=110)
    plt.close()


def score_hist_by_tag(pe, pf, ge, gf, tags, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sim = (pe @ ge.t()).numpy()
    gf_np = gf.numpy(); pf_np = pf.numpy()
    genuine = np.array([sim[i, np.where(gf_np == pf_np[i])[0][0]] for i in range(len(pf_np))])
    tags_arr = np.array(tags)
    difficulty_order = ["Real", "Easy", "Medium", "Hard"]

    def sev(t):
        for i, d in enumerate(difficulty_order):
            if d in t:
                return i
        return -1

    sevs = np.array([sev(t) for t in tags])
    plt.figure(figsize=(7, 4.5))
    for s in sorted(set(sevs)):
        if s < 0:
            continue
        vals = genuine[sevs == s]
        plt.hist(vals, bins=40, alpha=0.55, density=True,
                label=f"{difficulty_order[s]} (n={len(vals)}, mean={vals.mean():.3f})")
    plt.xlabel("genuine cosine similarity (probe vs its own template)")
    plt.ylabel("density"); plt.legend()
    plt.title("Genuine score by alteration severity")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "score_hist_by_tag.png"), dpi=100)
    plt.close()

    return {difficulty_order[s]: {"mean": float(genuine[sevs == s].mean()),
                                  "std": float(genuine[sevs == s].std())}
           for s in sorted(set(sevs)) if s >= 0}


def rank_gap_analysis(pe, pf, ge, gf, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sim = (pe @ ge.t()).numpy()
    gf_np = gf.numpy(); pf_np = pf.numpy()
    n = sim.shape[0]
    true_score = np.empty(n)
    best_wrong = np.empty(n)
    correct = np.empty(n, dtype=bool)
    for i in range(n):
        ti = np.where(gf_np == pf_np[i])[0][0]
        true_score[i] = sim[i, ti]
        row = sim[i].copy()
        row[ti] = -2.0
        best_wrong[i] = row.max()
        correct[i] = true_score[i] >= row.max()
    gap = true_score - best_wrong

    lo, hi = np.percentile(gap, [1, 99])
    bins = np.linspace(min(lo, -0.05), max(hi, 0.05), 60)
    plt.figure(figsize=(7, 4.5))
    plt.hist(gap[correct], bins=bins, alpha=0.6, label=f"correct top-1 (n={correct.sum()})", density=True)
    plt.hist(gap[~correct], bins=bins, alpha=0.6, label=f"incorrect top-1 (n={(~correct).sum()})", density=True)
    n_clipped = int(((gap < bins[0]) | (gap > bins[-1])).sum())
    plt.axvline(0, color="k", lw=1)
    plt.xlabel("true-match score - best wrong-match score"
              + (f"  ({n_clipped} outliers outside view)" if n_clipped else ""))
    plt.ylabel("density")
    plt.legend(); plt.title("Rank-1 decision margin")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "rank_gap.png"), dpi=100)
    plt.close()

    wrong_gap = gap[~correct]
    near_miss = float((wrong_gap > -0.02).mean()) if len(wrong_gap) else None
    return {"rank1": float(correct.mean()), "median_gap_correct": float(np.median(gap[correct])) if correct.any() else None,
            "median_gap_incorrect": float(np.median(wrong_gap)) if len(wrong_gap) else None,
            "frac_incorrect_within_0.02": near_miss}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="models/best_model.pt")
    ap.add_argument("--splits", default="models/splits.json")
    ap.add_argument("--packed", default="data/packed/test.pt")
    ap.add_argument("--graph_dir", default=None)
    ap.add_argument("--outdir", default="results/stage4")
    args = ap.parse_args()

    device = torch.device("cpu")
    model, cfg = load_model(args.checkpoint, device)
    sp = json.load(open(args.splits))
    graph_dir = args.graph_dir or sp.get("graph_dir") or cfg["data"]["graph_dir"]
    fingers = sp["test"]

    packed = args.packed if (args.packed and os.path.exists(args.packed)) else None
    gallery, probes = collect(graph_dir, fingers, packed=packed)
    print(f"test fingers={len(fingers)}  gallery={len(gallery)}  probes={len(probes)}")

    gf_ids = list(gallery.keys())
    fidx = {f: i for i, f in enumerate(gf_ids)}
    ge = embed_paths(model, [gallery[f] for f in gf_ids], device)
    gf = torch.tensor([fidx[f] for f in gf_ids])
    pe = embed_paths(model, [p for p, _, _ in probes], device)
    pf = torch.tensor([fidx[f] for _, f, _ in probes])
    tags = [t for _, _, t in probes]

    os.makedirs(args.outdir, exist_ok=True)
    gstats = gallery_intra_similarity(ge, args.outdir)
    print("gallery inter-identity similarity:", gstats)

    tsne_plot(ge, gf, gf_ids, pe, pf, tags, args.outdir)
    print("t-SNE plot written")

    sev_stats = score_hist_by_tag(pe, pf, ge, gf, tags, args.outdir)
    print("genuine score by severity:", sev_stats)

    gap_stats = rank_gap_analysis(pe, pf, ge, gf, args.outdir)
    print("rank-1 gap analysis:", gap_stats)

    metrics = {"gallery_inter_identity_similarity": gstats,
              "genuine_score_by_severity": sev_stats,
              "rank1_gap": gap_stats}
    json.dump(metrics, open(os.path.join(args.outdir, "stage4_metrics.json"), "w"), indent=2)
    print(f"\nwrote {args.outdir}/stage4_metrics.json + tsne_embeddings.png, "
         "gallery_intra_similarity.png, score_hist_by_tag.png, rank_gap.png")


if __name__ == "__main__":
    main()
