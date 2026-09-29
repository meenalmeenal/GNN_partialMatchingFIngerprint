"""
End-to-end smoke test with synthetic data - no dataset required.

Verifies that graph construction, the dataset/sampler, the model, the training
loop, gallery matching and evaluation all run and wire together correctly.

    python src/smoke_test.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import yaml

from minutiae import Minutia, extract_minutiae, binarize_and_skeletonize
from graph_build import minutiae_to_graph
from dataset import FingerprintTripletDataset, triplet_collate, split_subjects
from train import train
from match import load_model, build_gallery, rank_matches, embed_graph_file
from evaluate import (collect as ev_collect, embed_paths as ev_embed,
                      identification as ev_identification, verification as ev_verification)


def synth_minutiae(rng, n, jitter=0.0, drop=0.0):
    pts = []
    for _ in range(n):
        if rng.random() < drop:
            continue
        pts.append(Minutia(
            x=int(np.clip(rng.normal(128, 60) + rng.normal(0, jitter), 5, 250)),
            y=int(np.clip(rng.normal(128, 60) + rng.normal(0, jitter), 5, 250)),
            angle=float(rng.uniform(-np.pi, np.pi)),
            type=int(rng.random() < 0.4),
        ))
    return pts


def main():
    tmp = tempfile.mkdtemp(prefix="fp_smoke_")
    graph_dir = os.path.join(tmp, "graphs")
    rng = np.random.default_rng(0)

    n_subjects, samples_per = 12, 5
    for s in range(n_subjects):
        base = synth_minutiae(rng, rng.integers(25, 40))
        for k in range(samples_per):
            # same finger -> jittered / partially-occluded copies of the base minutiae
            pts = [Minutia(int(np.clip(m.x + rng.normal(0, 4), 5, 250)),
                           int(np.clip(m.y + rng.normal(0, 4), 5, 250)),
                           m.angle + rng.normal(0, 0.1), m.type)
                   for m in base if rng.random() > 0.25]
            g = minutiae_to_graph(pts, k=5)
            d = os.path.join(graph_dir, f"subj{s:02d}")
            os.makedirs(d, exist_ok=True)
            torch.save(g, os.path.join(d, f"sample{k}.pt"))
    print(f"[1/5] built synthetic graphs for {n_subjects} subjects -> {graph_dir}")

    # minutiae extraction path (on a synthetic striped pattern)
    img = ((np.sin(np.linspace(0, 30 * np.pi, 256))[None, :] * np.ones((256, 1))) > 0).astype(np.uint8) * 255
    skel = binarize_and_skeletonize(img)
    _ = extract_minutiae(skel)
    print("[2/5] minutiae extraction ran on a synthetic image")

    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")))
    cfg["data"]["graph_dir"] = graph_dir
    cfg["data"]["packed_dir"] = os.path.join(tmp, "nopack")   # force per-file path
    cfg["train"].update(epochs=2, batch_size=8, iters_per_epoch=5, val_iters=3,
                        checkpoint_dir=os.path.join(tmp, "models"))
    os.chdir(tmp)                                             # results/<tag>/ lands in tmp
    train(cfg, tag="smoke")
    print("[3/5] training loop completed, checkpoint written")

    device = torch.device("cpu")
    model, _ = load_model(os.path.join(tmp, "models", "best_model.pt"), device)
    gallery = build_gallery(model, graph_dir, device)
    q = embed_graph_file(model, next(iter(
        __import__("glob").glob(os.path.join(graph_dir, "**", "*.pt"), recursive=True))), device)
    top = rank_matches(q, gallery, top_k=3)
    assert top and len(top[0]) == 2
    print(f"[4/5] gallery match ran, top-1 = {top[0]}")

    splits = json.load(open(os.path.join(tmp, "models", "splits.json")))
    test_fingers = splits["test"] or splits["val"] or splits["train"]
    gal, probes = ev_collect(graph_dir, test_fingers)
    gfl = list(gal)
    fidx = {f: i for i, f in enumerate(gfl)}
    ge = ev_embed(model, [gal[f] for f in gfl], device)
    gf = torch.tensor([fidx[f] for f in gfl])
    pe = ev_embed(model, [p for p, _, _ in probes], device)
    pf = torch.tensor([fidx[f] for _, f, _ in probes])
    ident = ev_identification(pe, pf, ge, gf)
    ver, _ = ev_verification(pe, pf, ge, gf)
    print(f"[5/5] evaluate ran: rank-1={ident['rank1']*100:.1f}%  EER={ver['eer']*100:.1f}%")

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
