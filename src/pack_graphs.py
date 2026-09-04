"""
Consolidate the per-file .pt graphs into one packed file per split.

Loading 55k tiny pickles at train time costs minutes of disk I/O per epoch start;
a single packed file loads in ~1-2 s. Run once after graph_build + make_splits.

    python src/pack_graphs.py --graph_dir data/graphs --splits models/splits.json --out data/packed

Each output file (train.pt / val.pt / test.pt) is a dict:
    { finger_id: { tag: torch_geometric.data.Data } }
where tag is the sample name (e.g. "Real_real", "Altered-Hard_Obl").
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch


def pack_split(graph_dir, fingers, out_path):
    packed, n = {}, 0
    t = time.time()
    for f in fingers:
        d = os.path.join(graph_dir, f)
        if not os.path.isdir(d):
            continue
        samples = {}
        for p in glob.glob(os.path.join(d, "*.pt")):
            tag = os.path.basename(p)[:-3]
            samples[tag] = torch.load(p, weights_only=False)
            n += 1
        if samples:
            packed[f] = samples
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(packed, out_path)
    print(f"  {out_path}: {len(packed)} fingers, {n} graphs, {time.time() - t:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_dir", default="data/graphs")
    ap.add_argument("--splits", default="models/splits.json")
    ap.add_argument("--out", default="data/packed")
    args = ap.parse_args()

    sp = json.load(open(args.splits))
    for name in ("train", "val", "test"):
        pack_split(args.graph_dir, sp[name], os.path.join(args.out, f"{name}.pt"))
    print("done")


if __name__ == "__main__":
    main()
