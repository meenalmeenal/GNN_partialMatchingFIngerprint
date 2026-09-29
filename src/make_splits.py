"""
Write a train/val/test split over finger identities to a JSON file, decoupled
from training so every stage (train, evaluate, experiments) reads the same split.

    python src/make_splits.py --graph_dir data/graphs --config configs/default.yaml
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml

from dataset import split_subjects


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_dir", default=None, help="defaults to config data.graph_dir")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="models/splits.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    graph_dir = args.graph_dir or cfg["data"]["graph_dir"]
    val_split = cfg["train"]["val_split"]
    test_split = cfg["train"]["test_split"]
    seed = cfg["train"]["seed"]

    train, val, test = split_subjects(graph_dir, val_split, test_split, seed=seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {"graph_dir": graph_dir, "seed": seed,
             "train": train, "val": val, "test": test}, f, indent=2)

    print(f"Wrote {args.out}")
    print(f"  train={len(train)}  val={len(val)}  test={len(test)}  "
          f"(total usable fingers={len(train) + len(val) + len(test)})")


if __name__ == "__main__":
    main()
