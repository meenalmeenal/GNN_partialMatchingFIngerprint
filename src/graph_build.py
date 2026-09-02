"""
Convert minutiae lists into PyTorch Geometric graph objects.

Node features: [x_norm, y_norm, sin(angle), cos(angle), type]
Edges: k-nearest-neighbor by Euclidean distance (undirected).
Edge features: [distance, relative_angle] (optional, used by edge-conditioned GCN variants).
"""
import argparse
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors

from minutiae import minutiae_from_image, Minutia


def minutiae_to_graph(minutiae: list[Minutia], k: int = 5, img_size=(256, 256)) -> Data:
    if len(minutiae) < 2:
        raise ValueError("Need at least 2 minutiae to build a graph")

    coords = np.array([[m.x, m.y] for m in minutiae], dtype=np.float32)
    w, h = img_size

    # Node features: normalized position, orientation (sin/cos), type
    feats = []
    for m in minutiae:
        feats.append([
            m.x / w,
            m.y / h,
            np.sin(m.angle),
            np.cos(m.angle),
            float(m.type),
        ])
    x = torch.tensor(feats, dtype=torch.float32)

    # k-NN edges (undirected, self-loops excluded)
    k_eff = min(k + 1, len(minutiae))  # +1 because point itself is its own nearest neighbor
    nbrs = NearestNeighbors(n_neighbors=k_eff).fit(coords)
    _, indices = nbrs.kneighbors(coords)

    edge_set = set()
    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:  # skip self
            edge_set.add((i, int(j)))
            edge_set.add((int(j), i))

    edges = sorted(edge_set)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Edge attributes: normalized distance + relative angle
    edge_attr = []
    for i, j in edges:
        dist = np.linalg.norm(coords[i] - coords[j]) / np.sqrt(w ** 2 + h ** 2)
        rel_angle = minutiae[i].angle - minutiae[j].angle
        edge_attr.append([dist, np.sin(rel_angle), np.cos(rel_angle)])
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(minutiae))


def _build_one(args):
    path, rel, out_path, k = args
    try:
        minutiae = minutiae_from_image(path)
        graph = minutiae_to_graph(minutiae, k=k)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(graph, out_path)
        return rel, len(minutiae), None
    except Exception as e:
        return rel, None, str(e)


def build_graph_dataset(input_dir: str, output_dir: str, k: int = 5, workers: int = 1):
    """
    Expects input_dir laid out as: input_dir/<subject_id>/<image>.png (SOCOFing-style).
    Saves one .pt file per image to output_dir/<subject_id>/<image>.pt
    """
    os.makedirs(output_dir, exist_ok=True)
    image_paths = glob.glob(os.path.join(input_dir, "**", "*.*"), recursive=True)
    image_paths = [p for p in image_paths if p.lower().endswith((".png", ".bmp", ".jpg", ".tif"))]

    tasks = []
    for path in image_paths:
        rel = os.path.relpath(path, input_dir)
        out_path = os.path.join(output_dir, os.path.splitext(rel)[0] + ".pt")
        tasks.append((path, rel, out_path, k))

    ok, failed = 0, 0
    failures, minutiae_counts = [], {}

    if workers and workers > 1:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            for i, (rel, n, err) in enumerate(pool.imap_unordered(_build_one, tasks, chunksize=16)):
                if err is None:
                    minutiae_counts[rel] = n
                    ok += 1
                else:
                    failures.append({"path": rel, "error": err})
                    failed += 1
                if (i + 1) % 2000 == 0:
                    print(f"  {i + 1}/{len(tasks)} processed ({ok} ok, {failed} failed)")
    else:
        for i, t in enumerate(tasks):
            rel, n, err = _build_one(t)
            if err is None:
                minutiae_counts[rel] = n
                ok += 1
            else:
                print(f"[skip] {rel}: {err}")
                failures.append({"path": rel, "error": err})
                failed += 1
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(tasks)} processed ({ok} ok, {failed} failed)")

    log = {
        "input_dir": input_dir, "k": k,
        "images": len(image_paths), "ok": ok, "failed": failed,
        "failure_rate_pct": round(100 * failed / len(image_paths), 2) if image_paths else None,
        "failures": failures,
        "minutiae_counts": minutiae_counts,
    }
    import json
    with open(os.path.join(output_dir, "_build_log.json"), "w") as f:
        json.dump(log, f, indent=2)

    print(f"Done. {ok} graphs built, {failed} failed. Log -> {os.path.join(output_dir, '_build_log.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Root dir of raw fingerprint images")
    parser.add_argument("--output", required=True, help="Where to save .pt graph files")
    parser.add_argument("--k", type=int, default=5, help="k for k-NN graph construction")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                        help="parallel worker processes")
    args = parser.parse_args()
    build_graph_dataset(args.input, args.output, k=args.k, workers=args.workers)
