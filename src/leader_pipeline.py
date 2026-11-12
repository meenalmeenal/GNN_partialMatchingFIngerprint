"""
Stage 9d: rebuild every minutiae graph with the pretrained LEADER detector (pyfing) instead of the
hand-built extractor. Same graph format (x/256 coordinates, [x, y, sin, cos, type], k=5 edges), so the
models, matchers and evaluation code are unchanged.

  socofing : data/SOCOFing_organized/<finger>/<tag>.BMP -> 160x160 (as before) -> LEADER
             -> data/packed_leader/{train,val,test}.pt            ({finger: {tag: Data}}, splits.json)
  fvc      : FVC set A (and set B for tuning) -> per-database fixed ridge-period scale (Stage 8)
             -> LEADER -> data/fvc_packed_leader[_b]/<db>.pt        ({finger: {impression: Data}})

Detection threshold: all candidates with quality >= 0.3 (best repeatability on FVC set B, see
extractor_dev.py); a print with fewer than MIN_KEEP such minutiae keeps its best MIN_KEEP candidates
(quality >= 0.1), so a damaged print still yields a graph. Capped at the best 100 (SOCOFing) /
200 (FVC) by quality. Direction comes from the network (a true direction, not an undirected line).

    KERAS_BACKEND=torch python src/leader_pipeline.py socofing
    KERAS_BACKEND=torch python src/leader_pipeline.py fvc
"""
import argparse
import glob
import json
import os
import sys
import time

os.environ.setdefault("KERAS_BACKEND", "torch")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import torch

from augment import build_graph_fast

torch.set_grad_enabled(False)
Q_KEEP, Q_FLOOR, MIN_KEEP = 0.3, 0.1, 6
FVC_ROOT = "data/FVC/74034_3_En_4_MOESM1_ESM"


def get_leader():
    import pyfing as pf
    return pf.Leader(pf.LeaderParameters(minutia_quality_threshold=Q_FLOOR))


def to_graph(mins, cap):
    """LEADER minutiae -> Data (None if fewer than 3 usable minutiae)."""
    m = [(a.x, a.y, a.direction, 0 if a.type == "E" else 1, a.quality) for a in mins if a.type in ("E", "B")]
    if not m:
        return None
    m.sort(key=lambda r: -r[4])
    keep = [r for r in m if r[4] >= Q_KEEP]
    if len(keep) < MIN_KEEP:
        keep = m[:MIN_KEEP]
    keep = keep[:cap]
    if len(keep) < 3:
        return None
    a = np.array(keep, dtype=np.float64)
    return build_graph_fast(a[:, 0], a[:, 1], a[:, 2], a[:, 3])


def run_chunked(alg, images, bs=64):
    return alg.run_on_db(images, batch_size=bs)


def build_socofing(args):
    sp = json.load(open("models/splits.json"))
    alg = get_leader()
    os.makedirs(args.out_socofing, exist_ok=True)
    for split in ("test", "val", "train"):
        out, n_img, n_fail, t0 = {}, 0, 0, time.time()
        fingers = sp[split][: args.limit] if args.limit else sp[split]
        for lo in range(0, len(fingers), 100):          # chunks of 100 fingers (~900 images)
            keys, imgs = [], []
            for f in fingers[lo:lo + 100]:
                for p in sorted(glob.glob(os.path.join("data/SOCOFing_organized", f, "*.BMP"))):
                    g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                    keys.append((f, os.path.splitext(os.path.basename(p))[0]))
                    imgs.append(cv2.resize(g, (160, 160), interpolation=cv2.INTER_CUBIC))
            for (f, tag), mins in zip(keys, run_chunked(alg, imgs)):
                n_img += 1
                d = to_graph(mins, 100)
                if d is None:
                    n_fail += 1
                    continue
                out.setdefault(f, {})[tag] = d
            print(f"  {split}: {lo + 100}/{len(fingers)} fingers, {n_img} prints, {n_fail} without a graph, "
                  f"{time.time() - t0:.0f}s", flush=True)
        # the gallery print must exist for every finger, otherwise the finger cannot be evaluated
        missing = [f for f in fingers if "Real_real" not in out.get(f, {})]
        torch.save(out, os.path.join(args.out_socofing, f"{split}.pt"))
        nodes = [d.num_nodes for v in out.values() for d in v.values()]
        print(f"{split}: {len(out)} fingers, {len(nodes)} graphs, minutiae/print median {np.median(nodes):.0f} "
              f"(p10 {np.percentile(nodes, 10):.0f}, p90 {np.percentile(nodes, 90):.0f}); "
              f"fingers lacking a Real_real graph: {len(missing)}", flush=True)


def parse_name(path):
    import re
    m = re.match(r"^(\d+)_(\d+)\.", os.path.basename(path))
    return (int(m.group(1)), int(m.group(2))) if m else None


def build_fvc(args):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from demo_core import fvc_scales
    from fvc_pipeline import calibrate_target_period
    scales = fvc_scales(calibrate_target_period())
    alg = get_leader()
    for suffix, out_dir, fmax in (("a", "data/fvc_packed_leader", 100), ("b", "data/fvc_packed_leader_b", 110)):
        os.makedirs(out_dir, exist_ok=True)
        for y in (2002, 2004):
            for db in (1, 2, 3, 4):
                name = f"fvc{y}_db{db}"
                paths = [p for p in sorted(glob.glob(f"{FVC_ROOT}/FVC{y}/Dbs/Db{db}_{suffix}/*.tif")) if parse_name(p)]
                imgs = [cv2.resize(cv2.imread(p, cv2.IMREAD_GRAYSCALE), None, fx=scales[name], fy=scales[name],
                                   interpolation=cv2.INTER_CUBIC) for p in paths]
                out, fail, t0 = {}, 0, time.time()
                for lo in range(0, len(imgs), 128):
                    for p, mins in zip(paths[lo:lo + 128], run_chunked(alg, imgs[lo:lo + 128], 4)):
                        d = to_graph(mins, 200)
                        f, i = parse_name(p)
                        if d is None:
                            fail += 1
                            continue
                        out.setdefault(f, {})[i] = d
                nodes = [d.num_nodes for v in out.values() for d in v.values()]
                torch.save(out, os.path.join(out_dir, name + ".pt"))
                print(f"{name}_{suffix}: {len(nodes)} graphs, {fail} failed, minutiae/print median "
                      f"{np.median(nodes):.0f} (p10 {np.percentile(nodes, 10):.0f}, p90 {np.percentile(nodes, 90):.0f}) "
                      f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["socofing", "fvc"])
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N fingers per split")
    ap.add_argument("--out-socofing", default="data/packed_leader")
    args = ap.parse_args()
    {"socofing": build_socofing, "fvc": build_fvc}[args.what](args)
