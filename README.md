# GNN Partial Fingerprint Matching

Match partial/damaged fingerprints against a full database using **Graph Convolutional Networks (GCNs)** over minutiae points — no image completion/inpainting required.

## Idea
1. Extract minutiae (ridge endings, bifurcations) from a fingerprint image.
2. Build a graph: minutiae = nodes, k-nearest-neighbor spatial links = edges.
3. Train a GCN-based Siamese network to embed graphs such that graphs from the *same finger* are close in embedding space, and different fingers are far apart.
4. Match a partial print's graph embedding against a gallery of full-print embeddings via cosine similarity / nearest neighbor.

## Pipeline
```
raw image --(minutiae extraction: MINDTCT / custom CNN)--> minutiae list (x, y, angle, type)
        --> k-NN graph construction --> PyG Data object
        --> GCN Siamese encoder --> embedding
        --> similarity search against gallery embeddings --> match / no-match
```

## Datasets
- **SOCOFing**: synthetic partial/obliterated/rotated prints — primary training set.
- **FVC2002 / FVC2004**: real sensor noise — robustness testing.
- **NIST SDs**: latent/partial forensic pairs — advanced evaluation (access-restricted).

## Repo structure
```
data/            raw & processed datasets (gitignored)
src/
  minutiae.py    minutiae extraction from images
  graph_build.py image/minutiae -> PyG graph objects
  dataset.py     PyG Dataset + pair/triplet sampling
  model.py       GCN Siamese / triplet network
  train.py       training loop
  match.py       gallery embedding + retrieval/matching
  evaluate.py    ROC, EER, Rank-1 accuracy metrics
configs/
  default.yaml   hyperparameters
notebooks/       exploration
models/          saved checkpoints (gitignored)
```

## Setup
```bash
pip install torch torchvision torch-geometric opencv-python scikit-image scikit-learn numpy pyyaml
```

## Quick start
```bash
# 0. sanity-check the whole pipeline on synthetic data (no dataset needed)
python src/smoke_test.py

# 1. organize raw SOCOFing into an <identity>/<sample> image tree
python src/prepare_socofing.py --input data/SOCOFing --output data/SOCOFing_organized --link

# 2. images -> k-NN minutiae graphs
python src/graph_build.py --input data/SOCOFing_organized --output data/graphs

# 3. train (writes models/best_model.pt + models/splits.json)
python src/train.py --config configs/default.yaml

# 4. evaluate on the held-out test subjects
python src/evaluate.py --gallery data/graphs --test data/graphs

# 5. match a single partial print
python src/match.py --partial path/to/partial.png --gallery data/graphs
```
Run scripts from the repo root; each entry point adds `src/` to `sys.path` itself.

## Status
🚧 Work in progress.

- **Stage 1 (data pipeline) — done.** SOCOFing organized into 6,000 finger identities;
  55,270 minutiae graphs built (0 failures); train/val/test split by finger
  (4,200 / 900 / 900) in `models/splits.json`. Report: `results/stage1_report.json`.
- **Stage 1.5 (extraction quality) — done.** Rewrote the classical minutiae extractor
  (segmentation mask, gradient orientation field, orientation-selective Gabor, spur
  pruning). Minutiae/print: median 32 (was ~198). See `src/tune_extraction.py` and
  `results/extraction_preview.png`.
- **Stage 2 (evaluation harness) — done.** `src/evaluate.py` now scores only the
  held-out test fingers, gallery = clean `Real` print, probes = altered prints.
  Reports Rank-1/5 + CMC, EER + ROC AUC, per-alteration breakdown, with plots.
  Untrained-model baseline (the floor to beat): Rank-1 9.7 %, EER 27.5 %
  (`results/stage2_notes.md`, `results/untrained_baseline/`).
- **Stage 3 (first training run) — done.** 3-layer GCN + triplet loss (random
  negatives), early-stopped at epoch 25. Test set vs untrained baseline:
  EER 27.5 % → **17.1 %**, ROC AUC 0.805 → **0.903** (verification improved);
  Rank-1 9.7 % → 3.8 % (identification regressed — random-negative loss + pooling
  smooths the same-impression positional signal). See `results/stage3_notes.md`.
  Fast in-RAM training via `src/pack_graphs.py` (2.5 s/step → 39 ms/step).
- **Next:** Stage 4 — embedding diagnostics.
