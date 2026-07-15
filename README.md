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
python src/graph_build.py --input data/SOCOFing --output data/graphs
python src/train.py --config configs/default.yaml
python src/match.py --partial path/to/partial.png --gallery data/graphs/gallery
```

## Status
🚧 Work in progress — scaffold stage.
