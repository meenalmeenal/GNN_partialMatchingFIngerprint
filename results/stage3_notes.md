# Stage 3 — first training run (baseline)

Config: `configs/default.yaml` — 3-layer GCN, hidden 64, embedding 128, k-NN(5) graphs,
triplet margin loss (**random negatives**), Adam lr 1e-3 + ReduceLROnPlateau, batch 32,
400 random triplet batches/epoch, early stop patience 8.

Data: packed splits (`data/packed/`), 4,200 train / 900 val / 900 test fingers (disjoint).
Runtime: ~20 s/epoch on CPU, early-stopped at epoch 25 (best val loss 0.303 @ epoch 17).

## Training curve (`results/stage3/train_curve.png`)
- train loss 0.70 → 0.35, val loss 0.44 → 0.30, both plateau.
- d(anchor, negative) jumps to ~1.3 by epoch 2 and stays; d(anchor, positive) stalls at
  ~0.40 and never improves → the model learns to repel negatives fast but cannot pull
  genuine pairs tight (underfitting on the positive side).

## Test-set results  (`results/stage3/eval_*`)

| metric | untrained baseline | **Stage 3 (trained)** |
|---|---|---|
| Rank-1 | 9.7 % | **3.8 %**  ↓ |
| Rank-5 | 19.3 % | 13.8 %  ↓ |
| EER | 27.5 % | **17.1 %**  ↓ (better) |
| ROC AUC | 0.805 | **0.903**  ↑ (better) |

Per-alteration Rank-1: Easy 3–9 %, Medium 3–5 %, Hard ~2 %.

## Reading it — training helped verification but hurt identification

- **Verification improved a lot** (EER 27→17 %, AUC 0.80→0.90). The score histogram went
  from a collapsed spike in [0.994, 1.0] to a real spread: genuine peaks at ~1.0,
  impostor roughly uniform over [-1, 1]. Average genuine/impostor separation is now good.
- **Rank-1 dropped below the untrained floor.** Two reasons:
  1. The untrained 9.7 % came from near-raw minutiae-*position* overlap (a probe is a
     perturbed copy of its gallery print). Mean/max pooling + a coarse training objective
     smooths that fine positional signal away in favour of class-separation features.
  2. **Random-negative** triplet loss never sees hard confusers, so it optimises average
     separation, not the top of the ranking. Rank-1 needs the true template to beat all
     899 others — and the impostor histogram still has a bump near 1.0, so for most probes
     a handful of wrong templates outscore the right one.

This is the expected weak first baseline and it points directly at the fixes:
hard-negative mining, better pooling / attention, edge features, rotation handling
(Stages 5, 6, 8). Verification AUC 0.90 is a reasonable starting point to improve on.

## Reproduce
```bash
python src/pack_graphs.py --graph_dir data/graphs --splits models/splits.json --out data/packed
python src/train.py --config configs/default.yaml --tag stage3 --epochs 50 --iters-per-epoch 400
python src/evaluate.py --checkpoint models/best_model.pt --splits models/splits.json --outdir results/stage3
```
