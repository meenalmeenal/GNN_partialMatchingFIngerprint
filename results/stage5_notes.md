# Stage 5 — classical baseline matcher

`src/classical_match.py` — the thing the GNN has to beat.

## Algorithm (generalized Hough alignment, standard minutiae matching)
1. For every same-type minutia pair (probe *i*, gallery *j*), compute the rigid
   transform (dx, dy, dθ) that maps *i* onto *j*.
2. Vote: bin all candidate transforms; the bin with the most votes is the consensus
   alignment between the two prints (this is what makes it rotation/translation
   invariant — SOCOFing's `CR` alteration is exactly a rotation).
3. Apply that transform to every probe minutia, greedily match to the nearest gallery
   minutia within tolerance (one-to-one).
4. Score = Dice coefficient = `2 * matched / (n_probe + n_gallery)`.

Minutiae are reconstructed directly from the existing graph node features (no
re-extraction needed): `x, y = feat[:2] * 160`, `theta = atan2(sin, cos)`.

## Scale note
Matching is O(fingers²) per query set (each probe compared against every gallery
template) and each comparison is itself O(n·m). Evaluated on a **200-finger subset**
of the test split (~0.5 ms/comparison, 1,670 probes × 200 templates ≈ 156 s). The GNN
was re-evaluated on the *same* 200-finger subset for a fair head-to-head (its full
900-finger numbers from Stage 3 remain the primary result elsewhere).

## Head-to-head (same 200 test fingers, same probes)

| metric | GNN (Stage 3) | **Classical (Hough)** |
|---|---|---|
| Rank-1 | 10.8 % | **89.3 %** |
| Rank-5 | 34.1 % | 94.0 % |
| EER | 18.0 % | **5.9 %** |
| ROC AUC | 0.899 | **0.971** |

Per-alteration Rank-1 (classical): Easy 96.5–99.5 %, Medium 86.3–97.4 %,
Hard 73.0–90.1 % (worst on `CR`/`Obl`, same difficulty ordering as the GNN, but every
number is far higher).

## Reading it

**The classical matcher wins decisively at this stage.** This is the expected and
useful outcome of Stage 5, not a failure — it isolates exactly what the GNN is missing:

1. **Rotation invariance.** Hough alignment explicitly solves for rotation before
   scoring. The GNN's node features are absolute `(x, y)` coordinates — a rotated
   probe produces a genuinely different input, and nothing in the architecture
   corrects for it.
2. **Precise local correspondence vs. global pooling.** The classical method matches
   individual minutiae one-to-one; the GNN collapses the whole graph into one 128-d
   vector via mean/max pooling, discarding exactly the fine positional detail that
   drives correct matches (consistent with the Stage 4 finding that Rank-1 losses are
   near-misses, not blowouts — the signal exists but gets blurred by pooling).

This directly prioritizes Stage 8: rotation-invariant features (or rotation
augmentation) and a matching-aware architecture (attention/correspondence pooling
instead of mean/max) are now the clear highest-value fixes, with a concrete target
to close the gap against (Rank-1 89 %, EER 5.9 %).

## Reproduce
```bash
python src/classical_match.py --splits models/splits.json --limit-fingers 200
python src/evaluate.py --checkpoint models/best_model.pt --splits models/splits.json --limit 200 --outdir results/stage5_gnn_subset
```
