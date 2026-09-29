# Stage 4 — embedding diagnostics

Analysis of the Stage 3 checkpoint on the 900 test fingers. Code: `src/diagnostics.py`,
interactive version: `notebooks/01_diagnostics.ipynb`.

## 1. Do same-finger embeddings actually cluster? — yes, loosely

`results/stage4/tsne_embeddings.png`: a t-SNE of 35 random test fingers (gallery
template = star, its altered probes = dots, colored by finger). Same-colored points
mostly sit together, confirming the GCN has learned real identity structure — but
clusters visibly overlap at their edges, and a few probes land in the wrong cluster
entirely. This is consistent with, not contradictory to, the weak Rank-1 from Stage 3.

## 2. Is the embedding space spread across identities? — mostly, but with a bad tail

`results/stage4/gallery_intra_similarity.png`: pairwise cosine similarity between all
900 gallery templates (different fingers only).

| | value |
|---|---|
| mean | 0.031 (~orthogonal on average — good) |
| std | 0.610 |
| p90 | 0.885 |
| max | ~1.000 |

The **average** separation between identities is good, but the distribution is wide
with a sharp spike near 1.0: a meaningful fraction of *different* fingers already sit
almost on top of each other in embedding space, before any probe is even involved. With
899 competing templates per query, hitting one of these confusable pairs by chance is
likely — this alone caps Rank-1 independent of probe/alteration quality.

## 3. Genuine scores are strong and behave sensibly

`results/stage4/score_hist_by_tag.png`: mean genuine similarity (probe vs its own
template) is 0.890 (Easy) → 0.853 (Medium) → 0.830 (Hard) — the expected ordering, and
all substantially above the ~0 mean impostor similarity. The verification signal
(EER 17%, AUC 0.90 from Stage 3) is real, not noise.

## 4. Rank-1 failures are mostly near-misses, not blowouts

`results/stage4/rank_gap.png` — for every probe, `margin = true-match score - best
wrong-match score`:

| | |
|---|---|
| correct top-1 (n=284) | median margin **+0.0006** — wins are razor-thin |
| incorrect top-1 (n=7114) | median margin -0.046; **35.7%** are within 0.02 of flipping |

**This is the key finding.** Even the *correct* predictions barely win, and over a
third of the *incorrect* ones are near-coin-flips against the best wrong template, not
catastrophic mismatches. Combined with finding 2 (a heavy-tailed impostor distribution),
the picture is: the encoder has learned genuine identity signal, but doesn't separate it
sharply enough to survive 899-way competition. This is a **separation/margin problem**,
not a "the model learned nothing" problem — and it is directly attackable by:
- hard-negative mining (train against the confusable pairs identified here, Stage 8)
- a larger or better-regularized embedding space / different pooling
- addressing the specific near-duplicate identity pairs (worth inspecting individually)

## Reproduce
```bash
python src/diagnostics.py --checkpoint models/best_model.pt --splits models/splits.json
# or open notebooks/01_diagnostics.ipynb
```
