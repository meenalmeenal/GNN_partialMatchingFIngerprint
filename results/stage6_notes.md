# Stage 6 — partiality sweep

`src/partiality_sweep.py`. The core research question: as less of the print is available,
how do the GCN and the classical Hough matcher degrade?

## Protocol
- Same 200 test fingers and 1,670 altered probes as Stage 5 (gallery = full clean `Real`
  print per finger; chance Rank-1 = 1/200 = 0.5 %).
- Each probe is cropped to a contiguous square window covering 100 / 75 / 50 / 25 / 10 % of
  the print area (window centred on a random minutia, clamped inside the image, resampled
  up to 25 times to reach >= 3 minutiae). The k-NN graph is rebuilt from the surviving
  minutiae. One crop per probe, seed 0; **identical crops for every method**.
- Methods: trained GCN (Stage 3 checkpoint, unchanged), untrained GCN (floor), classical
  Hough matcher. The gallery is never cropped.
- Minutiae per probe (mean): 32.6 / 24.6 / 17.6 / 10.1 / 5.4. No fallback crops were needed
  (min 3 minutiae at 50 % and below).

## Results  (`sweep_metrics.json`, `sweep_rank1.png`, `sweep_eer.png`)

Rank-1 identification (%):

| area kept | classical | GCN (trained) | GCN (untrained) |
|---|---|---|---|
| 100 % | **89.3** | 10.8 | 18.2 |
| 75 %  | **71.1** | 4.3 | 8.4 |
| 50 %  | **46.2** | 2.3 | 4.0 |
| 25 %  | **21.0** | 1.7 | 2.4 |
| 10 %  | **5.2**  | 1.3 | 1.7 |

EER (%): classical 5.9 → 14.0 → 23.4 → 33.1 → 39.9; trained GCN 18.0 → 33.2 → 40.2 →
43.9 → 44.1; untrained GCN 26.1 → 34.6 → 40.2 → 43.4 → 45.5.

## Reading it

**The hypothesis "the GCN degrades more gracefully than classical matching" is not
supported by the current model.** Classical degrades steadily but stays far ahead until
the very smallest crops (about 40x chance at 25 % area, about 10x chance at 10 %). The trained GCN
does not degrade gracefully, it collapses: Rank-1 falls to 4.3 % at 75 % area and is at
or near chance (0.5 %) from 50 % down. From 50 % downwards its EER is indistinguishable
from the untrained network's, i.e. the Stage 3 training bought **no** robustness to
partial queries. (At 100 % the untrained net even beats it on Rank-1, 18.2 % vs 10.8 % on
this 200-finger subset, consistent with the Stage 3 finding that training helped
verification but not identification.)

**Important caveat before drawing a thesis conclusion from this.** The GCN was trained
only on full-size prints (triplets of altered/real graphs with ~33 nodes). It has never
seen a cropped, ~5-18-node graph, so mean/max pooling over a much smaller node set shifts
the embedding off the distribution it learned. This sweep therefore measures "a model
trained on full prints, tested on partial prints", which is a fair statement about the
current model but is **not** a test of whether a GCN *can* handle partial prints. The
remaining stages are exactly what answers that: train with random-crop augmentation
(plus rotation augmentation and hard negatives, Stage 9) and re-run this sweep.

Other caveats:
- Classical scoring is Dice = 2m/(n+m), so a probe with few minutiae has a lower maximum
  score regardless of match quality. Ranking per probe (Rank-1/5) is unaffected, but pooled
  EER/AUC for the classical method at small areas is somewhat pessimistic.
- `graph_build.py` normalises coordinates by 256 although images are 160 px, and
  `classical_match.py` reconstructs with 160 (a uniform 0.625x shrink of both prints). The
  sweep crops in true pixel space (x_norm * 256) and keeps the classical convention
  unchanged, so the 100 % row reproduces Stage 5 exactly (89.3 / 10.8). Tolerances in the
  Hough matcher are therefore effectively ~1.6x looser in true pixels than their nominal
  values; worth cleaning up before the final report.
- One random crop per probe; no confidence intervals yet. With 1,670 probes the Rank-1
  standard error is about +/- 1 point at mid-range values, so the classical-vs-GCN gap is
  far outside noise, but the GCN-trained vs GCN-untrained differences below 50 % are not
  meaningful.

## Next
Stage 7 (interactive interface) can proceed against the current checkpoint. For the
research question, the highest-value next experiment is crop + rotation augmentation
training, then re-running `partiality_sweep.py` (a cheap precursor to full Stage 9).

## Reproduce
```bash
python src/partiality_sweep.py --limit-fingers 200 --outdir results/stage6
```
