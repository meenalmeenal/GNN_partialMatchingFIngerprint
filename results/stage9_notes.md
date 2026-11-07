# Stage 9 — model improvements

Goal: close the gap to the classical matcher and, in particular, make the GNN degrade
gracefully as the query print shrinks (the Stage 6 / 6b result: it collapsed to chance).

## Diagnosis that drove the design
Stage 4-6b pointed at three separable causes, so each was changed on its own:
1. **Loss**: triplet loss with one random negative is weak (only one negative per query).
2. **Features**: absolute (x, y, angle) node features, so rotated/shifted prints look different.
3. **Pooling**: mean/max pooling squeezes a whole print into one vector, so a small crop and
   its full print land far apart no matter how the model is trained (Stage 6b ruled out
   "never saw partial prints" as the main cause).

## What was built (`src/model_v2.py`, `src/train_v2.py`, `src/evaluate_v2.py`)
One `Matcher` class with three architectures, trained with identical data, augmentation
(random crop of the query only), optimiser and schedule:

| arch | features | readout | params |
|---|---|---|---|
| `v1_global` | Stage 3 GCN, absolute x/y/angle | mean+max pool -> 128-d vector | 25k |
| `inv_global` | **relative geometry only** (edge-conditioned GNN) | mean+max pool -> 64-d vector | 55k |
| `inv_local` | same relative-geometry GNN | **no pooling**: per-minutia descriptors, scored by **MaxSim** | 47k |

- **Invariant features** are computed inside the model from the stored node features: per
  edge, the distance and the neighbour bearing relative to each node's orientation, and the
  orientation difference, all as doubled angles (orientation is an undirected line; on matched
  minutiae it agrees as a direction 91.9 % of the time, as a line 93.8 %). Verified:
  scores change by < 2e-7 under rotation + translation + random orientation flips
  (the Stage 3 model changes by 1e-2); this is now part of `smoke_test.py`.
- **MaxSim matching**: score(Q, G) = mean over query minutiae of the best cosine match among
  gallery minutiae. A learned analogue of the classical correspondence count; a partial query
  only needs *its own* minutiae to find matches. Scoring is a batched matrix product (7,398
  probes x 900 gallery in 139 s on CPU), checked identical to a naive loop.
- **Loss**: in-batch InfoNCE: each partial query sees 63 other gallery prints as negatives
  (same-finger collisions masked), instead of one random negative.
- Training: query = random crop of 10-100 % of the print, gallery print whole (the test
  scenario). `inv_local_cont` = the `inv_local` run continued for 50 more epochs (warm start):
  after the first 40 epochs the learning rate had not yet decayed and validation loss was still
  falling.

## Results (200 test fingers, 1,670 probes, identical crops; chance Rank-1 = 0.5 %)
Figure: `results/stage9/headline.png`. Raw numbers: `results/stage9/sweep_metrics.json`.

**Rank-1 identification (%)**

| area kept | classical | Stage 3 GCN | triplet+crop (6b) | `v1_global` (loss only) | `inv_global` | `inv_local` (40 ep) | **`inv_local` (+50 ep)** |
|---|---|---|---|---|---|---|---|
| 100 % | **89.3** | 10.8 | 12.0 | 42.0 | 25.9 | 72.0 | 78.1 |
| 75 %  | **71.1** | 4.3 | 4.1 | 15.0 | 8.3 | 52.3 | 59.7 |
| 50 %  | 46.2 | 2.3 | 2.6 | 5.2 | 5.1 | 43.5 | **50.4** |
| 25 %  | 21.0 | 1.7 | 1.6 | 1.8 | 2.3 | 31.6 | **38.0** |
| 10 %  | 5.2 | 1.3 | 1.0 | 2.0 | 0.8 | 17.2 | **22.3** |

**Verification, EER % / AUC** (per-probe test-normalisation applied to classical and to every
Stage 9 model, see caveat 4; the two oldest GCN columns are raw)

| area | classical | `v1_global` | `inv_global` | **`inv_local` (+50 ep)** |
|---|---|---|---|---|
| 100 % | **4.9 / 0.978** | 11.0 / 0.958 | 16.2 / 0.914 | 7.6 / 0.973 |
| 75 %  | **12.9 / 0.929** | 20.2 / 0.878 | 27.7 / 0.798 | 14.5 / 0.923 |
| 50 %  | 21.2 / 0.861 | 28.4 / 0.786 | 32.6 / 0.741 | **19.8 / 0.882** |
| 25 %  | 30.7 / 0.761 | 34.5 / 0.707 | 37.6 / 0.670 | **27.1 / 0.813** |
| 10 %  | 38.4 / 0.665 | 36.8 / 0.681 | 41.4 / 0.613 | **34.3 / 0.734** |

**Full test protocol** (all 900 fingers, 7,398 altered probes, full prints, `inv_local` +50 ep;
`results/stage9/full_inv_local_cont/`): Rank-1 **73.1 %**, Rank-5 80.4 %, EER 7.9 % (19.1 % raw),
AUC 0.973 (0.893 raw). The Stage 3 model scored Rank-1 3.8 %, EER 17.1 % on the same protocol.
Per alteration, Rank-1: Easy-Zcut 97.8, Easy-CR 85.5, Medium-Zcut 84.3, Easy-Obl 79.4,
Hard-Zcut 74.3, Medium-CR 68.9, Hard-CR 59.7, Medium-Obl 58.9, Hard-Obl 47.4 %.

## Reading it
1. **The correspondence-style matcher is what fixes partial prints.** The three changes
   contribute very differently:
   - *Loss only* (`v1_global` vs the crop-augmented triplet model): full-print Rank-1 12.0 -> 42.0 %,
     but it still collapses on crops (5.2 % at 50 % area).
   - *Invariant features alone* (`inv_global`) did **not** help; it is worse than the
     absolute-coordinate model at 100 % area (25.9 vs 42.0 %). On SOCOFing the prints are not
     rotated relative to each other, so absolute position is genuinely informative to a pooled
     vector. Invariance only pays off once the readout is local.
   - *Removing the pooling* (`inv_local`) is the step change: 52 % -> 43 % -> 32 % -> 17 % Rank-1
     at 75/50/25/10 % area, against 8/5/2/1 % for the pooled `inv_global`.
2. **Against the classical matcher** (same 200 fingers): classical still wins on near-complete
   prints (89.3 vs 78.1 % at 100 %; 71.1 vs 59.7 % at 75 %). The learned matcher overtakes it
   at 50 % area and the lead grows as the print shrinks: 38.0 vs 21.0 % at 25 %, 22.3 vs 5.2 %
   (4x) at 10 %, and it has the lower EER from 50 % downwards. Retained Rank-1 from 100 % to
   10 % area: 29 % for the learned matcher, 6 % for classical. **The project's central
   hypothesis, that a learned graph embedding degrades more gracefully than classical matching
   as the print shrinks, is now supported for this model and protocol** (and was not supported
   by the Stage 3 model).
3. **Calibration matters for verification.** MaxSim is trained to *rank* within a gallery, so
   its raw scores are not comparable across queries (raw AUC 0.893 vs 0.973 after per-probe
   z-scoring). Identification is unaffected by normalisation.
4. Speed (CPU, not an apples-to-apples comparison: the classical matcher is single-threaded
   numpy, the learned scorer is batched torch on 8 threads): about 21 microseconds per
   probe-gallery pair vs about 470 microseconds for the classical matcher.

## Caveats (please read before quoting these numbers)
1. **SOCOFing probes are edited copies of the gallery impression**, so every unaltered region
   is pixel-identical and its extracted minutiae match exactly. All methods benefit; absolute
   numbers will be lower on independent captures of a finger (elastic distortion, different
   pressure). The learned model has never seen distortion and has no tolerance window like the
   classical matcher. **This makes the cross-dataset test (FVC, Stage 8) the most important
   open experiment**, and distortion augmentation (position/angle jitter, dropped and spurious
   minutiae) is the obvious next training change.
2. One training run per architecture, no hyper-parameter search, no confidence intervals. On
   1,670 probes the Rank-1 standard error is about +/-1 point at mid-range values; the
   architecture gaps are far larger than that, but small differences (e.g. `inv_local` vs
   classical at 50 %: 50.4 vs 46.2) are not decisive on their own.
3. Not converged: the continued run used its whole 50-epoch budget and validation loss was
   still decreasing slowly (2.62 -> 2.34). A larger variant (hidden 96, 3 layers) was started
   and stopped for cost reasons (CPU contention) and is **untested**.
4. Test-normalisation divides each probe's scores by statistics over the gallery, so verification
   thresholds depend on the gallery. It was applied identically to the classical matcher.
   The Stage 3 / triplet+crop columns in the verification table of `sweep_metrics.json` are raw.
5. The classical matcher is O(fingers^2) and was only run on the 200-finger subset, so the
   classical-vs-learned comparison is at 200 fingers; the learned model's 900-finger numbers
   are not directly comparable to the classical 200-finger numbers.
6. Training crops and evaluation crops use the same generator (square windows), differently
   seeded. Real latent prints have irregular shapes and noise.
7. MaxSim is directed (query -> gallery) and may favour gallery prints with more minutiae; not
   measured.

## Next
- Stage 7 (interactive interface) can now run on `models/stage9_inv_local_cont.pt` through
  `model_v2.load_matcher` / `score_matrix`; this is the first model worth demoing.
- Stage 8 (FVC cross-dataset) and a distortion-augmented retrain are the key risks to retire.
- Optional: larger/longer `inv_local`, hybrid re-ranking (learned shortlist, classical
  verification), Hard-Obl failure analysis.

## Reproduce
```bash
python src/train_v2.py --arch v1_global  --tag stage9_v1_global
python src/train_v2.py --arch inv_global --tag stage9_inv_global
python src/train_v2.py --arch inv_local  --tag stage9_inv_local
python src/train_v2.py --arch inv_local  --init models/stage9_inv_local.pt --tag stage9_inv_local_cont --epochs 50 --patience 6
python src/partiality_sweep.py --limit-fingers 200 --outdir results/stage9 \
    --models aug_crop=models/best_model_aug_crop.pt \
    --models-v2 v1_global=models/stage9_v1_global.pt inv_global=models/stage9_inv_global.pt \
                inv_local=models/stage9_inv_local.pt inv_local_cont=models/stage9_inv_local_cont.pt
python src/evaluate_v2.py --checkpoint models/stage9_inv_local_cont.pt --outdir results/stage9/full_inv_local_cont
python src/plot_stage9.py
```
