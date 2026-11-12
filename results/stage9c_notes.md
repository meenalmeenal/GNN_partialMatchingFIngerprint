# Stage 9c — evaluation fixes (900-finger sweep, open-set, tuned classical baseline)

Why: the Stage 7 demo gave much lower partial-print numbers than the Stage 9 sweep, and the
assumptions list showed several evaluation shortcuts (200-finger gallery, closed-set only, single
probe/crop draw, classical matcher never tuned for FVC). This stage fixes those that are cheap.
No model was retrained; the model is `stage9_inv_local_cont` (SOCOFing) throughout.

## What was run
- `src/eval_fixes.py`: closed-set sweep on the **full 900-finger test gallery**, one random altered probe per
  finger (900 queries), 5 print areas, two window-placement modes, 95 % bootstrap CIs over queries, learned model
  repeated over 3 probe/crop seeds; plus an **open-set** test (450 fingers enrolled, 450 unknown, 2 random
  enrolment splits; accept if the best score exceeds a threshold). Raw score matrices are cached in `data/cache/`.
- `src/tune_classical.py`: grid search of the Hough tolerances (120 settings) on **FVC set B** (fingers 101-110,
  8 databases) only; then `eval_fvc.py --classical-tuned` on the held-out set-A fingers 51-100.
- `src/plot_stage9c.py` -> `results/stage9c/summary_tables.md`, `headline.png`. Raw numbers: `results/stage9c/*.json`.

## Results
**Closed-set Rank-1 %, 900-print gallery (95 % CI over queries; learned mean ± sd over 3 seeds in brackets)**

| windows | method | 100 % | 75 % | 50 % | 25 % | 10 % |
|---|---|---|---|---|---|---|
| minutia-centred | learned | 72.0 [69-75] (71.9 ± 0.4) | 53.0 [50-56] (53.9 ± 0.8) | 42.7 [40-46] (42.1 ± 0.7) | 33.0 [30-36] (30.6 ± 2.1) | 18.3 [16-21] (18.1 ± 1.7) |
| minutia-centred | classical | 83.3 [81-86] | 64.3 [61-67] | 37.8 [35-41] | 13.2 [11-15] | 1.4 [1-2] |
| uniform-random | learned | 72.0 | 24.7 [22-27] (26.0 ± 1.3) | 14.9 [12-17] (15.1 ± 0.4) | 6.2 [5-8] (7.4 ± 1.1) | 2.3 [1-3] (2.7 ± 0.7) |
| uniform-random | classical | 83.3 | 39.1 [36-42] | 13.7 [11-16] | 2.2 [1-3] | 1.0 [0-2] |

**Open-set (z-scored top score), minutia-centred windows. DIR = enrolled query correctly identified at rank 1 and accepted,
at the threshold giving the stated false-accept rate on unknown queries**

| method | metric | 100 % | 75 % | 50 % | 25 % | 10 % |
|---|---|---|---|---|---|---|
| learned | DIR @ FAR 1 % | 57.6 | 39.8 | 28.0 | 18.3 | 9.3 |
| learned | DIR @ FAR 10 % | 66.4 | 50.2 | 38.7 | 26.0 | 14.6 |
| classical | DIR @ FAR 1 % | 49.4 | 31.0 | 7.6 | 1.0 | 0.1 |
| classical | DIR @ FAR 10 % | 73.6 | 55.3 | 27.0 | 5.8 | 0.6 |

**FVC held-out fingers 51-100, mean of the 6 real-sensor databases, Rank-1 %**

| method | 100 % | 75 % | 50 % | 25 % | 10 % |
|---|---|---|---|---|---|
| classical, default tolerances (SOCOFing-tuned) | 17.1 | 15.2 | 13.0 | 8.4 | 4.9 |
| classical, tuned on set B | 21.4 | 20.0 | 14.1 | 8.5 | 4.2 |
| learned, fine-tuned on FVC | 21.4 | 19.6 | 18.1 | 14.2 | 7.5 |
| learned, SOCOFing only | 13.0 | 12.3 | 11.0 | 9.6 | 6.2 |

Verification EER % (cohort-normalised) at 100 / 25 % area: classical default 43.4 / 46.6, classical tuned 40.1 / 44.2,
learned fine-tuned 36.5 / 40.3. Tuned tolerances: xy_tol 26, ang_tol 0.5, xy_bin 14, ang_bin 0.4 (default 12 / 0.5 / 8 / 0.35);
on set B pooled AUC 0.614 -> 0.667, Rank-1 29 -> 43 % (10-finger galleries). Figure: `results/stage9c/headline.png`.

## Reading it
1. **The Stage 7 gap is mostly window placement, not gallery size.** Same model, same 900-print gallery: at 50 % area
   Rank-1 is 42.7 % with windows centred on a minutia and 14.9 % with uniform random windows. A window placed over a
   sparse or empty region leaves too few minutiae to identify anyone. The Stage 9 headline (50 % at 50 % area, 200 fingers)
   is consistent with 42.7 % at 900 fingers, so the relative conclusions survive the 4.5x larger gallery.
   The demo's numbers described a harsher partial-print model than the sweep; both are now documented.
2. **Learned vs classical, with CIs.** Classical is clearly better on full and 75 % prints (83 vs 72 %, 64 vs 53 %; CIs do
   not overlap). At 50 % the CIs overlap (42.7 vs 37.8). Below that the learned matcher is clearly ahead
   (33.0 vs 13.2 % at 25 %, 18.3 vs 1.4 % at 10 %). With random windows both collapse and the learned advantage is small
   (6.2 vs 2.2 % at 25 %).
3. **Seed spread is small** (sd 0.4-2.1 points over probe/crop seeds), far below the learned-vs-classical gaps at 25 % and
   10 %. This is evaluation-sampling variance only: **training-seed variance is not measured** (see Not done).
4. **Open-set works, with a calibration caveat.** Rejecting unknown prints is possible, but how well depends on the matcher:
   at full area the classical matcher is better at 10 % FAR (73.6 vs 66.4 %), the learned one at 1 % FAR (57.6 vs 49.4 %),
   and the learned matcher is clearly ahead at <= 50 % area. The learned matcher's raw MaxSim score is poorly calibrated across
   queries (DIR @ FAR 1 % only 2.8 % on raw score vs 57.6 % after z-scoring against the enrolled gallery), so a reject
   threshold must be applied to the z-score. Classical Dice scores work raw.
5. **Tuning the classical baseline closes part of the FVC gap.** Tuned tolerances raise classical Rank-1 on held-out FVC from
   17.1 to 21.4 % at full area (equal to the fine-tuned learned model) and from 15.2 to 20.0 % at 75 %. The learned model keeps its
   advantage at <= 50 % area (18.1 vs 14.1 % at 50 %, 14.2 vs 8.5 % at 25 %) and in EER (36.5 vs 40.1 % at 100 %). So the Stage 8 statement
   that the fine-tuned model beats classical "everywhere" was partly an artefact of an untuned baseline: it holds for partial prints only.

## Caveats
- One probe per finger per seed (900 queries); the CIs are over queries, not over training runs.
- The tuning grid's best point lies on its upper edge for xy_tol (26) and xy_bin (14); the surface is flat there (top 6 settings within
  0.003 AUC), but larger tolerances were not tried. Set-B graphs use scale factors estimated from set B itself, slightly different from set A's.
- Open-set uses closed-set top-1 plus a threshold; non-matching enrolled probes are not counted as correct.
- FVC databases have 50-print galleries (about 350 queries per database): differences of a point or two are noise.

## Not done (deliberately)
- **Training-seed replication** of the learned model (about 100 min of CPU per seed). The model will be retrained after the extractor
  change, so seeds should be replicated on that final model.
- Extractor replacement, demo reject threshold, user-enrolled gallery (next steps).

## Reproduce
```bash
python src/eval_fixes.py --modes centred --out results/stage9c/sweep_centred.json
python src/eval_fixes.py --modes random --levels 0.75 0.5 0.25 0.1 --out results/stage9c/sweep_random.json
for y in 2002 2004; do for db in 1 2 3 4; do python src/fvc_pipeline.py \
  --root data/FVC/74034_3_En_4_MOESM1_ESM/FVC$y/Dbs/Db${db}_b --out data/fvc_packed_b/fvc${y}_db${db}.pt --set all --workers 2; done; done
python src/tune_classical.py
for db in fvc2002_db{1..4} fvc2004_db{1..4}; do python src/eval_fvc.py --db $db --finger-range 51-100 --no-classical \
  --classical-tuned results/stage9c/classical_tuned.json --outdir results/stage9c/fvc_tuned; done
python src/plot_stage9c.py
```
