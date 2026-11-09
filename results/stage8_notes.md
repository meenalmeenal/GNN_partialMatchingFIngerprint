# Stage 8 — cross-dataset robustness (SOCOFing -> real FVC fingerprints)

Question: do the Stage 9 results survive on real, independent impressions? SOCOFing probes are
edited *copies* of the gallery print, so unaltered regions match exactly; FVC genuine pairs are
different captures of a finger (pressure, displacement, elastic distortion, noise).

## Data and protocol
- FVC2002 DB1-4 and FVC2004 DB1-4, set A (100 fingers x 8 impressions each; DB1-3 of each are
  real sensors, DB4 are synthetic SFinGe). Source: the FVC supplement to the *Handbook of
  Fingerprint Recognition* (Springer ESM, 830 MB), downloaded with the user's permission. The raw
  images are **not** redistributed here (`data/` is gitignored).
- Gallery = impression 1 of each finger; identification queries = impressions 2-8; verification
  (FVC style) = all ordered pairs of different impressions of a finger (genuine) vs impression-1
  queries against other fingers' templates (impostor). Partial prints: the query is cropped to a
  fraction of its own extent; templates stay whole; identical crops for every method.
  Verification is reported impostor-cohort normalised (raw values are in the json files).
- Models are trained on SOCOFing only ("zero-shot") unless stated. **Fine-tuning** uses FVC fingers
  1-40 (validation 41-50) pooled over all 8 databases; **fingers 51-100 are held out** from
  training and model selection, and every method is scored on those same fingers.
- Reference point (FVC2002 results report, shipped in the package): the 31 participating systems
  reach DB1 EER from 0.10 % (best) to 50 %; the bottom of the field is PA16 16.3 %, PA22 17.3 %,
  PA25 35 %. A competent system is far below 5 %.

## What had to be fixed before any model could be compared (`src/fvc_pipeline.py`)
1. **Ridge-spacing normalisation.** The SOCOFing extractor resizes everything to 160x160, which
   assumes SOCOFing's ridge spacing. FVC images are rescaled so their ridge period matches the
   SOCOFing target. The obvious spectral period estimator is wrong on SOCOFing (it locks onto a
   harmonic: 4.8 px vs the true 8.2 px, measured as mask area / skeleton length) but agrees with the
   skeleton measure on all eight FVC sensors, so the target (8.22 px) is calibrated on SOCOFing
   with the skeleton measure.
2. **One scale per sensor, not per image.** Per-image period estimates vary by 11-25 % within one
   database, which put two impressions of the same finger at different scales. A fixed per-database
   scale raised extraction repeatability from AUC 0.56 to 0.66 (FVC2002 DB1 0.60 -> 0.75, DB2
   0.63 -> 0.81). This was a real bug in my first cross-dataset pipeline, not a model property.
3. Extractor repeatability was measured on FVC **set B** only (fingers 101-110, provided by FVC
   for tuning), with orientation-free rigid registration. Its harness recovers a rotated, shifted
   copy at AUC 0.994 with 30 % of points dropped, 20 % spurious and 2 px jitter (0.70 at 50 %/50 %/4 px),
   while real genuine pairs score 0.56-0.66, i.e. **our extracted minutiae are less repeatable
   across impressions than that harsh synthetic case**. Nothing tried improved on the baseline
   extractor's parameters: border 12/20, dedup 10, spur 12/16, blob cleaning 20/40, coherence
   filter 0.3/0.5 and their combinations (AUC 0.52-0.57 vs 0.56 baseline, per-image scale), and
   multi-wavelength Gabor made it worse (0.50-0.59 vs 0.66, per-database scale). The extractor is
   therefore kept identical to the SOCOFing one so that the cross-dataset test uses one extractor.

## Results

**Zero-shot, all 100 fingers, full-print queries (chance Rank-1 = 1 %); Rank-1 % / EER %**

| database | classical | Invariant GNN+MaxSim (Stage 9) | pooled GCN + InfoNCE | Stage 3 GCN |
|---|---|---|---|---|
| FVC2002 DB1 | 23.7 / 40.1 | 16.9 / 36.1 | 3.4 / 45.9 | 1.7 / 48.5 |
| FVC2002 DB2 | 31.1 / 39.9 | 20.4 / 36.8 | 3.0 / 46.6 | 1.4 / 45.0 |
| FVC2002 DB3 | 15.4 / 43.6 | 6.9 / 39.4 | 3.4 / 42.1 | 2.0 / 46.8 |
| FVC2002 DB4 (synthetic) | 18.6 / 39.0 | 12.0 / 31.3 | 3.0 / 45.4 | 1.3 / 48.0 |
| FVC2004 DB1 | 3.7 / 48.7 | 3.0 / 47.7 | 1.9 / 44.4 | 1.4 / 41.4 |
| FVC2004 DB2 | 8.1 / 44.1 | 4.0 / 42.2 | 1.6 / 46.2 | 2.4 / 47.4 |
| FVC2004 DB3 | 23.7 / 37.7 | 10.3 / 33.4 | 5.1 / 38.4 | 1.6 / 47.7 |
| FVC2004 DB4 (synthetic) | 12.6 / 42.4 | 9.1 / 37.4 | 2.3 / 46.1 | 1.4 / 46.3 |

**Held-out fingers 51-100, mean over the 6 real-sensor databases (chance Rank-1 = 2 %); Rank-1 %**

| method | 100 % | 75 % | 50 % | 25 % | 10 % |
|---|---|---|---|---|---|
| Classical Hough (no training) | 17.1 | 15.2 | 13.0 | 8.4 | 4.9 |
| **Invariant GNN + MaxSim, fine-tuned on real FVC** | **21.4** | **19.6** | **18.1** | **14.2** | **7.5** |
| Invariant GNN + MaxSim, SOCOFing only | 13.0 | 12.3 | 11.0 | 9.6 | 6.2 |
| + synthetic impression noise (SOCOFing only) | 10.4 | 9.5 | 8.7 | 6.4 | 4.7 |
| pooled GCN, stronger loss (SOCOFing only) | 5.0 | 5.1 | 3.6 | 3.2 | 2.2 |
| Stage 3 GCN (SOCOFing only) | 3.0 | 3.4 | 2.3 | 2.1 | 2.0 |

Verification EER % (cohort-normalised), same setting, 100 % / 25 % area: classical 43.4 / 46.6;
fine-tuned 36.5 / 40.3; SOCOFing-only 40.5 / 42.4. Full per-database tables:
`results/stage8/summary_tables.md`; all raw numbers: `results/stage8/heldout/*.json`;
figure: `results/stage8/headline.png`.

## Reading it
1. **Everything collapses on real impressions.** The Stage 9 model went from Rank-1 78.1 % (SOCOFing, 200
   fingers) to 13.0 % on real FVC prints; the classical matcher from 89.3 % to 17.1 %. In absolute terms
   all of our systems sit at the very bottom of the FVC field (EER 36-47 % vs 0.1-5 % for competent
   systems). The cross-dataset gap is large and shared, and the dominant cause is the minutiae pipeline
   (extraction repeatability across independent impressions) rather than the matcher: even the classical
   matcher, which has no training to over-fit to SOCOFing, falls the same way.
2. **The Stage 9 ranking does not transfer zero-shot.** On SOCOFing the learned matcher beat classical at
   <= 50 % area; on real FVC the SOCOFing-only model is *below* classical on full prints (13.0 vs 17.1 %)
   and roughly on par at small areas (9.6 vs 8.4 % at 25 %, 6.2 vs 4.9 % at 10 %; differences of this size
   are within noise). It is still far better than the older pooled GCNs, which are at or near chance on real
   data (Stage 3: 3.0 %).
3. **A small amount of real data fixes most of the *relative* gap.** Fine-tuning on 320 FVC fingers (40 per
   database) gives the best result at every area: 21.4 vs 17.1 % (classical) at 100 %, 14.2 vs 8.4 % at 25 %,
   and it beats the classical matcher on **all 8 databases** at both 100 % and 25 % area. Caveat: the
   held-out fingers come from the same sensors and sessions as the training fingers, so this is in-domain
   adaptation, not cross-sensor generalisation.
4. **Synthetic impression noise did not help.** The recipe (warp up to 4 px, jitter up to 2 px, up to 30 %
   dropped/30 % spurious minutiae, 10 % type flips, applied to query and gallery) was too strong: the model
   barely learned (in-batch accuracy 9 % vs 45-52 % for the clean model) and transfer got worse
   (10.4 vs 13.0 % Rank-1). A weaker or curriculum schedule is untested, not ruled out.
5. Treating orientation as an undirected line in the classical matcher (`hough_score_line`) did not help
   (20.9 vs 23.7 % on DB1), so the arbitrary orientation sign is not the bottleneck.

## Caveats
- The absolute numbers are poor and **must not be quoted as the performance of a fingerprint matcher**;
  they measure this particular simple extractor + these matchers on real prints.
- One training run per model, no tuning, 50 held-out fingers per database (about 350 queries; Rank-1
  standard error roughly 2 points per database, about 1 point on the 6-database mean). The
  fine-tuned vs classical gap (+4 to +6 points) is consistent across databases but each single
  difference is small.
- Classical-matcher parameters were not tuned for FVC (tolerances fixed from SOCOFing); the learned
  models were fine-tuned. The comparison is "classical as published here" vs "learned, adapted".
- DB4 of each year is synthetic and FVC2004 is deliberately perturbed, so per-database numbers vary a lot.
- The partial-print sweep uses square windows relative to the print's own extent; real latent prints
  are irregular.

## Next
- The single highest-value improvement is a **robust minutiae extractor** (NIST MINDTCT, FingerNet or
  another learned detector). Our extractor's repeatability across impressions caps every method; none
  of the cheap parameter changes tried here moved it. Re-extracting SOCOFing with the same extractor and
  retraining would then be required for consistency.
- Then re-run `eval_fvc.py` unchanged; a milder noise schedule and from-scratch FVC training are cheap controls.
- Stage 7 (interface) can use `models/stage9_inv_local_cont.pt` or the fine-tuned
  `results/stage8_inv_local_fvcft/best_model.pt`, with the above limits stated in the UI.

## Reproduce
```bash
# data: FVC package unzipped under data/FVC/74034_3_En_4_MOESM1_ESM/
for y in 2002 2004; do for db in 1 2 3 4; do
  python src/fvc_pipeline.py --root data/FVC/74034_3_En_4_MOESM1_ESM/FVC$y/Dbs/Db${db}_a \
      --out data/fvc_packed/fvc${y}_db${db}.pt --set A; done; done
python src/eval_fvc.py --db fvc2002_db1 --outdir results/stage8/zero_shot_full --levels 1.0 \
    --old stage3=results/stage3/best_model.pt --models-v2 inv_local=models/stage9_inv_local_cont.pt
python src/train_v2.py --arch inv_local --init models/stage9_inv_local_cont.pt --noise --tag stage8_inv_local_noise
python src/train_v2.py --arch inv_local --init models/stage9_inv_local_cont.pt --lr 3e-4 --epochs 40 \
    --iters-per-epoch 200 --tag stage8_inv_local_fvcft \
    --fvc-dbs fvc2002_db1 fvc2002_db2 fvc2002_db3 fvc2002_db4 fvc2004_db1 fvc2004_db2 fvc2004_db3 fvc2004_db4
python src/eval_fvc.py --db <db> --finger-range 51-100 --outdir results/stage8/heldout \
    --old stage3=results/stage3/best_model.pt --models-v2 v1_global=models/stage9_v1_global.pt \
    inv_local=models/stage9_inv_local_cont.pt inv_local_noise=models/stage8_inv_local_noise.pt \
    inv_local_fvcft=models/stage8_inv_local_fvcft.pt
python src/plot_stage8.py
```
