# Stage 6b — does crop / rotation augmentation fix the partial-print collapse?

Follow-up to Stage 6. The Stage 3 GCN was only ever trained on full-size prints, so the
obvious objection to the Stage 6 result was "you never showed it a partial print". This
stage retrains the same GCN with partial-print augmentation and re-runs the identical sweep.

## What changed (and what did not)
- `src/augment.py` (new): `GraphAugmenter`. The anchor (query) is always augmented; the
  positive and negative are augmented with p = 0.3 so "small graph" is not a shortcut.
  - **crop**: contiguous square window covering U(10 %, 100 %) of the print (15 % of the
    time no crop), same window logic as the evaluation sweep, graph rebuilt from the
    surviving minutiae.
  - **crop+rot**: the above plus a random global rotation of +/-30 degrees about the image centre
    (orientation features rotated with it).
  - A vectorised k-NN graph builder (0.19 ms vs ~4.7 ms) keeps augmentation affordable. It
    matches `graph_build.minutiae_to_graph` (feature error 3e-7); the only differences, in 10
    of 400 checked graphs, are equidistant-neighbour ties broken differently.
- `train.py`: `--augment {none,crop,crop_rot}`, `--max-rot-deg`, `--ckpt-name` (so the
  Stage 3 checkpoint is not overwritten). Validation uses the same augmentation (fixed
  seed) so checkpoint selection reflects partial-print robustness.
- Everything else is Stage 3's recipe: 3-layer GCN, triplet loss with random negatives,
  Adam 1e-3, 400 iters/epoch, early stopping patience 8. Two runs, one seed each:
  `aug_crop` (ran the full 50 epochs, best val 0.627) and `aug_crop_rot` (early-stopped
  at epoch 33, best val 0.670). Validation losses are not comparable to Stage 3's 0.30
  because the task is much harder now.
- `partiality_sweep.py`: now scores several checkpoints (`--models name=path`) and can reuse
  the classical results (`--reuse-classical`, valid because crops are deterministic for a
  seed; asserted). Also fixed: the untrained floor is now seeded *before* it is built, so it
  is reproducible (17.7 % at 100 % area vs 18.2 % in Stage 6, which was unseeded).
  The Stage 3 row reproduces Stage 6 exactly.

## Results (200 test fingers, 1,670 probes, identical crops; chance Rank-1 = 0.5 %)

Rank-1 identification (%):

| area | classical | GCN Stage 3 | GCN aug_crop | GCN aug_crop_rot | untrained |
|---|---|---|---|---|---|
| 100 % | **89.3** | 10.8 | 12.0 | 6.9 | 17.7 |
| 75 %  | **71.1** | 4.3 | 4.1 | 3.6 | 7.1 |
| 50 %  | **46.2** | 2.3 | 2.6 | 2.0 | 4.8 |
| 25 %  | **21.0** | 1.7 | 1.6 | 1.4 | 2.6 |
| 10 %  | **5.2**  | 1.3 | 1.0 | 1.4 | 1.8 |

Verification ROC-AUC (EER %):

| area | classical | GCN Stage 3 | GCN aug_crop | GCN aug_crop_rot | untrained |
|---|---|---|---|---|---|
| 100 % | 0.971 (5.9) | 0.899 (18.0) | 0.898 (17.9) | 0.892 (19.2) | 0.804 (27.8) |
| 75 %  | 0.919 (14.0) | 0.720 (33.2) | **0.781 (29.1)** | 0.762 (30.3) | 0.700 (35.1) |
| 50 %  | 0.843 (23.4) | 0.626 (40.2) | **0.685 (36.0)** | 0.664 (37.2) | 0.635 (40.2) |
| 25 %  | 0.736 (33.1) | 0.576 (43.9) | **0.623 (40.1)** | 0.602 (41.7) | 0.586 (43.9) |
| 10 %  | 0.645 (39.9) | 0.574 (44.1) | **0.625 (40.5)** | 0.610 (41.2) | 0.560 (45.8) |

Figures: `results/stage6b/sweep_rank1.png`, `sweep_eer.png`; training curves in
`results/stage6b_aug_crop*/`.

## Reading it

1. **Augmentation helps verification a little and identification not at all.** Crop
   augmentation lifts partial-print AUC by 0.05-0.06 (e.g. 0.626 -> 0.685 at 50 % area) and
   keeps full-print AUC unchanged (0.898), so the retraining did learn something real. But
   Rank-1 stays at 1-4 % from 75 % area down, within about +/-1 point of the Stage 3 model
   (that is sampling noise at this sample size). Rank-5 is similar (e.g. 17.1 % vs 13.1 %
   at 75 %, 4.9 % vs 5.7 % at 10 %).
2. **The classical matcher is still far ahead everywhere**: 71 % vs 4 % Rank-1 and AUC 0.919
   vs 0.781 at 75 % area.
3. **Rotation augmentation adds nothing here and slightly hurts.** The sweep's probes are not
   globally rotated, and the model's features are absolute coordinates, so it cannot turn
   rotation exposure into invariance; it only dilutes capacity. (This does not test rotated
   queries, which the SOCOFing `CR` alteration only partly covers.)

**Conclusion for the thesis question.** "The GCN collapsed because it never saw partial
prints" is now ruled out as the *main* explanation. Training on partial prints, with the
architecture unchanged, closes only a small part of the verification gap and none of the
identification gap. That points at the representation, as the Stage 4/5 diagnosis
suggested: absolute (x, y) features plus global mean/max pooling squeeze a whole graph into
one vector, so a small sub-region and its full print land far apart no matter how it is
trained. The next experiments should change the model, not the data: relative / rotation-
invariant edge features (the edge_attr we already compute but do not use), and a
correspondence-style matching head instead of a single global embedding (Stage 9). Hard-
negative mining remains untried and is a separate lever.

## Caveats
- One seed per configuration, no hyper-parameter tuning, 200-finger subset. Rank-1
  differences of about 1 point are noise; the AUC gains (0.05-0.06 on 1,670 genuine and
  33.4k impostor pairs) are well above that noise level but still single-run.
- `aug_crop` hit the 50-epoch budget with validation loss still creeping down (0.631 ->
  0.627 over the last 20 epochs); more epochs would give small gains at best given that
  plateau.
- Augmentation strength was a first guess (crop range, p_other = 0.3, +/-30 degrees).
- Anchor-only cropping with full-print positives mirrors the test scenario (partial query
  vs full gallery); other pairings were not tried.

## Reproduce
```bash
python src/train.py --tag stage6b_aug_crop --augment crop --ckpt-name best_model_aug_crop
python src/train.py --tag stage6b_aug_crop_rot --augment crop_rot --max-rot-deg 30 --ckpt-name best_model_aug_crop_rot
python src/partiality_sweep.py --limit-fingers 200 --outdir results/stage6b \
    --reuse-classical results/stage6/sweep_metrics.json \
    --models aug_crop=models/best_model_aug_crop.pt aug_crop_rot=models/best_model_aug_crop_rot.pt
```
