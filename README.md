# GNN Partial Fingerprint Matching

Match **partial or damaged fingerprints** against a gallery of full prints with a graph neural network over minutiae graphs,
and compare it with a classical Hough-alignment minutiae matcher. Research question: how does each degrade as the query print gets smaller?

**New to the project? Start with [`docs/`](docs/README.md)**: [project status](docs/PROJECT_STATUS.md) (what is done, results, limits, what is left),
[getting started](docs/GETTING_STARTED.md) (install, data, demo, re-running, troubleshooting) and a [glossary](docs/GLOSSARY.md).

**Headline results** (details and caveats in the Status section and `results/stage*_notes.md`):

| | classical (Hough) | learned (GNN + MaxSim) |
|---|---|---|
| SOCOFing, 897-print gallery, full print / 25 % / 10 % of the print, Rank-1 | 90 / 31 / 6 % | 88 / 61 / 45 % |
| Real FVC prints (held-out fingers, 50-print galleries), full print / 10 %, Rank-1 | 79 / 30 % (tuned) | 88 / 44 % (fine-tuned on FVC) |

## Method
```
image --(LEADER neural minutiae detector, pyfing)--> minutiae (x, y, direction, type)
      --> k=5 nearest-neighbour graph (PyG)
      --> relative-geometry GNN (rotation/shift invariant) --> one descriptor per minutia
      --> score(query, template) = mean over query minutiae of their best cosine match in the template ("MaxSim")
      --> rank the gallery; z-score the top score to decide "enrolled or not"
```
Trained with an in-batch contrastive (InfoNCE) loss on random crops of SOCOFing prints, optionally fine-tuned on a few real FVC fingers.
Evaluation is by finger identity (train/val/test = 4200/900/900 fingers, `models/splits.json`), never on training fingers.

## Setup
```bash
pip install -r requirements.txt
# optional GPU (4-5x faster): pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```
Python 3.12, Windows 11 was used; numpy must stay below 2; Keras runs on the PyTorch backend (no TensorFlow).
Run every script from the repository root (each entry point adds `src/` to `sys.path` itself).

## Data
Not included (`data/` is gitignored); neither is redistributed here, check each dataset's terms.
- **SOCOFing** (Kaggle, `ruizgara/socofing`; `bash scripts/stage1.sh` downloads and organises it with a Kaggle API key).
- **FVC2002 / FVC2004** (100 fingers x 8 impressions per database): the supplement (Springer ESM) of the *Handbook of Fingerprint Recognition*,
  unzipped to `data/FVC/74034_3_En_4_MOESM1_ESM/`.
- **Minutiae detector**: `pyfing` 0.7.3 (MIT) ships the pretrained LEADER weights.
Tracked in the repo: `models/splits.json`, the final checkpoints (`results/stage9d_leader_*/best_model.pt`), every result json / table / figure.

## Reproducing
```bash
bash scripts/reproduce.sh check    # 1 min: environment, smoke test, documented numbers vs result files
bash scripts/reproduce.sh quick    # a few min: tiny end-to-end run of the final pipeline (30 fingers)
bash scripts/reproduce.sh graphs   # LEADER graphs for SOCOFing + FVC          (~20-60 min GPU)
bash scripts/reproduce.sh train    # final model + FVC fine-tuning              (~31 min GPU)
bash scripts/reproduce.sh eval     # all headline tables/figures                (~2 h, classical matcher dominates)
bash scripts/reproduce.sh seeds    # training-seed replication (Stage 10)       (~2 h GPU + 1 h CPU)
```
`python src/verify_results.py` checks the numbers quoted in these docs against the result files (`--tol 4` after a re-run,
since retraining moves results by seed noise, see `results/stage10_notes.md`). Seeds are fixed everywhere (training `--seed`, split seed 42,
evaluation probe/crop seeds); GPU training is not bit-for-bit deterministic, see Stage 10.

## Interactive demo
```bash
streamlit run app/app.py
```
Pick a gallery (SOCOFing test split or an FVC database), choose a sample probe or upload a fingerprint image, cut it down to a partial print
with the area/window sliders, and see the closest enrolled prints from the learned matcher next to the classical matcher, the minutiae
correspondences behind the score, and an "enrolled / uncertain / probably not enrolled" verdict. Needs `data/packed_leader/test.pt`,
`data/fvc_packed_leader/` (+ the FVC images for the FVC galleries) and the checkpoints in `results/stage9d_leader_*/`.

## Repository layout
```
src/   minutiae.py, graph_build.py, pack_graphs.py   original hand-built extractor and graph building (Stages 1-9c)
       leader_pipeline.py                            LEADER extraction -> graphs (final)       extractor_dev.py  extractor comparison
       model.py / model_v2.py                        Stage 3 GCN / final invariant MaxSim matcher
       train.py / train_v2.py                        Stage 3 triplet trainer / final InfoNCE trainer (cpu|cuda)
       evaluate*.py, eval_fvc.py, eval_fixes.py      evaluation (SOCOFing, FVC, 900-finger sweep + open-set)
       classical_match.py, tune_classical.py         Hough baseline and its tuning on FVC set B
       partiality_sweep.py, augment.py, dataset.py   partial-print crops, augmentation, sampling
       demo_core.py, calibrate_reject.py             demo back end and the enrolled/unknown thresholds
       verify_results.py, aggregate_seeds.py         result checks / seed aggregation (Stage 10)
app/app.py   Streamlit demo        scripts/   stage1.sh, reproduce.sh, seed_replication.sh
configs/default.yaml   hyperparameters (Stage 3 trainer)    results/   per-stage notes (stageN_notes.md), metrics, figures, checkpoints
data/, models/*.pt   gitignored (datasets, graphs, checkpoints that results do not depend on)
```

## Legacy (Stages 1-9c, original extractor)
Those stages used the hand-built extractor: `bash scripts/stage1.sh` (data + graphs + splits), `python src/pack_graphs.py`,
`python src/train.py --config configs/default.yaml`, `python src/evaluate.py --checkpoint <ckpt> --outdir results/<dir>`,
`python src/partiality_sweep.py`, `python src/classical_match.py --limit-fingers 200`. Their result files are kept and verified by
`scripts/reproduce.sh check`; re-running them additionally needs `data/graphs`, `data/packed`, `data/fvc_packed`.

## Status
10 of 11 stages are done; the final report and presentation (Stage 11) remain. A condensed version of this section with a limits list and a to-do list is in
[`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md). Stage entries below are chronological in the order the work was done; stages 1-9c quote numbers from the original
hand-built extractor, which Stage 9d superseded.

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
- **Stage 4 (embedding diagnostics) — done.** `src/diagnostics.py` +
  `notebooks/01_diagnostics.ipynb`. Same-finger embeddings cluster (t-SNE) and genuine
  scores are strong (0.89→0.83 Easy→Hard), but gallery templates have a heavy-tailed
  inter-identity similarity (p90 0.885) and Rank-1 losses are mostly near-misses
  (35.7% within 0.02 of flipping) — a separation/margin problem, fixable with
  hard-negative mining. See `results/stage4_notes.md`.
- **Stage 5 (classical baseline) — done.** `src/classical_match.py`: generalized
  Hough alignment on the same minutiae graphs, rotation+translation invariant.
  Head-to-head on 200 test fingers: **Rank-1 89.3 % vs GNN's 10.8 %**, EER 5.9 % vs
  18.0 %. Classical wins decisively — isolates what the GNN lacks (rotation
  invariance, precise local correspondence vs. global pooling). See
  `results/stage5_notes.md`.
- **Stage 6 (partiality sweep) — done.** Probes cropped to 100/75/50/25/10 % of the print,
  same crops for all methods, 200 test fingers. Classical Rank-1: 89.3 → 71.1 → 46.2 →
  21.0 → 5.2 %. Trained GCN: 10.8 → 4.3 → 2.3 → 1.7 → 1.3 % (chance 0.5 %), no better
  than the untrained net below 50 % area. The "GCN degrades more gracefully" hypothesis is
  not supported by the current model, but the GCN was never trained on crops, so this is a
  statement about the Stage 3 model, not about GCNs in general. See `results/stage6_notes.md`.
- **Stage 6b (augmented retraining) — done.** Retrained the GCN with random-crop (and
  crop+rotation) augmentation and re-ran the sweep. Verification AUC at partial areas rises
  by 0.05-0.06 (0.626 -> 0.685 at 50 %), but Rank-1 stays at 1-4 % (classical: 71 % at 75 %
  area). So the collapse is **not** mainly a training-distribution problem; the global
  embedding (absolute coordinates + mean/max pooling) is the limit. See
  `results/stage6b_notes.md`.
- **Stage 9 (model improvements) — done, main result of the project so far.** Ablation of
  loss / invariant features / readout. A correspondence-style matcher (relative-geometry GNN,
  per-minutia descriptors, MaxSim scoring, in-batch InfoNCE) reaches Rank-1 78.1 / 59.7 / 50.4 /
  38.0 / 22.3 % at 100/75/50/25/10 % of the print, vs classical 89.3 / 71.1 / 46.2 / 21.0 / 5.2 %
  (200 fingers). Classical still wins on near-complete prints; the learned model overtakes it
  at <= 50 % area and degrades far more gracefully, so the central hypothesis is now supported.
  Full 900-finger protocol: Rank-1 73.1 % (Stage 3: 3.8 %). Caveats: SOCOFing probes are edited
  copies of the gallery impression (cross-dataset FVC test is the key open risk), single seeds,
  model not fully converged. See `results/stage9_notes.md`.
- **Stage 8 (cross-dataset robustness, real FVC prints) — done; sobering.** SOCOFing probes are
  edited copies of the gallery print, so everything collapses on real independent impressions:
  the Stage 9 matcher goes from 78 % to 13 % Rank-1, classical from 89 % to 17 % (held-out FVC
  fingers, chance 2 %), i.e. the bottom of the FVC2002 field. Cause is mainly the minutiae
  pipeline (extraction repeatability, AUC 0.66; a ridge-scale bug was found and fixed), not the
  matcher. Fine-tuning the learned matcher on a little real FVC data gives the best result
  everywhere (21.4 vs classical 17.1 % at 100 % area, 14.2 vs 8.4 % at 25 %, better on all 8
  databases); synthetic noise augmentation hurt. See `results/stage8_notes.md`.
- **Stage 7 (interactive demo) — done.** `streamlit run app/app.py`: choose a gallery (SOCOFing test or
  an FVC database), pick a sample probe or upload an image, cut it to a partial print, and see the closest
  enrolled prints from the learned matcher beside the classical matcher, with minutiae overlay,
  correspondence lines and score histogram. Demo-protocol Rank-1 (random window, whole gallery): SOCOFing
  900-print gallery 71.7 / 15.7 / 10.7 % at 100/50/25 % area (classical 81.0 / 15.3 / 4.0); real FVC
  (50-print galleries) 22.7 / 21.3 / 11.3 % (classical 18.0 / 14.3 / 7.0). Far below the Stage 9 sweep at
  partial areas (bigger gallery, sparser windows; not separated). See `results/stage7_notes.md`.
- **Stage 9c (evaluation fixes) — done.** Full 900-finger gallery with bootstrap CIs and 3 probe/crop seeds, an open-set
  test, and a classical baseline tuned on FVC set B. The Stage 7 gap was mostly window placement: at 50 % area the same model
  gets 42.7 % with minutia-centred windows vs 14.9 % with uniform random ones (900-print gallery). Learned vs classical:
  classical wins at 100/75 % (83/64 vs 72/53 %), tie at 50 %, learned clearly ahead at 25/10 % (33.0 vs 13.2, 18.3 vs 1.4 %).
  Open-set rejection works but the learned score needs z-normalisation. A tuned classical matcher equals the FVC-fine-tuned
  learned model at full area (21.4 %) so the learned advantage on real prints is limited to partial areas. Training-seed
  variance is not yet measured. See `results/stage9c_notes.md`.
- **Stage 9d (robust extractor, GPU) — done; biggest jump of the project.** Replaced the hand-built extractor with the pretrained
  LEADER detector (`pyfing`; FVC repeatability AUC 0.66 -> 0.90), rebuilt all graphs and retrained on the GPU (torch cu121, 14 s/epoch).
  Held-out real FVC prints, Rank-1 at 100 % area: classical 17.1 -> 78.8 % (tuned), learned 21.4 -> 76.3 % (SOCOFing-only) and
  **88.2 %** after FVC fine-tuning (verification EER 36.5 -> 7.4 %); learned still clearly ahead at small areas (10 %: 44.0 vs 30.4 %).
  SOCOFing (897-finger gallery): learned 88.4 / 85.1 / 75.1 / 60.8 / 44.8 % vs classical 90.3 / 87.2 / 69.1 / 30.9 / 6.1 % at 100/75/50/25/10 %.
  Caveat: LEADER was trained on three of the FVC databases used here (split reported; gains hold on the unseen ones).
  See `results/stage9d_notes.md`.
- **Stage 9e (demo on the best pipeline) — done.** The Streamlit demo now uses LEADER minutiae and the Stage 9d models, compares against the
  FVC-tuned classical matcher, and shows an open-set verdict ("likely enrolled / uncertain / probably not enrolled") from calibrated z-score thresholds.
  Demo-protocol Rank-1 (random window): FVC 87 / 84 / 73 % at 100/50/25 % area (classical 75 / 75 / 68); SOCOFing 89 / 39 / 24 % (classical 90 / 38 / 13).
  See `results/stage9e_notes.md`.
- **Stage 10 (reproducibility) — done.** Pinned `requirements.txt`, `scripts/reproduce.sh` (`check` / `quick` / `graphs` / `train` / `eval` / `seeds`),
  `src/verify_results.py` (15 documented numbers checked against the result files: all match), rewritten README. Training-seed replication of the
  final pipeline (3 independent runs + a same-seed rerun): seed sd about 1-2 points (FVC fine-tuned Rank-1 100 %: 87.3 ± 0.9, SOCOFing 25 %: 59.8 ± 1.1;
  the original 88.2 / 60.8 runs sit about 1 point above the mean); same seed on GPU is not bit-identical (up to 1.7 points). Conclusions are not seed-dependent.
  The wrapper's long modes were not run end to end from empty data. See `results/stage10_notes.md`.
- **Next:** Stage 11 (final report and presentation).
