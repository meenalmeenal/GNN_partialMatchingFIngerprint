# Stage 9d — robust minutiae extractor (pyfing LEADER) and retraining

Stage 8 concluded that the extractor, not the matcher, limits every method on real prints (repeatability AUC 0.66).
This stage replaces the hand-built extractor with a pretrained neural detector and re-runs the pipeline.

## What was done
1. **Choice of extractor.** `pyfing` 0.7.3 (MIT, R. Cappelli) with LEADER, a 0.9 M-parameter network that maps an image to
   minutiae (position, direction, type, quality). Installed with Keras 3 on the existing PyTorch backend (numpy pinned to 1.26.4).
   Also replaced the CPU-only torch with torch 2.4.1+cu121 to use the RTX 3050 (4 GB): LEADER runs at about 10 ms/image,
   and GNN training at 14 s/epoch (65 s on CPU). Smoke test still passes.
2. **Repeatability on FVC set B** (`src/extractor_dev.py`, same Stage 8 harness: fingers 101-110, registration AUC of
   genuine vs impostor pairs, 5 databases, one fixed scale per database). Ours 0.661; LEADER 0.885 at its default quality
   threshold 0.6, 0.904 at 0.3 (more minutiae, better AUC). The result is insensitive to the dpi setting (0.885-0.895).
   Quality threshold 0.3 was chosen, plus "keep the best 6 candidates" for damaged prints. Log: `results/stage9d/extractor_dev.txt`.
3. **Re-extraction** (`src/leader_pipeline.py`): all 55 k SOCOFing prints (160x160 as before) and all FVC set A and B images
   (per-database scale as in Stage 8), same graph format, so models/matchers/evaluation are unchanged. Medians: SOCOFing 17
   minutiae/print (old extractor: 32), FVC 27-49 (in the SOCOFing example inspected, our extractor put many minutiae on the image border; LEADER's sit on ridges).
   Failures: 8 + 2 + 1 of 6000 SOCOFing fingers have no usable gallery print and drop out (897 of 900 test fingers remain).
4. **Retraining** (`train_v2.py`, now with `--device cuda`, `--fvc-dir`): the Stage 9 `inv_local` matcher from scratch on the
   LEADER SOCOFing graphs (90 epochs, 21 min), then fine-tuned on FVC fingers 1-40 as in Stage 8 (held-out fingers 51-100).
5. **Classical baseline re-tuned** on LEADER set-B graphs (`tune_classical.py`: xy_tol 20, ang_tol 0.35, xy_bin 14, ang_bin 0.4).

## Results
**FVC held-out fingers 51-100, mean of the 6 real-sensor databases, Rank-1 % (chance 2 %)**

| method | 100 % | 75 % | 50 % | 25 % | 10 % |
|---|---|---|---|---|---|
| Classical, default tolerances, old extractor (Stage 8) | 17.1 | 15.2 | 13.0 | 8.4 | 4.9 |
| Classical, tuned, old extractor (Stage 9c) | 21.4 | 20.0 | 14.1 | 8.5 | 4.2 |
| Learned, fine-tuned on FVC, old extractor (Stage 8) | 21.4 | 19.6 | 18.1 | 14.2 | 7.5 |
| Classical, default tolerances, **LEADER** | 74.7 | 73.0 | 69.9 | 57.1 | 32.4 |
| Classical, tuned, **LEADER** | 78.8 | 77.6 | 74.4 | 58.9 | 30.4 |
| Old learned model (trained on old-extractor SOCOFing) on LEADER graphs | 54.4 | 54.2 | 49.0 | 41.1 | 24.3 |
| Learned, retrained on LEADER SOCOFing graphs (no FVC data) | 76.3 | 75.3 | 72.0 | 61.1 | 41.1 |
| **Learned, retrained + fine-tuned on FVC** | **88.2** | **87.1** | **82.7** | **70.1** | **44.0** |

Verification EER % (cohort-normalised) at 100 / 25 % area: old best 36.5 / 40.3 -> classical tuned LEADER 16.2 / 24.0,
learned LEADER SOCOFing-only 15.9 / 21.1, learned fine-tuned **7.4 / 15.0**. Per-database and per-area tables: `results/stage9d/summary_tables.md`;
figure `results/stage9d/headline.png`.

**SOCOFing, full test gallery, minutia-centred windows (95 % CI)**

| extractor | method | 100 % | 75 % | 50 % | 25 % | 10 % |
|---|---|---|---|---|---|---|
| old (900 fingers) | learned | 72.0 [69-75] | 53.0 [50-56] | 42.7 [40-46] | 33.0 [30-36] | 18.3 [16-21] |
| old | classical | 83.3 [81-86] | 64.3 [61-67] | 37.8 [35-41] | 13.2 [11-15] | 1.4 [1-2] |
| LEADER (897 fingers) | learned | 88.4 [86-90] | 85.1 [83-87] | 75.1 [72-78] | 60.8 [58-64] | 44.8 [42-48] |
| LEADER | classical | 90.3 [88-92] | 87.2 [85-89] | 69.1 [66-72] | 30.9 [28-34] | 6.1 [5-8] |

Open-set DIR at FAR 10 % (z-scored score), LEADER: learned 88 / 83 / 74 / 57 / 38 %, classical 89 / 85 / 60 / 17 / 2 % at 100 / 75 / 50 / 25 / 10 %
(old extractor, learned: 66 / 50 / 39 / 26 / 15 %).

## Reading it
1. **The extractor was the bottleneck.** Without retraining anything, swapping in LEADER lifted the untouched Stage 8
   models from 17.1 to 74.7 % (classical) and the old learned model from 13.0 to 54.4 % on held-out real FVC prints. Stage 8's diagnosis
   (extraction repeatability, not the matcher) is confirmed quantitatively: the same matchers go from the bottom of the FVC field to
   respectable numbers on the same images.
2. **Retraining on the new graphs helps further** (54.4 -> 76.3 % zero-shot, no FVC data), and FVC fine-tuning adds the rest
   (88.2 %; verification EER 7.4 %). The matcher mismatch matters: the old model was learned on the other extractor's noise.
3. **Learned vs classical now**: about equal on full prints (SOCOFing 88.4 vs 90.3 %, CIs overlap; FVC 76.3 SOCOFing-only vs 78.8 tuned
   classical), and the learned model is clearly ahead as the print shrinks (SOCOFing 25 %: 60.8 vs 30.9 %, 10 %: 44.8 vs 6.1 %; FVC 10 %: 41.1 vs 30.4 %).
   With FVC fine-tuning it leads everywhere. The central hypothesis (the learned matcher degrades more gracefully) holds on both datasets and
   with the better extractor.
4. **Open-set** is now usable: at 10 % false-accept rate 88 % of enrolled full prints are identified and accepted.
5. Fewer but cleaner minutiae (17 vs 32 per SOCOFing print) beat more but noisier ones.

## Caveat that matters: LEADER saw part of FVC
LEADER's paper (arXiv 2602.15493) lists its training data as FVC2002 DB3-A, FVC2004 DB1-A and DB3-A (800 images each), 100 images of FVC2002 DB2-A
and a few FFE images; FVC2002 DB1-A and NIST SD27/SD302 are test-only. So on three of our six real databases the extractor has seen the images
(with minutiae labels) in training, which can only flatter it. Split by this (Rank-1 at 100 / 50 / 25 / 10 %):

| method | unseen by LEADER (2002 DB1, 2004 DB2, both synthetic DB4s) | trained-on by LEADER (2002 DB3, 2004 DB1, 2004 DB3) |
|---|---|---|
| classical, tuned | 64.6 / 60.3 / 44.8 / 22.4 | 78.7 / 72.4 / 54.2 / 25.8 |
| learned, LEADER SOCOFing-only | 62.6 / 58.1 / 47.1 / 30.1 | 73.7 / 68.6 / 56.6 / 34.2 |
| learned, fine-tuned | 83.6 / 76.6 / 61.6 / 37.4 | 84.6 / 78.4 / 64.8 / 36.9 |

The jump (about 15-20 % to 60-84 % on unseen databases) is clearly not an artefact of contamination, but the absolute numbers on the three seen
databases (and the 6-database mean) are optimistic, and the unseen group contains the two synthetic databases (harder for the old extractor too). The fine-tuned
model gains little between groups (83.6 vs 84.6), the zero-shot ones gain 11-14 points. Set B (used for tuning the thresholds and the classical tolerances) is a different set of fingers from set A on the same sensors. A clean claim should lean on the unseen group.

## Other caveats
- The fine-tuned model is still in-domain: held-out fingers of the same sensors and sessions as the 40 training fingers per database.
- One training run per model (training-seed variance not yet measured; now cheap on GPU, about 20 min per run).
- SOCOFing: 3 of 900 test fingers lack a usable gallery graph and drop out; old and new rows differ by those 3.
- SOCOFing probes are edited copies of the gallery print, so extraction of the shared pixels is identical; the SOCOFing gains are not an
  independent-impression measurement (FVC is).
- Per database sizes: 50 fingers, about 350 queries (standard error about 2 points per database, about 1 on the 6-database mean).

## Not done / next
- The demo (`app/app.py`) still uses the old extractor and the old models; switching it to LEADER (`pyfing` import, new checkpoints) is a small, valuable follow-up.
- Training-seed replication on the new model, a milder noise schedule, bigger model (cheap on GPU now), and the remaining Stage 10 / 11.

## Reproduce
```bash
pip install pyfing keras "numpy==1.26.4"         # Keras runs on the torch backend: KERAS_BACKEND=torch
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 --index-url https://download.pytorch.org/whl/cu121   # optional, GPU
python src/extractor_dev.py --variants ours leader_norm leader_raw leader_norm_dpi433
python src/extractor_dev.py --variants leader_low --thresholds 0.3 0.4 0.5 0.6 0.7 0.8
python src/leader_pipeline.py socofing          # -> data/packed_leader/*.pt
python src/leader_pipeline.py fvc               # -> data/fvc_packed_leader[_b]/*.pt
python src/train_v2.py --arch inv_local --tag stage9d_leader_inv_local --packed-dir data/packed_leader --device cuda --epochs 90 --patience 10
python src/train_v2.py --arch inv_local --init models/stage9d_leader_inv_local.pt --lr 3e-4 --epochs 40 --iters-per-epoch 200 \
    --tag stage9d_leader_fvcft --fvc-dir data/fvc_packed_leader --device cuda --fvc-dbs fvc2002_db1 fvc2002_db2 fvc2002_db3 fvc2002_db4 fvc2004_db1 fvc2004_db2 fvc2004_db3 fvc2004_db4
python src/tune_classical.py --dir data/fvc_packed_leader_b --out results/stage9d/classical_tuned_leader.json
python src/eval_fvc.py --db <db> --packed-dir data/fvc_packed_leader --finger-range 51-100 --outdir results/stage9d/fvc_oldmodel --models-v2 inv_local_oldmodel=models/stage9_inv_local_cont.pt
python src/eval_fvc.py --db <db> --packed-dir data/fvc_packed_leader --finger-range 51-100 --no-classical --classical-tuned results/stage9d/classical_tuned_leader.json --outdir results/stage9d/fvc_tuned
python src/eval_fvc.py --db <db> --packed-dir data/fvc_packed_leader --finger-range 51-100 --no-classical --models-v2 inv_local_leader=models/stage9d_leader_inv_local.pt --outdir results/stage9d/fvc_newmodel
python src/eval_fvc.py --db <db> --packed-dir data/fvc_packed_leader --finger-range 51-100 --no-classical --models-v2 inv_local_leader_fvcft=models/stage9d_leader_fvcft.pt --outdir results/stage9d/fvc_ft
python src/eval_fixes.py --modes centred --packed-dir data/packed_leader --checkpoint models/stage9d_leader_inv_local.pt --cache-tag _leader --out results/stage9d/sweep_centred_leader.json
python src/plot_stage9d.py
```
