# Stage 10 — reproducibility

Goal: someone with the data and this repository can rebuild the final results, check that the documented numbers match the files,
and know how much a re-run is expected to move. Also closes the one open statistical gap: training-seed variance.

## What was added
- `requirements.txt`: the exact package versions used (Python 3.12, torch 2.4.1 CPU or cu121, numpy pinned below 2, Keras 3 on the torch backend, pyfing 0.7.3).
- `scripts/reproduce.sh` with modes `check`, `quick`, `graphs`, `train`, `eval`, `seeds`; each mode is the exact command sequence used in Stages 9d/9e.
- `src/verify_results.py`: 15 checks that read the result json files and compare them with the numbers quoted in the README/notes
  (tolerance 0.15 points on the committed artefacts; `--tol 4` after a re-run).
- `scripts/seed_replication.sh` + `src/aggregate_seeds.py`: training-seed replication of the final pipeline.
- README rewritten as an entry point (method, setup, data sources, reproduction table, repository layout, legacy commands); the stale
  demo and quick-start text from Stage 7 and earlier was removed or moved under "Legacy".

## What was verified in this stage
- `bash scripts/reproduce.sh check`: environment imports (numpy 1.26.4, Keras on the torch backend, pyfing), `smoke_test.py` passes,
  **15/15 documented numbers match the result files**.
- `bash scripts/reproduce.sh quick`: LEADER extraction of 30 fingers per split -> graphs -> 2 training epochs on the GPU -> evaluation, all executed
  without error in a few minutes. (The accuracy of this toy run is meaningless: 2 epochs.)
- `bash -n` syntax check of both shell scripts.
- **Not executed as one piece**: the `graphs`, `train` and `eval` modes (hours). Their commands are the ones that produced the committed results
  in Stages 9d/9e, copied verbatim, but the wrapper itself has not been run end to end from an empty `data/` folder, and everything was run on one machine.

## Training-seed replication (`results/stage10/seed_summary.md`, `results/stage10/seeds.png`)
The final pipeline (train from scratch on the LEADER SOCOFing graphs, fine-tune on FVC fingers 1-40, evaluate) was run for seeds 1 and 2, and
seed 42 was run again with the identical command as the original Stage 9d run. FVC numbers are the mean of the 6 real-sensor databases (held-out fingers 51-100),
SOCOFing uses the 897-finger gallery with minutia-centred windows (same probe draw for every run).

| metric (Rank-1 %) | seed 42 (original) | seed 1 | seed 2 | mean ± sd (n = 3) | seed 42 rerun |
|---|---|---|---|---|---|
| FVC zero-shot, 100 % | 76.3 | 74.4 | 75.6 | 75.4 ± 1.0 | 75.7 |
| FVC zero-shot, 10 % | 41.1 | 38.6 | 39.1 | 39.6 ± 1.3 | 40.5 |
| FVC fine-tuned, 100 % | 88.2 | 86.8 | 86.7 | 87.3 ± 0.9 | 87.2 |
| FVC fine-tuned, 50 % | 82.7 | 82.6 | 82.8 | 82.7 ± 0.1 | 83.9 |
| FVC fine-tuned, 10 % | 44.0 | 43.5 | 44.1 | 43.9 ± 0.3 | 45.7 |
| FVC fine-tuned EER, 100 % | 7.4 | 7.5 | 7.5 | 7.5 ± 0.1 | 7.7 |
| SOCOFing, 100 % | 88.4 | 86.8 | 88.2 | 87.8 ± 0.8 | 88.1 |
| SOCOFing, 50 % | 75.1 | 73.5 | 76.4 | 75.0 ± 1.5 | 75.7 |
| SOCOFing, 25 % | 60.8 | 58.5 | 60.0 | 59.8 ± 1.1 | 59.3 |
| SOCOFing, 10 % | 44.8 | 40.9 | 41.8 | 42.5 ± 2.0 | 43.6 |

## Reading it
1. **Seed noise is about 1-2 points** on every headline metric (sd of three runs; with n = 3 the sd itself is rough). The original run (seed 42) happened to be the best of
   the three on most rows, by about 1 point, so the Stage 9d headline numbers are slightly optimistic: 88.2 vs a three-run mean of 87.3 (FVC fine-tuned, 100 %), 88.4 vs 87.8
   (SOCOFing, 100 %), 44.8 vs 42.5 (SOCOFing, 10 %). Use the three-run means when quoting a single figure.
2. **Same seed does not mean the same result.** The seed-42 rerun differs from the original by up to 1.7 points (`+1.71` FVC fine-tuned 10 %, `-1.45` SOCOFing 25 %), so training
   on the GPU is not bit-for-bit deterministic (non-deterministic CUDA kernels and data-loading order effects; the loss curves differ slightly as well, best validation loss 1.3700 vs 1.3965).
   The rerun lands inside the spread of the independent seeds, i.e. a "same seed" rerun is as good as a new seed. Exact reproduction would need deterministic algorithms and a
   fixed-order data loader, which was not attempted.
3. **The conclusions do not depend on the seed.** Learned vs classical at partial areas is a gap of 25-40 points (SOCOFing 25 %: 59.8 ± 1.1 vs 30.9; 10 %: 42.5 ± 2.0 vs 6.1), far beyond
   seed noise. On full prints the comparison is small and runs the other way: SOCOFing classical 90.3 vs learned 87.8 ± 0.8 (a 2-3 point gap, about three sd), and FVC zero-shot learned 75.4 ± 1.0
   vs tuned classical 78.8. The FVC fine-tuned model is clearly ahead of classical (87.3 ± 0.9 vs 78.8). The classical numbers have no seed (deterministic) but carry the sampling error
   of the evaluation (about 1 point for FVC means).
4. The probe/crop draw was the same for all runs, so the table isolates training noise, not evaluation noise (that was measured in Stage 9c: sd 0.4-2.1 over probe/crop seeds).

## Limits
- Three runs per setting; one machine and one GPU; no other operating system or hardware tested.
- Stages 1-9c (original extractor) are verified through their stored result files only (`check`); they were not re-run in this stage.
- Data is not downloaded by the scripts (licences, Kaggle key, Springer ESM); the instructions are in the README.
- The seed study uses the FVC held-out fingers and sensors as before, so it measures training noise, not generalisation to new sensors.
