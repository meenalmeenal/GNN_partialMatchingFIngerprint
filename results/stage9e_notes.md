# Stage 9e — demo upgraded to the LEADER extractor and the retrained models

Follow-up to Stage 9d: the Stage 7 demo still used the old extractor and models, i.e. the weakest pipeline in the repo. This stage
switches it to the best one and adds an "is this finger enrolled at all?" verdict (open-set decision).

## What changed
- `src/demo_core.py`: SOCOFing and FVC galleries now load the LEADER graphs (`data/packed_leader`, `data/fvc_packed_leader`);
  uploaded images go through LEADER (SOCOFing: 160x160; FVC: the database's fixed ridge-period scale, as in Stage 9d). On a GPU machine
  the single-image `Leader.run()` of pyfing fails (it calls `.numpy()` on a CUDA tensor), so the app uses its batch call `run_on_db`.
  Checked that an uploaded image reproduces the stored graph exactly (SOCOFing 23/23, 12/12, 17/17, 18/18, 31/31 minutiae; FVC 49/49, 62/62, 50/50, 46/46).
- Models: `stage9d_leader_inv_local` (SOCOFing) and `stage9d_leader_fvcft` (fine-tuned on real FVC). The classical comparison on the FVC galleries
  uses the tolerances tuned on FVC set B (Stage 9d), a fairer baseline than the defaults.
- **Reject verdict** (`src/calibrate_reject.py`, `results/stage9e/reject_thresholds.json`): the top match's z-score (score vs all enrolled prints) is compared
  to thresholds calibrated per gallery and print area with half of the fingers enrolled and half unknown (4 random splits). "Likely enrolled"
  = z above the 1 % false-accept threshold, "Uncertain" = between the 10 % and 1 % thresholds, "Probably not enrolled" below. The page states the
  miss rate: e.g. on SOCOFing at 50 % area only 41 % (lenient) / 33 % (strict) of enrolled prints are identified *and* accepted; on FVC2002 DB1 full prints 95 / 91 %.
- Sidebar and "how it works" text updated with the Stage 9d numbers and the caveats (LEADER saw part of FVC; fine-tuning is in-domain).
- `src/eval_fixes.py` keeps defaulting to the Stage 9c model/graphs (pass `--packed-dir`/`--checkpoint` for others), so Stage 9c is still reproducible.

## What the demo does now (numbers for exactly its protocol: one random probe per finger, window placed uniformly at random, whole gallery)
`results/stage9e/demo_numbers.*` — Rank-1 %:

| setting | 100 % | 50 % | 25 % |
|---|---|---|---|
| SOCOFing, 897-print gallery (300 query fingers), learned | 89.0 | 39.0 | 24.0 |
| SOCOFing, 897-print gallery, classical | 90.3 | 38.0 | 13.3 |
| FVC real sensors (6 DBs, 50-print galleries), fine-tuned learned | 87.3 | 84.3 | 73.3 |
| FVC real sensors (same), classical (tuned) | 75.3 | 74.7 | 68.0 |

Uniform random windows are the hard case on SOCOFing (about 40 % at half the print versus 75 % with minutia-centred windows in Stage 9d/9c). On FVC they are
not harder because those prints carry far more minutiae. The four fixed-seed examples in `results/stage9e/examples.png` all hit rank 1, which is not
representative of the SOCOFing partial rows above.
Standard error: about 2.7 points on SOCOFing (300 queries), about 1.7 on the 6-database FVC means (50 queries each, so single databases are noisy).

## Caveats
- The reject verdict is calibrated on the benchmark's own distribution (same sensors, same window shape) and rejects some enrolled prints; for example on
  FVC2004 DB2 the default sample is ranked #1 yet judged "probably not enrolled" (z +1.9 vs the lenient threshold 2.1; only 58 % of enrolled prints pass it at 50 % area on that database). It is a statement about score distributions, not a guarantee.
- Not exercised in a browser: the file-upload widget itself (only the extraction behind it, which matches the stored graphs exactly).
- Same caveats as Stage 9d: the extractor was trained on part of FVC, the fine-tuned model is in-domain.
