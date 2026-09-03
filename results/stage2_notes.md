# Stage 2 — evaluation harness

`src/evaluate.py` rewritten to a correct, split-aware protocol.

## Protocol
- **Test set:** only the 900 held-out fingers in `models/splits.json` (`test`). No
  train/val identity ever enters evaluation.
- **Gallery (enrolled template):** the clean `Real` print of each test finger — one per finger.
- **Probes:** every altered print of each test finger (`CR` central rotation / `Obl`
  obliteration / `Zcut` z-cut, at Easy / Medium / Hard), 7,398 in total.
- The gallery template is excluded from the probe set (no self-match).

## Metrics
- Rank-1 / Rank-5 closed-set identification + full CMC curve (`eval_cmc.png`)
- Verification EER + ROC AUC (genuine = probe vs own template; impostor = probe vs 20
  random other templates) → `eval_roc.png`, `eval_scores.png`
- Per-alteration Rank-1 breakdown

## Untrained-model sanity check  (`results/untrained_baseline/`)

Random-initialised encoder, same protocol:

| metric | value |
|---|---|
| Rank-1 | 9.7 % |
| Rank-5 | 19.3 % |
| EER | 27.5 % |
| ROC AUC | 0.805 |

Per-alteration Rank-1: Easy 10–27 %, Medium 4–12 %, Hard 3–5 %.

**Reading it:** the untrained model is *not* at the 1/900 ≈ 0.1 % chance line. SOCOFing
probes are geometric/occlusion edits of the *same* impression, so a probe shares most
of its minutiae with its gallery template; even random mean/max pooling of minutiae
positions maps them to similar vectors. The score histogram confirms a collapsed
embedding — all cosine similarities in [0.994, 1.000], genuine only slightly above
impostor. So **the real floor to beat is ~10 % Rank-1 / ~27 % EER**, and the
per-alteration ordering (Hard << Easy, rotation `CR` hardest) already behaves sensibly.
Rotation is the weakest case — absolute-coordinate node features are not
rotation-invariant (flagged for Stage 8).

## Usage
```bash
python src/evaluate.py --checkpoint models/best_model.pt --splits models/splits.json
python src/evaluate.py --untrained --config configs/default.yaml --splits models/splits.json
```
