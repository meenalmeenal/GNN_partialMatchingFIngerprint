# Project status

*Last updated at the end of Stage 10 (everything except the final report is done).*

## 1. The project in one paragraph
We want to identify a person from a **partial or damaged fingerprint**. A print is turned into a small graph of its minutiae (ridge endings and
bifurcations). A graph neural network gives every minutia a descriptor that does not change when the print is rotated or shifted, and a query is
scored against each enrolled full print by how well its minutiae find matching minutiae there. We compare this learned matcher with a classical
hand-designed matcher (Hough alignment) and study how both degrade as the visible part of the print shrinks from 100 % to 10 %.
There is also an interactive demo: give it a (partial) print and it lists the closest enrolled prints.

## 2. Where we are: 10 of 11 stages done

| Stage | What | Status | Detail |
|---|---|---|---|
| 1, 1.5 | Data pipeline (SOCOFing -> graphs, finger-level splits) and first hand-built minutiae extractor | done | `results/stage1_report.json`, `src/minutiae.py` |
| 2 | Evaluation harness (Rank-1/5, CMC, EER, AUC, per-alteration, untrained floor) | done | `results/stage2_notes.md` |
| 3 | First GNN (triplet loss) | done | `results/stage3_notes.md` |
| 4 | Embedding diagnostics (why identification failed) | done | `results/stage4_notes.md` |
| 5 | Classical Hough baseline | done | `results/stage5_notes.md` |
| 6, 6b | Partiality sweep (crop to 100...10 %), augmented retraining | done | `results/stage6_notes.md`, `stage6b_notes.md` |
| 9 | Better matcher (invariant GNN + MaxSim + contrastive loss) | done | `results/stage9_notes.md` |
| 8 | Cross-dataset test on real FVC prints | done | `results/stage8_notes.md` |
| 7 | Interactive demo (Streamlit) | done | `results/stage7_notes.md`, `app/app.py` |
| 9c | Evaluation fixes: 900-finger gallery, CIs, open-set, tuned classical baseline | done | `results/stage9c_notes.md` |
| 9d | **Robust extractor (pretrained LEADER) + GPU retraining** | done | `results/stage9d_notes.md` |
| 9e | Demo upgraded to the best pipeline, "enrolled or not" verdict | done | `results/stage9e_notes.md` |
| 10 | Reproducibility: pinned environment, scripts, seed study | done | `results/stage10_notes.md` |
| 11 | Final report and presentation | **to do** | none yet |

Stage numbers are not chronological: 9 was done before 8 and 7, and 9c-9e are follow-ups. Each stage lives on its own `stage-*` branch with one
commit, and all of them are merged into `main`.

## 3. Results that matter

**Rank-1 identification (higher is better). "Area" = share of the query print that is visible.**

| Setting | 100 % area | 50 % | 25 % | 10 % |
|---|---|---|---|---|
| SOCOFing, 897-print gallery, classical | 90 | 69 | 31 | 6 |
| SOCOFing, 897-print gallery, **learned** | 88 | 75 | 61 | 45 |
| Real FVC prints, held-out fingers (50-print galleries), classical, tuned | 79 | 74 | 59 | 30 |
| Real FVC prints, learned, trained on SOCOFing only | 76 | 72 | 61 | 41 |
| Real FVC prints, **learned, fine-tuned on 40 FVC fingers per database** | **88** | 83 | 70 | 44 |

Verification EER on real FVC prints (full print): 7.4 % for the fine-tuned model, 16 % for the tuned classical matcher.
SOCOFing windows are centred on a minutia (an easier partial-print model than uniformly random windows, see Stage 9c). The training-seed spread is
about 1 to 2 points (Stage 10).

**The story in five sentences**
1. The first GNN (one pooled embedding of absolute coordinates) was *much worse* than the classical matcher (Rank-1 10.8 % vs 89.3 %), and data tricks
   (cropping, rotation augmentation) did not fix it: the representation was the problem (Stages 3 to 6b).
2. Changing to per-minutia descriptors built from relative geometry, scored by MaxSim, fixed that: it matches the classical matcher on full prints
   and degrades far more gracefully on partial ones (Stage 9).
3. On real FVC impressions everything collapsed (about 13 to 17 % Rank-1) because our hand-built minutiae extractor was not repeatable across impressions (Stage 8).
4. Swapping in a pretrained neural minutiae detector (LEADER) lifted *both* matchers to 75 to 80 %, and retraining plus fine-tuning gave 88 % (Stage 9d).
   The extractor, not the matcher, was the bottleneck.
5. Throughout, the learned matcher's real advantage is on **small partial prints**; on full prints the classical matcher is as good or slightly better.

## 4. Things you must know before quoting numbers (honest limits)
- **SOCOFing is optimistic**: the damaged probes are software edits of the same impression as the gallery print. FVC (real, independent impressions) is the realistic test.
- **LEADER saw part of FVC**: its authors trained it on FVC2002 DB3, FVC2004 DB1 and DB3 (and 100 images of DB2). We report results split by this (`results/stage9d_notes.md`);
  the improvement is still large on the databases it never saw (about 19 % to 84 % for the fine-tuned model), but the six-database means are somewhat optimistic.
- **Fine-tuned FVC results are in-domain**: held-out *fingers*, but the same sensors and sessions as the 40 training fingers. This is not generalisation to a new sensor.
- **Small test sets**: 50 held-out fingers per FVC database (about 350 queries); one standard error is about 2 points per database and 1 on the six-database mean.
- **Seeds**: three independent training runs differ by about 1 to 2 points; the original runs quoted in Stage 9d are about 1 point better than the three-run mean.
  GPU training is not bit-for-bit repeatable.
- **Partial prints are simulated** as square windows over extracted minutiae. Real latent prints (smudged, irregular, overlaid) are not covered; NIST SD27/SD302 are not available to us.
- **Demo**: uploaded images only work if they look like SOCOFing or like the selected FVC sensor.
- The old Stage 1 to 9c result files used the hand-built extractor and remain valid as "before" numbers; the demo and the headline numbers now use LEADER.

## 5. What is left

**Required**
- **Stage 11: final report and presentation.** All the content is available (this document and the stage notes). The mid-term report and presentation (made earlier from the college
  templates) stop at Stage 6b and need updating with Stages 9 to 10: the new method, the extractor finding, the real-print results, the seed study and the limitations.
  The semester label ("VII") on the mid-term documents was inferred and should be confirmed.

**Optional improvements (not started; roughly in order of value)**
1. **New-sensor generalisation test**: fine-tune on some FVC databases and test on a sensor never used in fine-tuning (leave-one-database-out). This is the cleanest answer to
   "does it work on a new scanner?".
2. Demo: let a user **enrol their own gallery**; calibrate the "enrolled?" thresholds per user gallery.
3. Irregular partial shapes (random masks, smudges) instead of square windows; harder noise models.
4. Bigger matcher, longer training, ensembling (all cheap now on the GPU); a milder synthetic-noise schedule (the first attempt hurt).
5. Latent-print data (NIST SD27/SD302) if access can be obtained.

## 6. Suggested split of the remaining work between teammates
| Task | Needs | Effort |
|---|---|---|
| Final report text (introduction, related work, method, experiments, discussion) | this folder and `results/stage*_notes.md` | main chunk |
| Updated slides with the Stage 9d/10 figures (`results/stage9d/headline.png`, `results/stage9c/headline.png`, `results/stage10/seeds.png`, `results/stage9e/examples.png`) | PowerPoint template | medium |
| Re-run `scripts/reproduce.sh check` and `quick` on your own machine and report any problem | `GETTING_STARTED.md` | small, very useful |
| Leave-one-sensor-out experiment | GPU or patience; `src/train_v2.py --fvc-dbs ...` | medium |
| Demo polish / user galleries | `app/app.py`, `src/demo_core.py` | medium |

## 7. Repository map (what to open)
- `results/stageN_notes.md`: one write-up per stage (what was run, numbers, interpretation, caveats, next step). Read these before touching a stage.
- `results/<stage>/`: metrics (json), tables (md), plots (png). `results/stage9d_leader_*/best_model.pt` are the final checkpoints (tracked, about 200 KB each).
- `src/`: all code (layout in the top-level README). `app/app.py`: demo. `scripts/`: `reproduce.sh`, `seed_replication.sh`, `stage1.sh`.
- `models/splits.json`: **the train/val/test split by finger. Never regenerate it**: every result depends on it.
- `data/`: datasets and graphs, not in git (see GETTING_STARTED).

## 8. Branch / commit map (all merged into `main`)
| Stage | Branch | Commit |
|---|---|---|
| 1 to 5 | `stage-1-data-pipeline` to `stage-5-classical-baseline` | merged earlier at `8fc8bd1` |
| 6 | `stage-6-partiality-sweep` | `17bc54d` |
| 6b | `stage-6b-augmented-training` | `3882b32` |
| 9 | `stage-9-model-improvements` | `57123e6` |
| 8 | `stage-8-cross-dataset` | `6a425a3` |
| 7 | `stage-7-interface` | `7c1f9d5` |
| 9c | `stage-9c-eval-fixes` | `3e4dadc` |
| 9d | `stage-9d-extractor` | `89be36e` |
| 9e | `stage-9e-demo-leader` | `7dbeb0d` |
| 10 | `stage-10-reproducibility` | `d3b147c` |
