# Getting started

How to set the project up, see it work, and re-run things. Read [PROJECT_STATUS.md](PROJECT_STATUS.md) first if you want the big picture.
Commands are for **Git Bash on Windows** (the `.sh` scripts need a bash shell) and are run **from the repository root**.

## 1. What you can do at each level of effort
| You have | You can |
|---|---|
| only the repository (no datasets) | read everything in `results/`, run `bash scripts/reproduce.sh check` (environment + synthetic smoke test + verifies that the documented numbers match the stored result files) |
| + SOCOFing and FVC downloaded (about 1.5 GB) and ~1 h of compute | build the graphs, run the demo, re-run evaluations |
| + an NVIDIA GPU | train the final models in about 30 minutes instead of hours |

## 2. Install
```bash
git clone https://github.com/meenalmeenal/GNN_partialMatchingFIngerprint.git
cd GNN_partialMatchingFIngerprint
python -m venv .venv && source .venv/Scripts/activate      # optional but recommended; Python 3.12 was used
pip install -r requirements.txt
# optional, NVIDIA GPU (4-5x faster training, ~10 ms per image minutiae extraction):
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 --index-url https://download.pytorch.org/whl/cu121
bash scripts/reproduce.sh check
```
`check` should end with `15/15 checks passed` and `SMOKE TEST PASSED`. If it does not, see Troubleshooting below.

## 3. Get the data (not in git)
1. **SOCOFing**: download from Kaggle (`ruizgara/socofing`) and unzip so that `data/SOCOFing/Real/` and `data/SOCOFing/Altered/` exist, then
   ```bash
   python src/prepare_socofing.py --input data/SOCOFing --output data/SOCOFing_organized --link
   ```
   (`bash scripts/stage1.sh` does the download with a Kaggle API key but also rebuilds the *old* graphs, which you do not need.)
2. **FVC2002 / FVC2004**: the supplement (Springer ESM) of the *Handbook of Fingerprint Recognition*, unzipped to `data/FVC/74034_3_En_4_MOESM1_ESM/`
   (it contains `FVC2002/Dbs/Db1_a ...` and `FVC2004/...`). If you cannot get it, ask a teammate who already has it. Check each dataset's terms before sharing copies.
3. **Graphs** (LEADER minutiae detector + graph building; the detector weights come with the `pyfing` package):
   ```bash
   bash scripts/reproduce.sh graphs        # about 20-60 min on a GPU; writes data/packed_leader/ and data/fvc_packed_leader*/
   ```
   Or, to get going faster, only the SOCOFing test split: `python src/leader_pipeline.py socofing --limit 100 --out-socofing data/packed_leader`
   (the demo then works with 100 test fingers per split).

## 4. Run the demo
```bash
streamlit run app/app.py          # opens http://localhost:8501 ; Ctrl+C to stop
```
1. Left sidebar: choose the **gallery** (SOCOFing, or one of the FVC databases = real prints). The model is chosen for you.
2. Choose a **query**: a sample from the dataset (finger + probe type) or an uploaded image (png/jpg/bmp/tif).
3. Use **Print area kept** and the **window position** sliders to cut the query down to a partial print; the orange box shows what is kept.
4. Read the results: closest enrolled prints from the learned matcher and from the classical matcher, the verdict "likely enrolled / uncertain / probably not enrolled",
   and under "3. Why did it pick this one?" the minutiae links that produced the score.
Restart Streamlit after editing anything under `src/` (it caches imported modules).

## 5. Re-run the experiments
`bash scripts/reproduce.sh <mode>`: `check` (1 min), `quick` (a few min, tiny end-to-end run), `graphs`, `train` (about 31 min GPU), `eval` (about 2 h),
`seeds` (about 2 h GPU + 1 h CPU). `python src/verify_results.py --tol 4` then compares your new numbers with the documented ones (retraining moves them by 1 to 2 points).
The mode table with the exact commands is in `scripts/reproduce.sh`; every stage's own command list is at the bottom of its `results/stageN_notes.md` ("Reproduce").

**Data flow of the final pipeline**
```
data/SOCOFing_organized/<finger>/<print>.BMP --src/leader_pipeline.py--> data/packed_leader/{train,val,test}.pt   {finger: {print: graph}}
data/FVC/.../Db*_a|b/*.tif                   --src/leader_pipeline.py--> data/fvc_packed_leader[_b]/<db>.pt       {finger: {impression: graph}}
data/packed_leader  --src/train_v2.py--> models/stage9d_leader_inv_local.pt --(fine-tune on FVC fingers 1-40)--> models/stage9d_leader_fvcft.pt
models + graphs --src/eval_fvc.py | src/eval_fixes.py--> results/stage9d/*.json --src/plot_stage9d.py--> tables + figures
models + graphs --app/app.py (src/demo_core.py)--> the demo
```
Checkpoints that results depend on are also copied into `results/stage9d_leader_*/best_model.pt` (tracked); `models/*.pt` is gitignored.

## 6. Working rules (so we do not break each other's results)
- **Never regenerate `models/splits.json`** (split by finger, seed 42): all results use it. Never evaluate on train or validation fingers.
- One stage = one branch `stage-N-<short-name>` off `main` = one commit containing code, `results/stageN_notes.md` (what was run, numbers, interpretation, next step),
  plots/metrics in `results/stageN/`, and an updated README Status entry. Report negative results too.
- Stage files explicitly (`git add <paths>`), never the whole tree. Do not commit `data/` or large checkpoints in `models/`.
- Use your own git name/e-mail for your commits.
- Compare methods on the **same finger subset** and report the gallery size; compare to the untrained floor and the classical baseline.
- `graph_build.py` (old extractor) uses multiprocessing: run **one** build at a time (set `OMP_NUM_THREADS=1`), otherwise two jobs overwrite each other.
- Do not run evaluations while training: both are CPU-heavy and slow each other down.

## 7. Troubleshooting
| Symptom | Fix |
|---|---|
| "A module compiled using NumPy 1.x cannot be run in NumPy 2" | `pip install "numpy==1.26.4"` (torch 2.4.1 and torch-geometric need numpy < 2) |
| `ModuleNotFoundError: tensorflow` when importing pyfing/keras | set `KERAS_BACKEND=torch` (the scripts do it; do it yourself in notebooks) |
| `can't convert cuda:0 device type tensor to numpy` from `pyfing` | pyfing's single-image `run()` fails on a GPU; use `run_on_db([img])` (as `demo_core.py` does) or run on the CPU |
| `CUDA out of memory` while extracting FVC images | smaller batch (`leader_pipeline.py` already uses 4); 4 GB GPUs cannot take large batches |
| `torch.cuda.is_available()` is False | the CPU wheel is installed: install the `+cu121` build (section 2) and check `nvidia-smi` |
| Demo shows old behaviour after editing code | stop and restart `streamlit run app/app.py` |
| Demo says a gallery or dataset is missing | build the graphs (section 3) and check that `data/packed_leader/test.pt` exists |
| `.sh` script not found / not executable | run it from Git Bash with `bash scripts/<name>.sh` |
| Numbers differ by 1-2 points after retraining | expected: seed noise and non-deterministic GPU kernels (`results/stage10_notes.md`) |

## 8. Where to read the code first (10 minutes)
1. `src/demo_core.py`: how a print becomes a graph, is encoded and searched (small, commented).
2. `src/model_v2.py`: the invariant GNN and the MaxSim score (`Matcher`, `sim_matrix`).
3. `src/train_v2.py`: the contrastive training loop. 4. `src/classical_match.py`: the Hough baseline. 5. `src/leader_pipeline.py`: extraction.
