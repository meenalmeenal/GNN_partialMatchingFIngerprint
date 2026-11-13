#!/usr/bin/env bash
# Reproduce the project's results. Run from the repository root:
#
#   bash scripts/reproduce.sh check     # 1 min : environment, smoke test, documented numbers vs result files
#   bash scripts/reproduce.sh quick     # ~5 min: tiny end-to-end run (30 fingers) of the final pipeline
#   bash scripts/reproduce.sh graphs    # ~20-60 min on a GPU (hours on CPU): LEADER graphs for SOCOFing + FVC
#   bash scripts/reproduce.sh train     # ~31 min on a GPU (hours on CPU): SOCOFing training + FVC fine-tuning of the final model
#   bash scripts/reproduce.sh eval      # ~2 h (classical matcher dominates): every headline table and figure
#   bash scripts/reproduce.sh seeds     # ~2 h GPU + 1 h CPU: training-seed replication (Stage 10)
#
# Data is NOT downloaded automatically (licences): see README "Data". Needed: data/SOCOFing_organized/ (from the
# Kaggle SOCOFing archive, via scripts/stage1.sh) and data/FVC/74034_3_En_4_MOESM1_ESM/ (Springer ESM of the
# Handbook of Fingerprint Recognition, 3rd ed.). The Stage 1-9c results that used the original hand-built extractor
# need data/graphs + data/packed (scripts/stage1.sh + src/pack_graphs.py) and are not re-run here (README, "Legacy").
set -euo pipefail
export KERAS_BACKEND=torch
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
FVC_DBS="fvc2002_db1 fvc2002_db2 fvc2002_db3 fvc2002_db4 fvc2004_db1 fvc2004_db2 fvc2004_db3 fvc2004_db4"
DEVICE="${DEVICE:-auto}"          # auto = cuda if available

step_check() {
  python - <<'PY'
import sys, torch
print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda", torch.cuda.is_available())
import numpy, keras, pyfing, torch_geometric, cv2, sklearn, skimage
assert numpy.__version__.startswith("1."), "numpy must be < 2"
print("numpy", numpy.__version__, "| keras", keras.__version__, "(backend", keras.backend.backend() + ")", "| pyfing ok")
PY
  python src/smoke_test.py | tail -2
  python src/verify_results.py
}

step_quick() {
  rm -rf data/cache/quick results/quick_test models/quick_test.pt
  python src/leader_pipeline.py socofing --limit 30 --out-socofing data/cache/quick
  python src/train_v2.py --arch inv_local --tag quick_test --packed-dir data/cache/quick --epochs 2 \
      --iters-per-epoch 10 --device "$DEVICE"
  python src/eval_fixes.py --modes centred --seeds 1 --levels 1.0 0.5 --packed-dir data/cache/quick \
      --checkpoint models/quick_test.pt --cache-tag _quick --out data/cache/quick/eval.json
  rm -rf results/quick_test models/quick_test.pt
  echo "quick run finished: LEADER extraction -> graphs -> training -> evaluation all executed"
}

step_graphs() {
  python src/leader_pipeline.py socofing
  python src/leader_pipeline.py fvc
}

step_train() {
  python src/train_v2.py --arch inv_local --tag stage9d_leader_inv_local --packed-dir data/packed_leader \
      --device "$DEVICE" --epochs 90 --patience 10
  python src/train_v2.py --arch inv_local --init models/stage9d_leader_inv_local.pt --lr 3e-4 --epochs 40 \
      --iters-per-epoch 200 --tag stage9d_leader_fvcft --fvc-dir data/fvc_packed_leader --device "$DEVICE" \
      --fvc-dbs $FVC_DBS
}

step_eval() {
  python src/tune_classical.py --dir data/fvc_packed_leader_b --out results/stage9d/classical_tuned_leader.json
  for DB in $FVC_DBS; do
    A="--db $DB --packed-dir data/fvc_packed_leader --finger-range 51-100"
    python src/eval_fvc.py $A --models-v2 inv_local_oldmodel=models/stage9_inv_local_cont.pt --outdir results/stage9d/fvc_oldmodel
    python src/eval_fvc.py $A --no-classical --classical-tuned results/stage9d/classical_tuned_leader.json --outdir results/stage9d/fvc_tuned
    python src/eval_fvc.py $A --no-classical --models-v2 inv_local_leader=models/stage9d_leader_inv_local.pt --outdir results/stage9d/fvc_newmodel
    python src/eval_fvc.py $A --no-classical --models-v2 inv_local_leader_fvcft=models/stage9d_leader_fvcft.pt --outdir results/stage9d/fvc_ft
  done
  python src/eval_fixes.py --modes centred --packed-dir data/packed_leader --checkpoint models/stage9d_leader_inv_local.pt \
      --cache-tag _leader --out results/stage9d/sweep_centred_leader.json
  python src/plot_stage9d.py
  python src/calibrate_reject.py
  python src/verify_results.py --tol 4.0
}

step_seeds() { bash scripts/seed_replication.sh; python src/aggregate_seeds.py; }

case "${1:-check}" in
  check) step_check ;; quick) step_quick ;; graphs) step_graphs ;; train) step_train ;;
  eval) step_eval ;; seeds) step_seeds ;;
  *) echo "usage: bash scripts/reproduce.sh {check|quick|graphs|train|eval|seeds}"; exit 2 ;;
esac
