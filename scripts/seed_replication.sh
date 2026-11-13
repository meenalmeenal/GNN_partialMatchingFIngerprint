#!/usr/bin/env bash
# Stage 10: training-seed replication of the final (LEADER) pipeline.
#   seeds 1 and 2  = independent training runs (variance estimate)
#   seed 42 again  = same seed as the original Stage 9d run (determinism check)
# Each seed: train from scratch on SOCOFing LEADER graphs -> fine-tune on FVC fingers 1-40 -> evaluate
# (FVC held-out fingers 51-100 zero-shot and fine-tuned; SOCOFing 897-finger gallery, learned model only).
# Run from the repo root; needs the GPU build of torch (about 21 + 9 min per seed) and ~30 min of CPU eval.
set -e
# a run counts as finished when its training-curve plot exists (the checkpoint appears after epoch 1)
export KERAS_BACKEND=torch OMP_NUM_THREADS=4
FVC_DBS="fvc2002_db1 fvc2002_db2 fvc2002_db3 fvc2002_db4 fvc2004_db1 fvc2004_db2 fvc2004_db3 fvc2004_db4"
SEEDS="${SEEDS:-1 2 42}"
for S in $SEEDS; do
  T=stage10_seed$S
  [ -f results/${T}/train_curve.png ] || python src/train_v2.py --arch inv_local --tag $T --seed $S --packed-dir data/packed_leader \
        --device cuda --epochs 90 --patience 10
  [ -f results/${T}_ft/train_curve.png ] || python src/train_v2.py --arch inv_local --init models/${T}.pt --lr 3e-4 --epochs 40 \
        --iters-per-epoch 200 --tag ${T}_ft --seed $S --fvc-dir data/fvc_packed_leader --device cuda --fvc-dbs $FVC_DBS
  for DB in $FVC_DBS; do
    python src/eval_fvc.py --db $DB --packed-dir data/fvc_packed_leader --finger-range 51-100 --no-classical \
        --models-v2 zs=models/${T}.pt ft=models/${T}_ft.pt --outdir results/stage10/seed$S
  done
  python src/eval_fixes.py --modes centred --seeds 1 --no-classical --packed-dir data/packed_leader \
        --checkpoint models/${T}.pt --cache-tag _$T --out results/stage10/seed$S/socofing_centred.json
done
