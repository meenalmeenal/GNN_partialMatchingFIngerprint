#!/usr/bin/env bash
# Stage 1 — data acquisition & preprocessing.
# Run from repo root:  bash scripts/stage1.sh
set -e

RAW=data/SOCOFing
ORG=data/SOCOFing_organized
GRAPHS=data/graphs

# 1. download SOCOFing (needs ~/.kaggle/kaggle.json). Skips if already present.
if [ ! -d "$RAW" ]; then
  echo ">> downloading SOCOFing from Kaggle..."
  kaggle datasets download -d ruizgara/socofing -p data/ --unzip
  # the archive extracts to data/SOCOFing/ (Real/ + Altered/)
fi

# 2. organize into <identity>/<sample> tree
echo ">> organizing..."
python src/prepare_socofing.py --input "$RAW" --output "$ORG" --link

# 3. images -> minutiae graphs
echo ">> building graphs (this is the slow step)..."
python src/graph_build.py --input "$ORG" --output "$GRAPHS"

# 4. train/val/test split over finger identities
echo ">> writing splits..."
python src/make_splits.py --graph_dir "$GRAPHS" --config configs/default.yaml

# 5. sanity report
echo ">> stage 1 report..."
python src/stage1_report.py --organized "$ORG" --graphs "$GRAPHS" --splits models/splits.json

echo ">> Stage 1 done."
