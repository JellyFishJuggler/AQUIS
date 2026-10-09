#!/bin/bash
# run_remaining.sh — Track 2 -> Track 3 driver. Launch via setsid so tool
# timeouts / suspend cannot kill it. Station CV already done (stationcv.json).
# Logs: data/scratch_exp/logs/{track2,track3}.log
set -u
cd /home/srijan/Downloads/Developement/AQUIS/ml
L=data/scratch_exp/logs
PY=venv/bin/python
V312=venv312/bin

echo "=== [$(date '+%F %T')] track2 start"
$PY data/scratch_exp/exp_track2_tiers.py > "$L/track2.log" 2>&1
rc=$?
echo "=== [$(date '+%F %T')] track2 rc=$rc"

echo "=== [$(date '+%F %T')] track3 gate: wait for pip (max 40 min)"
for i in $(seq 1 40); do
  pgrep -f "venv312/bin/pip install" >/dev/null || break
  sleep 60
done
if $V312/python -c "import torch, xgboost, sklearn, scipy, pandas, pyarrow" 2>/dev/null; then
  echo "=== [$(date '+%F %T')] track3 start (timeout 10800)"
  timeout 10800 $V312/python data/scratch_exp/exp_track3_dl.py > "$L/track3.log" 2>&1
  rc=$?
  echo "=== [$(date '+%F %T')] track3 rc=$rc"
  if [ $rc -eq 124 ] && [ ! -f data/scratch_exp/tracks/track3_result.json ]; then
    echo '{"verdict":"TIMEOUT","reason":"3h wall-clock box hit before result written"}' \
      > data/scratch_exp/tracks/track3_result.json
  fi
else
  echo "=== [$(date '+%F %T')] track3 SKIP (venv312 deps incomplete)"
  echo '{"verdict":"SKIP","reason":"torch/xgboost/scipy/pandas/pyarrow not importable in venv312"}' \
    > data/scratch_exp/tracks/track3_result.json
fi
echo "=== [$(date '+%F %T')] ALL DONE"
