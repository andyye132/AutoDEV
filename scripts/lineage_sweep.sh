#!/usr/bin/env bash
# Overnight invariant sweep on the laptop CPU: the lineage audit on every shipped level (fixed harness). One line per level.
cd /Users/andy132/projects/research/ACT/AutoDev; . .venv/bin/activate
OUT=runs/lineage_sweep.log; : > $OUT
for f in $(python -c "import json; print(' '.join(r['level'] for r in json.load(open('runs/hyak/scan_shipped_evolved.json'))['levels']))"); do
  n=$(basename $f .json)
  r=$(SCAN=runs/hyak/scan_shipped_evolved.json LEVEL=$n W=32 T=256 K=64 JAX_PLATFORMS=cpu timeout 900 python scripts/lineage_audit.py 2>&1 | grep "archived states" | tail -1)
  echo "$n: ${r:-ERROR}" >> $OUT
done
echo "SWEEP_DONE" >> $OUT
