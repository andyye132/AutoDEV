#!/usr/bin/env bash
# Pull finished clean-level scans from Hyak, classify any that are not yet classified (replay + severity), rebuild the
# real-bug table and summary. Safe to rerun; only new scan files are classified.
cd /Users/andy132/projects/research/ACT/AutoDev; . .venv/bin/activate
mkdir -p runs/hyak
rsync -az --timeout=120 'hyak:/gscratch/raivn/andy132/AutoDev/runs/hyak/scan_*.json' runs/hyak/ 2>/dev/null
rsync -az --timeout=120 'hyak:/gscratch/raivn/andy132/AutoDev/runs/hyak/fresh_levels' runs/hyak/ 2>/dev/null
FILES=""
for f in runs/hyak/scan_shipped*.json runs/hyak/scan_fresh*.json; do
  [ -f "$f" ] || continue
  case "$f" in *classified*|*_racy*) continue;; esac   # *_racy = pre-fix (archive-race) scans, never used
  c="${f%.json}.classified.json"
  if [ ! -f "$c" ] || [ "$f" -nt "$c" ]; then
    idx=""; case "$f" in *fresh*) idx="--index";; esac
    echo "classifying $f $idx"; JAX_PLATFORMS=cpu .venv/bin/python scripts/classify_hits.py --scan "$f" --size l $idx > "${f%.json}.classify.log" 2>&1 || { echo "classify failed: $f"; tail -3 "${f%.json}.classify.log"; }
  fi
  [ -f "$c" ] && FILES="$FILES,$c"
done
# scans without the lineage log are only used until their lineage-logged replacements exist
if ls runs/hyak/scan_fresh_builtins_log.classified.json >/dev/null 2>&1; then
  FILES=$(echo "$FILES" | tr ',' '\n' | grep -v "scan_fresh.classified\|scan_shipped.classified" | tr '\n' ',' | sed 's/,$//')
else
  :   # (pre-fix local classifications are quarantined as runs/hyak_scan_*_racy.classified.json and are not used)
fi
FILES="${FILES#,}"
echo "tables from: $FILES"
python scripts/make_bug_table.py --classified "$FILES" --tex results/tables/table_bugs.tex --out results/bugs.json | tail -25
