#!/usr/bin/env bash
# Budget-sized Named vs Masked comparison plus inspection, for one GPU session (~1.5 h on an H200).
# Resumable: rerunning continues unfinished runs from their last saved state.
#   nohup bash scripts/pod_budget_run.sh > runs/budget.log 2>&1 &
set -uo pipefail
export HF_HOME=${HF_HOME:-/root/hf}
source /root/venv/bin/activate
MODEL=${MODEL:-Qwen/Qwen3.6-35B-A3B}
OUT=runs/budget
COMMON=(--backend hf --model "$MODEL" --world counterfactual --seed 0 --updates 8 --seeds-per-update 4
        --eval-every 4 --eval-seeds 2 --lr 4e-5 --loss renormalised --checkpoint-every 2 --out "$OUT")
for r in named masked; do
  if [ ! -f "$OUT/$r-cf/result.json" ]; then
    echo "=== train $r: $(date -u +%T)"
    resume=(); [ -f "$OUT/$r-cf/state.json" ] && resume=(--resume)
    iuw train "${COMMON[@]}" --rendering "$r" --run-id "$r-cf" "${resume[@]}" > "$OUT/$r-train.json" 2> "$OUT/$r-train.err" \
      || { echo "train $r failed"; tail -5 "$OUT/$r-train.err"; exit 1; }
  fi
  echo "=== inspect $r: $(date -u +%T)"
  python scripts/inspect_run.py --model "$MODEL" --adapter "runs/hf-checkpoints/$r-cf-u008" --rendering "$r" \
    --mapping 0 --seeds 4 --return-trip-seeds 1 --out "$OUT/inspect-$r" > "$OUT/$r-inspect.log" 2> "$OUT/$r-inspect.err" \
    || { echo "inspect $r failed"; tail -5 "$OUT/$r-inspect.err"; }
done
iuw analyze --runs "$OUT" > "$OUT/analysis.json" 2> "$OUT/analysis.err"
echo "=== done $(date -u +%T)"
