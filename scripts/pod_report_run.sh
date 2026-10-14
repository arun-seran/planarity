#!/usr/bin/env bash
# Fills the 2x2 grid (Named/Masked x counterfactual/reference) to three seeds per cell with the
# budget-run settings, inspects the counterfactual adapters, then writes analysis.json and report.md.
# The two seed-0 counterfactual runs from scripts/pod_budget_run.sh are reused when present
# (copy runs/budget/{named,masked}-cf and runs/hf-checkpoints/{named,masked}-cf-u00{4,8} first).
# Resumable: finished runs and inspections are skipped, interrupted runs resume.
#
#   nohup bash scripts/pod_report_run.sh > runs/report.log 2>&1 &                 # one pod
#   SHARD=0 NSHARDS=2 nohup bash scripts/pod_report_run.sh > runs/report.log 2>&1 & # pod 1 of 2
#   SHARD=1 NSHARDS=2 ...                                                          # pod 2 of 2
#   BACKEND=mock OUT=/tmp/x bash scripts/pod_report_run.sh                          # offline dry run
#
# JOBS="named:reference:0 masked:counterfactual:1 ..." replaces the SEEDS grid with an explicit,
# ordered job list (useful when the budget only covers part of the grid).
# INSPECT=matched (default) inspects Named u8 and Masked u4 + u8 of seed 0 (Masked u4 is the
# checkpoint whose score matches Named u8); INSPECT=all adds u8 of every counterfactual run;
# INSPECT=none skips inspection.
set -uo pipefail
BACKEND=${BACKEND:-hf}
MODEL=${MODEL:-Qwen/Qwen3.6-35B-A3B}
OUT=${OUT:-runs/report}
SEEDS=${SEEDS:-"0 1 2"}
SHARD=${SHARD:-0}
NSHARDS=${NSHARDS:-1}
UPDATES=${UPDATES:-8}
INSPECT=${INSPECT:-matched}
CKPT=runs/hf-checkpoints
if [ "$BACKEND" = hf ]; then
  export HF_HOME=${HF_HOME:-/root/hf}
  [ -f /root/venv/bin/activate ] && source /root/venv/bin/activate
fi
export PYTHONUNBUFFERED=1
mkdir -p "$OUT"
COMMON=(--backend "$BACKEND" --model "$MODEL" --updates "$UPDATES" --seeds-per-update 4 --eval-every 4
        --eval-seeds 2 --lr 4e-5 --loss renormalised --checkpoint-every 2 --out "$OUT")

run_id() {  # rendering world seed; seed-0 counterfactual keeps the budget run's name
  local w=cf; [ "$2" = reference ] && w=ref
  if [ "$2" = counterfactual ] && [ "$3" = 0 ]; then echo "$1-cf"; else echo "$1-$w-s$3"; fi
}

for r in named masked; do  # reuse the budget runs
  [ -d "runs/budget/$r-cf" ] && [ ! -d "$OUT/$r-cf" ] && cp -r "runs/budget/$r-cf" "$OUT/"
done

i=0; mine=()
if [ -n "${JOBS:-}" ]; then
  for j in $JOBS; do
    [ $((i % NSHARDS)) = "$SHARD" ] && mine+=("${j//:/ }")
    i=$((i + 1))
  done
else
  for s in $SEEDS; do
    for w in reference counterfactual; do
      for r in named masked; do
        [ $((i % NSHARDS)) = "$SHARD" ] && mine+=("$r $w $s")
        i=$((i + 1))
      done
    done
  done
fi

failed=0
for job in ${mine[@]+"${mine[@]}"}; do
  read -r r w s <<< "$job"
  id=$(run_id "$r" "$w" "$s")
  [ -f "$OUT/$id/result.json" ] && { echo "=== $id: done"; continue; }
  echo "=== train $id: $(date -u +%T)"
  resume=(); [ -f "$OUT/$id/state.json" ] && resume=(--resume)
  if ! iuw train "${COMMON[@]}" --rendering "$r" --world "$w" --seed "$s" --run-id "$id" ${resume[@]+"${resume[@]}"} \
      > "$OUT/$id.train.json" 2> "$OUT/$id.train.err"; then
    echo "train $id failed"; tail -5 "$OUT/$id.train.err"; failed=1
  fi
done

inspect() {  # run_id rendering mapping update
  local name="inspect-$1-u$(printf %03d "$4")"
  [ -f "$OUT/$name/inspection.json" ] && { echo "=== $name: done"; return; }
  [ -d "$CKPT/$1-u$(printf %03d "$4")" ] || { echo "=== $name: no checkpoint"; return; }
  echo "=== $name: $(date -u +%T)"
  python scripts/inspect_run.py --model "$MODEL" --adapter "$CKPT/$1-u$(printf %03d "$4")" --rendering "$2" \
    --mapping "$3" --seeds 4 --return-trip-seeds 1 --out "$OUT/$name" > "$OUT/$name.log" 2> "$OUT/$name.err" \
    || { echo "$name failed"; tail -5 "$OUT/$name.err"; failed=1; }
}

if [ "$BACKEND" = hf ] && [ "$INSPECT" != none ]; then
  if [ "$SHARD" = 0 ]; then
    inspect named-cf named 0 "$UPDATES"            # the matched pair first
    inspect masked-cf masked 0 $((UPDATES / 2))
    inspect masked-cf masked 0 "$UPDATES"
  fi
  if [ "$INSPECT" = all ]; then
    for job in ${mine[@]+"${mine[@]}"}; do
      read -r r w s <<< "$job"
      [ "$w" = counterfactual ] && [ "$s" != 0 ] && inspect "$(run_id "$r" "$w" "$s")" "$r" "$s" "$UPDATES"
    done
  fi
fi

iuw analyze --runs "$OUT" > "$OUT/analysis.json" 2> "$OUT/analysis.err"
python scripts/report.py --runs "$OUT" --out "$OUT/report.md" || failed=1
echo "=== shard $SHARD done $(date -u +%T) (failed=$failed)"
exit $failed
