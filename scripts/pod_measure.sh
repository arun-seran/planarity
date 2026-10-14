#!/usr/bin/env bash
# Throughput and memory on the main model, to price a Runpod run against Tinker.
# Writes everything under runs/measure/ (copy it back with rsync). Run detached:
#   nohup bash scripts/pod_measure.sh > runs/measure.log 2>&1 &
set -uo pipefail
export HF_HOME=${HF_HOME:-/root/hf}
source /root/venv/bin/activate
MODEL=${MODEL:-Qwen/Qwen3.6-35B-A3B}
OUT=runs/measure; mkdir -p "$OUT"
gpu() { nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader; }
step() {  # name, command...
  local name=$1; shift
  echo "=== $name: $(date -u +%FT%TZ)"
  local t0=$SECONDS
  "$@" > "$OUT/$name.json" 2> "$OUT/$name.err"; local rc=$?
  echo "=== $name: exit $rc after $((SECONDS - t0))s; GPU now $(gpu)"
  tail -3 "$OUT/$name.err"
}
step smoke iuw smoke --backend hf --model "$MODEL" --episodes 8
step train-attn iuw train --backend hf --model "$MODEL" --updates 2 --seeds-per-update 2 --eval-seeds 1 \
  --run-id measure-attn --out "$OUT/runs"
step train-experts iuw train --backend hf --model "$MODEL" --lora-experts --updates 2 --seeds-per-update 2 \
  --eval-seeds 1 --run-id measure-experts --out "$OUT/runs"
echo "=== done $(date -u +%FT%TZ)"
