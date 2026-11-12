#!/bin/bash
# Phase 4b evaluation queue: raw Qwen3 first (builds the shared BPC document mask), then each
# trained model as soon as its train_log.json exists, on the GPU with the most free memory.
cd "$(dirname "$0")"; [ -f ../env.sh ] && . ../env.sh; . "$B/.venv/bin/activate" 2>/dev/null || true
mkdir -p "$B/logs/lm_eval"
gpu() { nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1; }
[ -f "$B/results/phase4/lm/base-Qwen3/belebele.csv" ] || CUDA_VISIBLE_DEVICES=$(gpu) python lm_eval.py --base > "$B/logs/lm_eval/base-Qwen3.log" 2>&1
RUNS="scratch-BPE scratch-MorphBPE scratch-SPM-Unigram scratch-SPM-Unigram-morph transplant-BPE transplant-MorphBPE transplant-SPM-Unigram transplant-SPM-Unigram-morph qwen-Qwen3"
left="$RUNS"
while [ -n "$left" ]; do
  next=""
  for r in $left; do
    if [ -f "$B/results/phase4/lm/$r/belebele.csv" ]; then continue; fi
    if [ -f "$B/lm_runs/$r/train_log.json" ]; then
      CUDA_VISIBLE_DEVICES=$(gpu) python lm_eval.py --run $r > "$B/logs/lm_eval/$r.log" 2>&1 || echo "EVAL FAILED $r" >> "$B/logs/lm_eval/failures.txt"
    else next="$next $r"; fi
  done
  left="$next"; [ -n "$left" ] && sleep 120
done
echo EVAL_DONE >> "$B/logs/lm_eval/status.txt"
