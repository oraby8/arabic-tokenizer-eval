#!/bin/bash
# Phase 4b: 9 training runs queued over 4 GPUs (~8.5 h on 4x L40S with torch.compile).
cd "$(dirname "$0")"; [ -f ../env.sh ] && . ../env.sh; . "$B/.venv/bin/activate" 2>/dev/null || true
mkdir -p "$B/logs/lm"
q() {  # q GPU "tok mode micro" ...
  gpu=$1; shift
  for job in "$@"; do
    set -- $job
    CUDA_VISIBLE_DEVICES=$gpu python lm_train.py --tok $1 --mode $2 --micro $3 --compile > "$B/logs/lm/$2-$1.log" 2>&1 \
      || echo "FAILED $2-$1" >> "$B/logs/lm/failures.txt"
  done
}
q 0 "BPE scratch 16" "SPM-Unigram scratch 16" "BPE transplant 8" &
q 1 "MorphBPE scratch 16" "SPM-Unigram-morph scratch 16" "MorphBPE transplant 8" &
q 2 "SPM-Unigram-morph transplant 8" "SPM-Unigram transplant 8" &
q 3 "Qwen3 qwen 8" &
wait
echo LM_DONE >> "$B/logs/lm/status.txt"
