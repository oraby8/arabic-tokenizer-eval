#!/bin/bash
# Phase 4b compute-matched controls: each scratch model retrained with its twin's step count.
cd "$(dirname "$0")"; [ -f ../env.sh ] && . ../env.sh; . "$B/.venv/bin/activate" 2>/dev/null || true
mkdir -p "$B/logs/lm" "$B/logs/lm_eval"
job() { # job GPU TOK STEPS
  r=scratch-$2-steps$3
  CUDA_VISIBLE_DEVICES=$1 python lm_train.py --tok $2 --mode scratch --micro 16 --compile --steps $3 --tag steps$3 > "$B/logs/lm/$r.log" 2>&1 \
    && CUDA_VISIBLE_DEVICES=$1 python lm_eval.py --run $r > "$B/logs/lm_eval/$r.log" 2>&1 \
    || echo "FAILED $r" >> "$B/logs/lm/ctrl_failures.txt"
}
job 0 SPM-Unigram-morph 1816 &   # morph model at the plain model's compute (sees 64% of the data)
job 1 SPM-Unigram 2825 &         # plain model at the morph model's compute (1.56 epochs)
job 2 MorphBPE 1807 &
job 3 BPE 2285 &
wait; echo CTRL_DONE >> "$B/logs/lm/status.txt"
