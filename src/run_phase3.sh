#!/bin/bash
# Phase 3: normalization / social-cleaning / diacritics ablations (requires Phase 2 outputs).
set -e
cd "$(dirname "$0")"
[ -f ../env.sh ] && . ../env.sh
: "${B:?set B to the repo root (see env.sh.example)}"
mkdir -p "$B/logs"
python prep_phase3.py
python ablation.py > "$B/logs/ablation.log" 2> "$B/logs/ablation.err"
python summarize_phase3.py
python trained_norm.py --data
for j in bpe-light bpe-aggressive unigram-light; do
  RAYON_NUM_THREADS=30 python trained_norm.py --job $j > "$B/logs/trained_norm_$j.log" 2>&1 &
done
wait
python trained_norm.py --eval
echo PHASE3_DONE
