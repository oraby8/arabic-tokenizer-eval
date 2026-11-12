#!/bin/bash
# Phase 2: natural corpora, ALDi-filtered training mix, 8 reference tokenizers, counting, analysis.
# Needs 1+ GPU for ALDi scoring (aldi_train.py is sharded over NGPU GPUs).
set -e
cd "$(dirname "$0")"
[ -f ../env.sh ] && . ../env.sh
: "${B:?set B to the repo root (see env.sh.example)}"
NGPU=${NGPU:-4}
mkdir -p "$B/logs"
python download_phase2.py
python prep_natural.py
for c in fineweb wiki padic; do CORPUS=$c CUDA_VISIBLE_DEVICES=0 python aldi.py; done
for i in $(seq 0 $((NGPU - 1))); do
  SHARD=$i NSHARDS=$NGPU CUDA_VISIBLE_DEVICES=$i python aldi_train.py > "$B/logs/aldi_train_$i.log" 2>&1 &
done
wait
python train_reference.py --data-only > "$B/logs/train_data.log" 2>&1
for j in bpe-mix-32k bpe-mix-64k bpe-mix-128k unigram-mix-32k unigram-mix-64k unigram-mix-128k bpe-msa-64k unigram-msa-64k; do
  [ -f "$B/tokenizers/$j.json" ] || RAYON_NUM_THREADS=12 python train_reference.py --job $j > "$B/logs/train_$j.log" 2>&1 &
done
wait
for c in flores fineweb wiki padic; do
  python count_tokens.py --corpus $c --workers 32 > "$B/logs/count_$c.log" 2> "$B/logs/count_$c.err"
done
for t in core all; do
  TOKSET=$t python analyze.py > "$B/logs/analyze_$t.log" 2>&1
  TOKSET=$t python analyze_natural.py > "$B/logs/analyze_natural_$t.log" 2>&1
done
echo PHASE2_DONE
