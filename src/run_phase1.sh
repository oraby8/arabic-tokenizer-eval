#!/bin/bash
# Phase 1: FLORES+ (MSA + 9 dialects + English) for every tokenizer in src/registry.py.
set -e
cd "$(dirname "$0")"
[ -f ../env.sh ] && . ../env.sh
: "${B:?set B to the repo root (see env.sh.example)}"
mkdir -p "$B/logs"
python download_flores.py
python prep_flores.py
CORPUS=flores CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python aldi.py
python count_tokens.py --corpus flores --workers 32 > "$B/logs/count_flores.log" 2> "$B/logs/count_flores.err"
for t in core all; do TOKSET=$t python analyze.py > "$B/logs/analyze_$t.log" 2>&1; done
echo PHASE1_DONE
