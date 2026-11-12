#!/bin/bash
# Train the 7 SentencePiece models in two waves (memory), 24 threads each.
cd "$(dirname "$0")"; [ -f ../env.sh ] && . ../env.sh; . "$B/.venv/bin/activate" 2>/dev/null || true
for j in plain-64k morpheme-64k plain-32k morpheme-32k; do
  /usr/bin/time -f "%e s wall, %M KB" python train_spm.py --job $j > "$B/logs/spm_$j.log" 2>&1 &
done; wait
for j in plain-128k morpheme-128k clitic-64k; do
  /usr/bin/time -f "%e s wall, %M KB" python train_spm.py --job $j --threads 32 > "$B/logs/spm_$j.log" 2>&1 &
done; wait
echo SPM_DONE
