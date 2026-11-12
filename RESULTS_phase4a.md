# Phase 4a: morphology-constrained tokenizers (MorphBPE and SentencePiece Unigram + morph constraint)

Run date: 2026-11-11 · Code: `src/morph_segment.py`, `src/train_morph.py`, `src/train_spm.py`, `src/morph_eval.py`, `src/phase4_summary.py` · Outputs: `results/phase4/`

## What was built
Everything below uses **the same 1B-character training mix and the same vocabulary sizes (32K, 64K, 128K)** as the Phase 2 reference tokenizers. The only change is a morphological constraint at training time.

1. **Morpheme boundaries** come from CAMeL Tools' context-free (MLE) disambiguators. Each word type that appears at least twice is segmented once.

   | Text in the training mix | Analyzer |
   |---|---|
   | MSA, Najdi and Moroccan lines | `calima-msa-r13` |
   | Egyptian lines | `calima-egy-r13` (handles dialect morphology, e.g. م+ب+ي+عرف+ش) |

   Two granularities:
   - **Clitic** (CAMeL `d3seg`): splits و ف ب ل ك س ال and pronoun suffixes.
   - **Morpheme** (CAMeL `bwtok`): also splits inflectional affixes (ي+كتب+ون, مدرس+ة). This is closer to the Farasa segmentation that Fanar's MorphBPE used. `bwtok` normalizes spelling, so its boundaries are mapped back onto the original word by character alignment.

   Coverage: 54% of running MSA words and 50% of Egyptian words get at least one morpheme boundary, and 40% / 35% get a clitic boundary.
2. **MorphBPE** is byte-level BPE in which no merge may cross a morpheme boundary; it is our own implementation of Algorithm 1 of the Fanar report.
   - Neither public repository was usable: `llm-lab-org/MorphBPE` is empty, and `h9-tec/MorphBPE` is an unlicensed toy implementation that calls the Farasa web API.
   - Boundaries are enforced by a training-time pre-tokenizer split at the markers. It is removed after training, so the result is a standard BPE tokenizer that runs on plain text, as in the paper.
3. **SentencePiece Unigram + morph constraint** uses real SentencePiece 0.2.3 with `--pretokenization_delimiter`, so no piece can span a boundary.
   - The baseline is plain SentencePiece Unigram with identical settings (normalization `identity`, byte fallback, character coverage 0.9999).
   - The earlier Hugging Face Unigram references used a different trainer, so they aren't used as the baseline.
4. **Replication:** the same constraint was also trained with the Hugging Face `tokenizers` Unigram trainer. Results are in the appendix.

## Results (each constrained tokenizer compared with its unconstrained twin)

Column definitions:
- **Fertility:** tokens per word on FLORES MSA.
- **Chars/token:** web MSA, and absolute dialect text (mean over FineWeb, Wikipedia and PADIC).
- **Penalty:** dialect vs MSA, per character, as in Phase 2.
- **F1:** agreement between token boundaries inside words and CAMeL boundaries, on held-out evaluation text.

| Vocab | Algorithm | Constraint | Fertility MSA | Δ vs twin | Chars/token MSA web | Chars/token dialect | Penalty all | Penalty PADIC | F1 MSA morph | F1 MSA clitic | F1 Egyptian morph |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 32K | BPE | none | 1.26 | – | 4.10 | 3.27 | 1.163 | 1.060 | 0.10 | 0.07 | 0.16 |
| 32K | BPE | morpheme | 1.64 | **+30%** | 3.13 | 2.94 | 1.035 | 0.988 | 0.66 | 0.51 | 0.62 |
| 32K | SPM-Unigram | none | 1.24 | – | 4.09 | 3.27 | 1.175 | 1.079 | 0.27 | 0.22 | 0.36 |
| 32K | SPM-Unigram | morpheme | 1.92 | **+54%** | 2.69 | 2.42 | 1.072 | 1.029 | **0.96** | 0.82 | **0.90** |
| 64K | BPE | none | 1.16 | – | 4.44 | 3.51 | 1.166 | 1.060 | 0.07 | 0.05 | 0.12 |
| 64K | BPE | morpheme | 1.56 | **+34%** | 3.29 | 3.13 | 1.019 | 0.969 | 0.62 | 0.48 | 0.57 |
| 64K | BPE | clitic | 1.42 | +22% | 3.59 | 3.26 | 1.067 | 1.011 | 0.43 | 0.54 | 0.39 |
| 64K | SPM-Unigram | none | 1.15 | – | 4.44 | 3.52 | 1.175 | 1.072 | 0.19 | 0.16 | 0.27 |
| 64K | SPM-Unigram | morpheme | 1.88 | **+64%** | 2.75 | 2.52 | 1.054 | 1.007 | **0.96** | 0.82 | **0.90** |
| 64K | SPM-Unigram | clitic | 1.54 | +35% | 3.27 | 2.90 | 1.099 | 1.045 | 0.75 | **0.97** | 0.69 |
| 128K | BPE | none | 1.10 | – | 4.69 | 3.75 | 1.157 | 1.052 | 0.04 | 0.03 | 0.08 |
| 128K | BPE | morpheme | 1.47 | **+34%** | 3.49 | 3.33 | 1.015 | 0.961 | 0.55 | 0.43 | 0.49 |
| 128K | SPM-Unigram | none | 1.08 | – | 4.71 | 3.76 | 1.166 | 1.058 | 0.11 | 0.10 | 0.18 |
| 128K | SPM-Unigram | morpheme | 1.85 | **+71%** | 2.79 | 2.61 | 1.036 | 0.988 | **0.96** | 0.82 | 0.89 |

For comparison, the real LLM tokenizers score an MSA morpheme F1 of 0.09 (ALLaM) to 0.24 (Llama-3.1). Gemma-3 scores 0.21, Qwen3.5 0.18 and Karnak 0.23.

## What this shows

1. **The morphology constraint costs a lot of tokens:** +30–34% for MorphBPE and +54–71% for SentencePiece Unigram at morpheme granularity, at every vocabulary size.
   - Clitic-only constraints are cheaper (+22% BPE, +35% Unigram).
   - This is much more than the "slight increase in fertility" reported for MorphBPE. A likely reason is granularity: our morpheme-level segmentation also splits inflectional affixes, while the clitic-level version is closer to "slight".
   - **A bigger vocabulary doesn't recover the cost.** The SentencePiece morph penalty grows from +54% to +71% as the vocabulary goes from 32K to 128K. Whole words, the main source of extra compression at larger vocabularies, are exactly what the constraint forbids.
2. **The dialect penalty shrinks, but dialects don't get cheaper.**
   - The penalty falls from about 1.17 to about 1.02 for BPE (morpheme level), and PADIC even drops below 1.0.
   - The cause is that MSA gets much more expensive (−26% chars/token), while dialect text loses less (−11%).
   - The reason is uneven coverage: the analyzers segment Levantine, Tunisian, Najdi and most Maghrebi words poorly, so **the constraint binds mainly on MSA**.
   - **In absolute terms every constrained tokenizer is less efficient on dialects** than its twin (−7% to −31%).
3. **SentencePiece enforces the constraint at inference; MorphBPE largely doesn't.**
   - SentencePiece-morph token boundaries match morpheme boundaries 96% of the time on MSA, and 90% on Egyptian.
   - MorphBPE reaches only 0.55–0.66, and that **drops as the vocabulary grows** (0.66 → 0.55). MorphBPE forbids cross-boundary merges only during *training*. At inference the learned merges apply greedily to whole words, and a pair such as ال+م is a legitimate merge *inside* other words (العالم), so it fires across the boundary in المكتبة as well.
   - The more merges the vocabulary has, the more often this happens. A drop-in MorphBPE is therefore only partly morphological.
4. **Plain Unigram is more morphological than plain BPE for free.** Its F1 is 0.11–0.27, against 0.04–0.10 for BPE. This matches the article's point that Unigram gets some of the morphological benefit without any morphology data, although the size of the effect here is much smaller.
5. **Plain BPE is the least morphological of everything measured.** It is less aligned even than the real LLM tokenizers, because it is also the most compressed: larger tokens span more boundaries.

## Caveats
- **The gold boundaries come from the same segmenter used to build the constraints.** F1 therefore measures whether the constraint holds at inference, not linguistic correctness. The morph tokenizers have a built-in advantage on this metric.
- **The segmenters are weak on dialects.** Only Egyptian has a dialect analyzer; Gulf and Moroccan lines used the MSA analyzer. A better dialect segmenter would change the dialect results most.
- **Granularity matters,** and the Fanar paper's exact segmentation (Farasa) was not available offline at this scale.
- Fertility and compression are cost measures. **Whether the constraint helps the model is tested in Phase 4b** (bits-per-char and Belebele).

## Appendix: Hugging Face `tokenizers` Unigram with the same constraint
Here is a replication with a second trainer, at 64K:

| Tokenizer | Fertility MSA | Chars/token | F1 MSA morph | F1 Egyptian morph |
|---|---|---|---|---|
| HF Unigram (no constraint) | 1.20 | 4.26 | 0.24 | 0.31 |
| HF Unigram + morpheme | 1.83 | 2.80 | 0.94 | 0.88 |
| SPM Unigram + morpheme | 1.88 | 2.73 | 0.96 | 0.90 |

The two trainers agree closely, so the SentencePiece result isn't a quirk of one implementation.
