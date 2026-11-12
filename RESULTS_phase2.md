# Phase 2: natural dialect text and reference tokenizers trained for this study

Run date: 2026-11-10 · Code: `src/` · Outputs: `results/phase2/core/`, plus `results/flores/core/`, which now includes the reference tokenizers

## What was added
1. **Natural text** (truncated to 2,000 characters per document; same preprocessing as Phase 1: NFC + diacritics removed):
   - **FineWeb-2 web text**, test split, 3,000 documents per variety: MSA, Egyptian, Moroccan, Najdi, Levantine, Mesopotamian, Tunisian.
   - **Wikipedia**, 3,000 articles each: Arabic, Egyptian, Moroccan.
   - **PADIC**, 7,213 parallel conversational sentences: MSA, Algiers, Annaba, Syrian, Palestinian. Converted from Buckwalter transliteration to Arabic script with CAMeL `bw2ar`.
2. **8 reference tokenizers** (Hugging Face `tokenizers`), trained on the FineWeb-2 *train* split, which doesn't overlap the evaluation documents:
   - Byte-level BPE and Unigram on **mix** (500M characters of MSA + 500M of dialect) at 32K, 64K and 128K vocabulary.
   - Byte-level BPE and Unigram on **msa** (1B characters of MSA) at 64K vocabulary.
   - Levantine, Mesopotamian and Tunisian were never used for training. They serve as **unseen dialects**.
3. **Vocabulary utilization:** the share of a tokenizer's Arabic-script tokens that actually appear on the web evaluation text.

## Data problems found and handled
1. **FineWeb-2's dialect labels are unreliable.**
   - The "Moroccan" (`ary_Arab`) subset is about 98% MSA news. Its mean ALDi is 0.03; only 2.2% of training documents and 73 of 3,000 test documents score ≥ 0.3.
   - Egyptian is about 72% MSA-like. Najdi, Levantine, Mesopotamian and Tunisian are mostly dialectal.
   - Fixes:
     - The dialect half of the training **mix** uses only documents with ALDi ≥ 0.3 (scored on 4 GPUs, about 10M sentences).
     - Evaluation uses only dialectal documents (ALDi ≥ 0.3) and MSA documents with ALDi < 0.1.
     - A first set of tokenizers trained on the raw labels was discarded (not included in the repo).
2. **Dialect Wikipedias are about 50% bot templates.** Egyptian Wikipedia contains thousands of near-identical galaxy stubs.
   - DeepSeek-V3 has memorized whole template words, for example " الشعاعيه". This made Egyptian Wikipedia look *cheaper* than MSA for it (penalty 0.82).
   - Fix: any document sharing an 8-word sequence with at least 1% of documents in its set is excluded. After filtering, DeepSeek's value is 1.13.
3. **Some evaluation subsets are small after filtering:**
   - FineWeb Moroccan: 73 documents
   - Egyptian Wikipedia: 198 documents

   Their confidence intervals are wider. Moroccan is better covered by Moroccan Wikipedia (1,267 documents) and by FLORES.

Evaluation subsets after filtering (`eval_subsets.csv`):

| Corpus | MSA | Dialect sets (documents, mean ALDi) |
|---|---|---|
| FineWeb-2 | 2,931 (0.007) | Egyptian 851 (0.53) · Najdi 1,890 (0.59) · Levantine 2,236 (0.62) · Mesopotamian 2,692 (0.72) · Tunisian 2,135 (0.58) · Moroccan 73 (0.51) |
| Wikipedia | 2,649 (0.016) | Egyptian 198 (0.40) · Moroccan 1,267 (0.51) |
| PADIC (parallel) | 7,213 (0.07) | Algiers (0.72) · Annaba (0.68) · Syrian (0.71) · Palestinian (0.67) |

## Main results

The tables use these column definitions:
- **Chars/token:** higher is more efficient.
- **Dialect penalty:** MSA chars/token ÷ dialect chars/token for the same tokenizer. 1.20 means dialect text needs 20% more tokens per character than MSA.
- **"All" penalty:** averaged over all 12 dialect sets in the three corpora.

### 1. Training on dialect text cuts the dialect penalty by about half, at a small cost on MSA (controlled experiment)
Same algorithm, same vocabulary size (64K), same total of 1B training characters. The only change is whether half the text is dialect:

| | BPE msa | BPE mix | Unigram msa | Unigram mix |
|---|---|---|---|---|
| MSA chars/token (web) | 4.55 | 4.44 (−2.6%) | 4.39 | 4.20 (−4.2%) |
| MSA chars/token (FLORES) | 4.54 | 4.41 (−3.0%) | 4.49 | 4.26 (−5.1%) |
| Dialect penalty, all sets | 1.31 | **1.17** | 1.42 | **1.19** |
| Dialect penalty, PADIC (parallel) | 1.20 | **1.06** | 1.29 | **1.08** |
| Dialect penalty, **unseen** dialects (web) | 1.41 | **1.26** | 1.54 | **1.28** |
| FLORES Moroccan penalty | 1.18 | **1.05** | 1.27 | **1.06** |
| Dialect chars/token (absolute, mean of corpora) | 3.18 | 3.51 (+10%) | 2.83 | 3.32 (+17%) |

- The 95% bootstrap CIs are about ±0.01 on every penalty, so all these differences are far outside noise.
- The gain carries over to dialects that were **never in the training data** (Levantine, Mesopotamian, Tunisian).

### 2. A larger vocabulary improves efficiency but doesn't reduce the dialect penalty
Byte-level BPE trained on the mix:

| Vocab size | MSA chars/token | Dialect penalty |
|---|---|---|
| 32K | 4.10 | 1.163 |
| 64K | 4.44 | 1.166 |
| 128K | 4.69 | 1.157 |

The same holds for Unigram. Getting dialects close to MSA efficiency depends on the *training data*, not on the vocabulary size.

### 3. BPE is more efficient than Unigram here
At every vocabulary size, BPE gets more characters per token than Unigram on both MSA and dialects (64K mix: 4.44 vs 4.20 on MSA, 3.51 vs 3.32 on dialects). Their penalties are similar (1.17 vs 1.19).

This contradicts the "Unigram is an underrated baseline" point from the article only on **compression**. The article's claim concerned morphological alignment, which this study doesn't measure.

### 4. The models you selected

| Tokenizer | MSA chars/token (web) | Dialect chars/token (abs.) | Penalty, all | Penalty, PADIC | Arabic tokens used |
|---|---|---|---|---|---|
| ALLaM-7B | 4.16 | **3.11** | 1.24 | 1.15 | 96% |
| Jais-adapted | 4.12 | 2.96 | 1.28 | 1.19 | 99% |
| Karnak (6B = 40B) | 3.74 | 2.89 | 1.21 | 1.13 | **69%** |
| Jais-3-0.9B | 3.72 | 2.78 | 1.24 | 1.16 | 94% |
| Falcon-H1 | 3.50 | 2.60 | 1.25 | 1.14 | 85% |
| Qwen3.5 | 3.34 | 2.52 | 1.23 | 1.14 | 89% |
| gpt-oss | 3.27 | 2.56 | 1.20 | 1.12 | 80% |
| Gemma-3 = Fanar-2 = Nile-Chat | 2.96 | 2.56 | **1.10** | **1.05** | 73% |
| Gemma-4 (≈ Gemma-3)¹ | 2.96 | 2.56 | 1.10 | 1.05 | 73% |
| Atlas-Chat (Gemma-2) | 2.91 | 2.47 | 1.13 | 1.09 | 80% |
| Fanar-1 | 2.91 | 2.46 | 1.13 | 1.09 | 96% |
| Aya-Expanse = Command-A/R/R7B-Arabic | 2.90 | 2.37 | 1.17 | 1.08 | 80% |
| Llama-4 | 2.78 | 2.30 | 1.14 | 1.08 | 77% |
| Qwen2.5 = Qwen3 | 2.72 | 2.24 | 1.17 | 1.12 | 91% |
| DeepSeek-V3 | 2.69 | 2.30 | 1.13 | 1.09 | 88% |
| Llama-3.x | 2.65 | 2.30 | 1.11 | 1.06 | 80% |

¹ Gemma-4 gives identical token ids to Gemma-3 on FLORES and PADIC, but different ids on some FineWeb documents. The efficiency is the same to 3 decimals, so it is listed separately but behaves the same.

What the table shows:
- **ALLaM and Jais-adapted are the most efficient on dialects in absolute terms**, but their penalty (1.24–1.28) is about the same as the MSA-only reference tokenizers (1.31). This suggests their vocabularies were built mostly from MSA.
- **Gemma-3/4 has the lowest dialect penalty** of the real tokenizers, but from a low starting point. It is about 30% less efficient than ALLaM on MSA.
- **Karnak doesn't use 31% of its Arabic tokens** on 25M characters of MSA and dialect web text. Those vocabulary entries contribute nothing on this text.

  ALLaM, Jais and Fanar-1 use at least 94%. This measurement depends on corpus size (25M characters), so read it as a relative comparison.
- **Maghrebi dialects are the worst** in the parallel PADIC data. Algiers and Annaba have penalties of 1.08–1.33, while Syrian and Palestinian are 0.99–1.10. This agrees with FLORES, where Moroccan was the worst.
- **Mesopotamian is the worst set on web text.** That ranking mixes dialect with *domain*, because forum text is compared against MSA news (see caveats).

### 5. RQ4 on FLORES, real LLM tokenizers only
- `analyze.py` now computes RQ4 on the Arabic-centric and global tokenizers only. An earlier version also included the reference tokenizers. They are 72–93% Arabic by design, which inflated the correlation (ρ = 0.975) and made vocabulary size look negatively related to efficiency (ρ = −0.47).
- **Core set (n = 15):** number of Arabic tokens ρ = 0.96; vocabulary size ρ = −0.28 (p = 0.31); R² = 0.95 for log(vocab) + log(number of Arabic tokens).
- **Extended set (n = 25):** number of Arabic tokens ρ = 0.96; vocabulary size ρ = +0.05 (p = 0.83).

## Caveats
- **The reference tokenizers are Arabic-only.** Their English fertility is 1.5–2.1, against about 1.05–1.2 for the real tokenizers. Their parity vs English is therefore meaningless, and it is left blank (NaN) in `summary.csv` and `parity_vs_english.csv`.

  They show the best efficiency each vocabulary size can reach for Arabic. A real bilingual tokenizer of the same size would score lower, so the comparison with ALLaM (64K, bilingual) at the same size is not like-for-like.
- **Web and Wikipedia penalties are not parallel.** MSA and dialect documents differ in topic and register, so those numbers combine dialect and domain effects. **PADIC and FLORES are the clean, same-meaning measurements.**
- **ALDi is itself a model.** Dialect selection inherits its errors, especially for Maghrebi dialects.
- **PADIC** is small, conversational, and comes from a transliterated source. The conversion back to Arabic script may not perfectly recover the original spelling.
- **MADAR was not used.** The official corpus requires a request to CAMeL Lab, and the Hugging Face repo named `IQSeC-Lab/MADAR` is an unrelated malware dataset.
- **QADI was not used**, because it contains tweet IDs only, without the tweet text.
- **Falcon-Arabic** has no public tokenizer.
- **This is still a cost and efficiency study, not a model-quality study.**

## Files
- `results/phase2/core/summary.csv`: the table above
- `results/phase2/core/dialect_penalty_all_corpora.csv`: tokenizer × dialect set, with 95% CIs in `metrics_long.csv`
- `results/phase2/core/eval_subsets.csv`: document counts, ALDi and template share for each evaluation subset
- `results/phase2/core/figures/msa_vs_dialect_efficiency.png`, `dialect_penalty_all_corpora.png`
- `results/flores/core/`: FLORES tables now include the 8 reference tokenizers
- Trained reference tokenizers: `tokenizers/*.json`
