# Phase 1: Arabic tokenizer efficiency on FLORES+ (MSA + 9 dialects)

Run date: 2026-11-10 · Code: `src/` · Outputs: `results/flores/core/` (selected models) and `results/flores/all/` (with extra reference tokenizers)

## Setup
- **Data:** FLORES+ dev + devtest. That is 2,009 parallel sentences for English, MSA and 9 dialects: Egyptian, Sudanese, North and South Levantine, Mesopotamian, Najdi, Ta'izzi-Adeni, Moroccan and Tunisian.
- **Preprocessing (identical for all tokenizers):** Unicode NFC, then diacritics removed with CAMeL `dediac_ar`.
- **Counting:** tokens are counted with `add_special_tokens=False` and no chat template.
- **Words:** one fixed splitter (CAMeL `simple_word_tokenize`) is used for every language.
- **Tokenizers:** 24 selected repos, which reduce to **15 distinct tokenizers** after deduplicating identical outputs. Aliases:
  - Llama-3.1 = Llama-3.2
  - Qwen2.5 = Qwen3
  - Gemma-3 = Gemma-4 = Fanar-2 = Nile-Chat
  - Aya-Expanse = Command-A = Command-R = Command-R7B-Arabic
  - Karnak-6B = Karnak-40B
- **Statistics:**
  - 95% CIs come from a paired bootstrap over sentences (2,000 resamples).
  - Wilcoxon signed-rank tests compare each dialect with MSA, per tokenizer.
- **Dialectness:** each set was scored with ALDi (Sentence-ALDi; 0 = MSA, 1 = strongly dialectal):

| Set | Najdi | Ta'izzi-Adeni | Mesopotamian | Tunisian | N. Levantine | S. Levantine | Sudanese | Moroccan | Egyptian |
|---|---|---|---|---|---|---|---|---|---|
| ALDi | 0.015 | 0.144 | 0.164 | 0.296 | 0.312 | 0.353 | 0.422 | 0.444 | 0.465 |

**Data warning:** the FLORES+ Najdi set is effectively MSA (ALDi 0.015). It is excluded from the dialect averages. Ta'izzi-Adeni and Mesopotamian are only weakly dialectal. FLORES dialect translations of formal news text are milder than real dialect text, so Phase 2 (MADAR plus natural dialect text) is needed before drawing firm conclusions about dialects.

## Main table (selected models, sorted by MSA parity vs English)

These columns describe each tokenizer:
- **Parity:** Arabic tokens ÷ English tokens for the same sentences. 1.0 means Arabic costs the same as English.
- **Dialect penalty (len-adj):** how much less efficient the tokenizer is on dialect text than on MSA, per character. It is averaged over 8 dialects, with Najdi excluded.

| Tokenizer | Vocab | % Arabic vocab | Fertility MSA | Chars/token MSA | Parity MSA [95% CI] | Dialect penalty (len-adj) | Worst dialect | Words in 128K ctx |
|---|---|---|---|---|---|---|---|---|
| Jais-adapted | 64K | 50.1 | 1.23 | 4.18 | 0.89 [0.88, 0.89] | +8.5% | Moroccan +17.8% | 104,300 |
| ALLaM-7B | 64K | 58.3 | 1.23 | 4.18 | 0.93 [0.93, 0.94] | +6.5% | Moroccan +15.9% | 104,500 |
| Jais-3-0.9B | 64K | 19.5 | 1.43 | 3.59 | 1.11 [1.10, 1.12] | +7.3% | Moroccan +14.1% | 89,700 |
| Karnak (6B = 40B) | 193K | 23.4 | 1.39 | 3.68 | 1.13 [1.12, 1.14] | +6.2% | Moroccan +12.3% | 91,900 |
| Falcon-H1 | 130K | 10.4 | 1.48 | 3.46 | 1.16 [1.15, 1.17] | +7.7% | Moroccan +14.5% | 86,300 |
| Qwen3.5 | 248K | 3.6 | 1.57 | 3.27 | 1.28 [1.27, 1.29] | +6.6% | Moroccan +12.5% | 81,700 |
| gpt-oss (o200k) | 200K | 4.0 | 1.61 | 3.18 | 1.35 [1.34, 1.36] | +5.9% | Moroccan +11.2% | 79,400 |
| Fanar-1 | 128K | 3.8 | 1.76 | 2.91 | 1.42 [1.41, 1.43] | +4.3% | Moroccan +8.2% | 72,600 |
| Aya-Expanse = Command-A = Command-R = Command-R7B-Arabic | 255K | 2.6 | 1.72 | 2.98 | 1.43 [1.42, 1.44] | +5.0% | Moroccan +8.9% | 74,400 |
| Gemma-3 = Gemma-4 = Fanar-2 = Nile-Chat | 262K | 3.2 | 1.74 | 2.95 | 1.43 [1.42, 1.44] | +3.0% | Moroccan +6.8% | 73,700 |
| Atlas-Chat (= Gemma-2) | 256K | 2.5 | 1.76 | 2.91 | 1.46 [1.45, 1.47] | +4.3% | Moroccan +8.2% | 72,600 |
| Llama-4 | 201K | 2.2 | 1.88 | 2.73 | 1.56 [1.55, 1.57] | +4.3% | Moroccan +8.0% | 68,100 |
| Qwen2.5 = Qwen3 | 152K | 2.7 | 1.93 | 2.65 | 1.57 [1.56, 1.58] | +5.1% | Moroccan +8.6% | 66,200 |
| DeepSeek-V3 | 129K | 2.5 | 1.91 | 2.68 | 1.59 [1.58, 1.60] | +4.1% | Moroccan +7.8% | 66,900 |
| Llama-3.1 = Llama-3.2 | 128K | 3.0 | 1.95 | 2.63 | 1.61 [1.60, 1.62] | +3.7% | Moroccan +7.0% | 65,700 |

Reference baselines for parity: characters 0.87, words 0.88, UTF-8 bytes 1.58. The same content takes about 13% fewer characters and words in Arabic than in English, so an "ideal" tokenizer would have parity below 1.0. Llama-3.1, DeepSeek-V3 and Qwen2.5/3 do **no better than raw UTF-8 bytes** relative to English.

## Answers to the research questions

**RQ1: Arabic vs English.**
- Only Jais-adapted and ALLaM reach **parity or better** (0.89–0.93).
- The other models need more tokens for Arabic than for English:

| Models | Arabic cost vs English |
|---|---|
| Jais-3-0.9B, Karnak, Falcon-H1 | 1.11–1.16× |
| Qwen3.5 | 1.28× |
| gpt-oss | 1.35× |
| Fanar-1, Aya/Command-A, Gemma-3/4 | 1.42–1.46× |
| Llama-4, Qwen2.5/3, DeepSeek-V3, Llama-3.x | 1.56–1.61× |

- In practice, a 128K context holds about 104K MSA words with ALLaM but 66K with Llama-3.x.

**RQ2: dialects vs MSA.**
- **Per character:** every tokenizer is less efficient on dialect text (+3.0% to +8.5% on average). **Moroccan is the worst dialect for all 15 tokenizers** (+6.8% to +17.8%). Tunisian and Sudanese are usually next.
- **Per sentence (same meaning, raw tokens):** the penalty is smaller, because dialect translations are 2–6% shorter in characters.
  - 104 of 120 paired Wilcoxon tests (dialect vs MSA, Najdi excluded) are significant at p < 0.001.
  - The 16 that are not are 13 Levantine, 2 Ta'izzi-Adeni and 1 Egyptian. Levantine sentences are short enough to cost about the same tokens as MSA.
  - No significance test was run on the length-adjusted (per-character) penalty.

**RQ3: Arabic-centric vs global.**
- **The models with dedicated Arabic vocabularies have the largest dialect penalties.**
  - Jais-adapted: +8.5% (Moroccan +17.8%)
  - ALLaM: +6.5%
  - Jais-3-0.9B: +7.3%
  - Gemma-3/4: only +3.0%
- Their MSA-trained vocabulary helps MSA more than it helps dialects. Even so, in absolute terms they remain the most efficient on every dialect.
- **Several "Arabic models" don't have their own Arabic tokenizer.** Fanar-2 and Nile-Chat use the Gemma-3 tokenizer unchanged, and Atlas-Chat uses the Gemma-2 tokenizer. Nile-Chat (Egyptian) and Atlas-Chat (Moroccan) therefore get no tokenizer benefit for the dialect they target.
- **Command-R7B-Arabic uses exactly the same tokenizer as Command-R, Command-A and Aya-Expanse.** Its Arabic specialization is in the model weights only, not the tokenizer.
- **Llama-4 is barely better than Llama-3** on Arabic (2.73 vs 2.63 chars/token), despite a 57% larger vocabulary. Only 2.2% of its vocabulary is Arabic.
- **Karnak extends Qwen3's vocabulary** from 152K to 193K entries, 23% of them Arabic. Its MSA efficiency improves from 2.65 to 3.68 chars/token (+39%).
- **Jais-3-0.9B uses a different tokenizer from Jais-3-375B** (64K vs 251K). The 375B tokenizer is more efficient (4.34 chars/token, from the `all` set).

**RQ4: what explains the differences?**
- **Vocabulary size does not** (Spearman ρ = −0.28, p = 0.31).
- **The number of Arabic tokens in the vocabulary does** (ρ = 0.96, p < 0.001).
- A linear fit, chars/token ≈ log(vocab) + log(number of Arabic tokens), gives R² = 0.95 (n = 15).
- The extended set (25 real tokenizers, `results/flores/all/rq4.txt`) gives the same picture: ρ = 0.96 for the number of Arabic tokens, and ρ = +0.05 for vocabulary size.
- Only the Arabic-centric and global tokenizers are used. The reference tokenizers trained in this study are mostly Arabic by design and would inflate the correlation.
- **This is a correlation, not a cause.** With n = 15, the number of Arabic tokens is probably a proxy for how much Arabic the tokenizer was trained on. Karnak has the most Arabic tokens (about 45K) but is clearly less efficient (3.68 chars/token) than ALLaM (about 37K, 4.19) and Jais-adapted (about 32K, 4.18). Extending a vocabulary isn't the same as training it on Arabic.

## Data-quality checks
- **Round-trip decoding:** every selected tokenizer decodes the text back exactly.
- **Byte fallback (letters encoded as raw bytes):**
  - Jais-adapted: 2,076 tokens
  - ALLaM: 38 tokens
  - Fanar-1 and Gemma-3: 2 each
  - Others: none

## Not covered yet
- **Falcon-Arabic has no public Hugging Face repo**, so it can't be tested locally.
- **"Used share" of the Arabic vocabulary** is not reported yet, because FLORES is too small for it. It will come with the Phase 2 natural corpora.
- **Parity is a cost and efficiency metric, not model quality.** See the "Morphemes without Borders" lesson.

## Figures (`results/flores/core/figures/`)
- `parity_heatmap.png`: parity for each tokenizer × variety
- `dialect_penalty_heatmap.png`: raw dialect penalty (tokens per sentence)
- `vocab_vs_efficiency.png`: chars/token against vocabulary size
