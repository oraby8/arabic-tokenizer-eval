# Phase 3: ablations (normalization, diacritics, social-media cleaning)

Run date: 2026-11-10 · Code: `src/prep_phase3.py`, `src/ablation.py`, `src/summarize_phase3.py`, `src/trained_norm.py` · Outputs: `results/phase3/`

Every number here compares **token counts on the same sentences, before and after a text transform**. "−2%" means the transform saves 2% of tokens. Paired bootstrap 95% CIs are about ±0.05 to ±0.1 percentage points for the large groups (`ablation_grouped.csv`). Claude and Gemini API counting was skipped as requested.

## Data
| Group | Sentences | Source |
|---|---|---|
| MSA | 12,153 | FLORES MSA, PADIC MSA, FineWeb-2 MSA (ALDi < 0.1) |
| Dialects | 55,246 | FLORES (8 dialects, Najdi excluded), PADIC (4), FineWeb-2 dialectal documents (ALDi ≥ 0.3) |
| Tweets | 15,548 | ArSarcasm-v2 (train + test), with dialect labels: MSA, Egyptian, Gulf, Levantine, Maghrebi |
| Diacritized | 2,500 | `arbml/arabic_text_diacritization` test split (Tashkeela, mostly Classical Arabic) |

The transforms:
- **Normalization** (applied to the main NFC + diacritics-removed text): alif (أ إ آ ٱ → ا), yaa (ى → ي), taa (ة → ه), tatweel (remove ـ), hamza (ؤ → و, ئ → ي), digits (٠-٩ → 0-9).
  - **Light** = alif + yaa + tatweel + digits, AraToken-style.
  - **Aggressive** = light + taa + hamza.
- **Social cleaning:** remove URLs and @mentions, strip # (underscores → spaces), remove emoji, shorten 3+ repeated letters to 2 (elongation). **clean_all** applies all of these plus tatweel removal.
- **Diacritics:** fully diacritized text, both as written and after Unicode NFC, compared with the same text without diacritics.

## 1. Normalizing only the input makes most tokenizers *worse*

Normalization applied before an existing tokenizer (% change in tokens):

| | Alif | Taa ة→ه | Light | Aggressive |
|---|---|---|---|---|
| MSA, real tokenizers (range) | −0.5 (Gemma) … +2.7 (Karnak) | +2.0 … +9.5 (Jais-adapted) | −0.5 (Gemma) … +2.6 (Falcon-H1) | +1.7 (Gemma) … +12.2 (Jais-adapted) |
| Dialects, real tokenizers (range) | −0.7 … +0.7 | +0.4 … +3.1 | −0.9 … +0.1 | −0.5 … +2.8 |

- **Why:** the vocabularies were learned from correctly spelled text, so "إلى", "أن" and "مدرسة" are already single tokens. Normalizing to "الى", "ان" and "مدرسه" produces spellings the vocabulary has seen less often.

  The more MSA-focused the vocabulary, the larger the loss. The worst are Jais-adapted and the MSA-only reference tokenizers (up to +17.9% under aggressive normalization).
- **On dialect text, light normalization saves a little** (−0.1% to −0.9%). Dialect writing already tends to drop hamza, so normalized forms are closer to how dialects are written.
- **Tatweel removal** always saves a little (−0.1% to −0.7%), but tatweel occurs in only about 5% of sentences. **Digit mapping** has almost no effect (it changes 1–2.5% of sentences).
- **Gemma-3/4 is the only tokenizer that gains from Alif unification**, and only by −0.5%.

## 2. Normalization helps only if the tokenizer is *trained* on normalized text, and even then only a little

This is a controlled test. The 64K reference tokenizers were retrained on the same mix corpus after normalizing it. Changes are relative to the original tokenizer on the original text:

| Tokenizer | Normalization | Setting | MSA | Dialects | Tweets |
|---|---|---|---|---|---|
| BPE-64K | light | inference only | +0.95% | −0.38% | −0.52% |
| BPE-64K | light | **trained + inference** | **−1.28%** | **−1.70%** | **−2.04%** |
| BPE-64K | aggressive | inference only | +7.84% | +1.59% | +1.80% |
| BPE-64K | aggressive | **trained + inference** | **−1.63%** | **−2.67%** | **−2.68%** |
| Unigram-64K | light | inference only | +1.66% | −0.25% | −0.36% |
| Unigram-64K | light | **trained + inference** | **−1.48%** | **−1.60%** | **−1.86%** |

- This **confirms AraToken's direction** (normalization reduces fertility), but the effect is **about 1.3–2.7%**, much smaller than the 8.5% the article reports from the paper's own like-for-like comparison.
- Aggressive normalization buys only 0.3–1.0 points more than light normalization, and it erases real distinctions: ة/ه marks gender and the possessive pronoun, and ؤ/ئ carry hamza.

  Combined with AraToken's finding that keeping hamza variants gave lower model loss, **the efficiency gain doesn't justify aggressive normalization.** Light normalization at both training and inference is the defensible option, and it is worth only about 1.5%.

## 3. Cleaning social-media text: URLs and handles are the main cost, elongation hardly matters

Tweets, % change in tokens:

| | URLs | Mentions | Hashtag marks | Emoji | Elongation | clean_all |
|---|---|---|---|---|---|---|
| Tweets affected | 52% | 39% | 56% | 34% | 20% | 83% |
| Real tokenizers (range) | −11.7 (Llama-4/DeepSeek) … −18.4 (ALLaM) | −3.2 … −5.5 | −4.1 … −7.6 | −1.4 … −5.3 | −0.6 … −1.6 | −21.0 (Qwen2.5) … −34.4 (ALLaM) |

- **URLs alone take 12–18% of all tweet tokens.** A `t.co` link is a random string, so every tokenizer spends many tokens on it.
- **Arabic-centric tokenizers gain the most from cleaning** (ALLaM −34%, Jais-adapted −32%). Their small vocabularies have few Latin and emoji tokens, so non-Arabic material is expensive for them.
- **Elongation** (جمييييل, هههههه) appears in 20% of tweets but costs only 0.6–1.6% of tokens.
- These transforms mostly **remove content**; they aren't lossless compression. Whether to drop URLs and handles is a data-policy decision, not a tokenizer property.

## 4. Diacritics multiply the token count by 2–8×

Fully diacritized text vs the same text without diacritics, % more tokens:

| Tokenizer | As written | After NFC | | Tokenizer | As written | After NFC |
|---|---|---|---|---|---|---|
| **Qwen3.5** | **+108%** | +108% | | Atlas-Chat / Fanar-1 | +208% | +209% |
| Llama-4 | +115% | +119% | | Jais-3-0.9B | +226% | +226% |
| ALLaM-7B | +129% | **+135%** | | Qwen2.5/3 | +239% | +239% |
| DeepSeek-V3 | +143% | +143% | | Aya / Command | +249% | +249% |
| Gemma-3/4 | +145% | +147% | | Karnak | +344% | +344% |
| Llama-3.x | +147% | +148% | | Jais-adapted | +377% | +377% |
| Falcon-H1 | +180% | +183% | | Reference tokenizers (trained without diacritics) | +343% … +709% | same |
| gpt-oss | +195% | +199% | | | | |

- **Qwen3.5 handles diacritics best** (+108%), and it improved sharply over Qwen2.5/3 (+239%). Its vocabulary evidently added diacritized forms.
- **Arabic-centric vocabularies differ widely.** ALLaM is good (+129%), while Jais-adapted (+377%) and Karnak (+344%) are among the worst. If a use case involves diacritized text (Quran, poetry, education, TTS front-ends), this matters more than any MSA efficiency difference.
- **NFC reorders stacked marks** (it puts fatha U+064E before shadda U+0651, the reverse of how text is normally typed). This changed 90% of the diacritized lines.
  - Tokenizers that apply NFC internally are unaffected: Qwen, Aya/Command, Karnak. The reference tokenizers are unaffected too, because they never merge diacritics.
  - Tokenizers without that internal NFC step lose up to **+6.3 points** (ALLaM). Llama-4, gpt-oss and Falcon-H1 lose about 3 points.
  - **Practical rule:** if a pipeline applies NFC, check that it matches what the tokenizer saw in training.
- The reference tokenizers do badly here because they were trained on text with diacritics removed (the main condition). It shows why the diacritics policy at training time must match the intended use.

## Caveats
- The diacritized set is **mostly Classical Arabic (fiqh texts)**. Fully diacritized modern MSA may give different absolute numbers, although the ranking should be similar.
- The ArSarcasm-v2 dialect labels are coarse: about 70% of tweets are labeled "MSA", although many of those are Gulf dialect.
- **These are efficiency effects only.** Whether a normalization helps or hurts *model quality* (AraToken's hamza result) is not measured here.

## Files
- `results/phase3/norm_MSA.csv`, `norm_dialect.csv`, `social_tweets.csv`, `diacritics.csv`: the tables, for every tokenizer
- `results/phase3/ablation_grouped.csv`: everything with 95% CIs. `ablation_long.csv`: per individual set.
- `results/phase3/trained_normalization.csv`: section 2
- `results/phase3/share_of_sentences_changed.csv`: how often each transform applies
- `results/phase3/figures_ablation_heatmaps.png`, `figures_diacritics.png`
- Normalized-trained tokenizers: `tokenizers/{bpe,unigram}-mixlight-64k.json`, `tokenizers/bpe-mixaggressive-64k.json`
