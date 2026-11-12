# How Much Does Arabic Cost? A Fair Comparison of 24 LLM Tokenizers on MSA and Dialects

In my last article, I reviewed five recent papers on Arabic tokenization and ended with an open question:
https://www.linkedin.com/pulse/why-arabic-hard-tokenizers-three-core-challenges-ahmed-oraby-vidde?trk=public_post_feed-article-content

**How efficient are today's popular LLM tokenizers on MSA and dialects, compared fairly?**

So I measured it. I took 24 models (Llama 3 and 4, Qwen 2.5/3/3.5, Gemma 3 and 4, DeepSeek-V3, Aya/Command, gpt-oss, Jais, ALLaM, Fanar, Falcon-H1, Karnak, Atlas-Chat, Nile-Chat) and ran them on MSA and up to 13 Arabic dialects. I also trained my own reference tokenizers, to separate what comes from the vocabulary size from what comes from the training data.

Everything is open source: code, results, and the trained tokenizers.
👉 https://github.com/oraby8/arabic-tokenizer-eval

## What "fair" means here

Tokenizer comparisons are easy to get wrong:

- **Different texts:** if each model is tested on different text, the content gets measured, not the tokenizer.
- **Different word definitions:** "tokens per word" depends on how a word is defined.
- **Special tokens:** BOS/EOS markers and chat templates inflate counts.
- **Dataset labels:** a set labeled "dialect" isn't necessarily dialect.

So I used:

- **Parallel corpora:** the same meaning in every variety (FLORES+ and PADIC).
- **One preprocessing pipeline and one word splitter** for every tokenizer.
- **Plain token counts:** no special tokens and no chat templates.
- **Dialectness checked, not assumed:** every document was scored with ALDi, a model that rates how dialectal a sentence is.
- **Confidence intervals and paired tests** on every token-count comparison.

Sections 1–8 are about cost and efficiency: how many tokens it takes to say the same thing. Section 9 adds small language-model experiments that test whether a tokenizer change also helps the model.

## 1. Arabic still costs more than English, for most models

On the same FLORES+ sentences, this is how many Arabic (MSA) tokens each tokenizer needs per English token:

![Arabic vs English token cost for 15 tokenizers](https://raw.githubusercontent.com/oraby8/arabic-tokenizer-eval/main/results/blog_figures/1_arabic_vs_english.png)

In practice, a 128K context holds about **104K Arabic words with ALLaM but only 66K with Llama-3**. Relative to English, Llama-3, DeepSeek-V3 and Qwen2.5/3 are no better than raw UTF-8 bytes: Arabic costs them as much extra as it costs a byte-level model (1.6× vs 1.58×). In absolute terms they are of course far better than bytes (about 2 tokens per MSA word instead of 9).

**Lesson:** for Arabic, the tokenizer can decide whether you get 1.6× more context and cost per word.

## 2. Vocabulary size doesn't matter; Arabic tokens do

Does a bigger vocabulary help? Barely:

- **Vocabulary size:** almost no correlation with efficiency (ρ = −0.28).
- **Number of Arabic tokens in the vocabulary:** almost perfect correlation (ρ = 0.96, over the 15 distinct real tokenizers).

![Number of Arabic tokens in the vocabulary vs MSA characters per token](https://raw.githubusercontent.com/oraby8/arabic-tokenizer-eval/main/results/blog_figures/2_arabic_tokens_vs_efficiency.png)

For example:

- **Llama-4** has a 201K vocabulary, but only 2.2% of it is Arabic. It is barely better than Llama-3.
- **Jais-13B** (85K) beats almost every 250K vocabulary.

This is a correlation, not a cause. The Arabic token count is most likely a proxy for how much Arabic the tokenizer was trained on. Karnak shows the difference: it has the most Arabic tokens of all (about 45K, added on top of Qwen3's vocabulary), yet it is clearly less efficient (3.68 characters per token) than ALLaM (about 37K, 4.19) and Jais (about 32K, 4.18), which were trained on Arabic from the start.

**Lesson:** "large multilingual vocabulary" says nothing about Arabic. Count the Arabic tokens, and remember that adding Arabic tokens to an existing vocabulary isn't the same as training it on Arabic.

## 3. Many "Arabic models" don't have an Arabic tokenizer

This surprised me. Several Arabic-focused models reuse a general tokenizer **unchanged**:

| Model | Tokenizer |
|---|---|
| Fanar-2, Nile-Chat (Egyptian) | Gemma-3 |
| Atlas-Chat (Moroccan) | Gemma-2 |
| Command-R7B-Arabic | the same as Command-R / Aya |

So the Egyptian and Moroccan models get **no tokenizer advantage on the dialect they target**.

The positive exception is **Karnak**. It extends Qwen3's vocabulary with Arabic tokens, which makes Arabic **39% more compact** than with plain Qwen3.

**Lesson:** "Arabic model" and "Arabic tokenizer" are separate claims. Check the tokenizer.

## 4. Every tokenizer pays a dialect penalty, and the Maghreb pays the most

Dialect text always costs more tokens per character than MSA. How much depends heavily on the text:

- **On FLORES+ (translated sentences), the penalty is small:** 3–8% on average for the global tokenizers and 4–9% for the Arabic-centric ones.
- **On natural dialect text** (web, Wikipedia, PADIC conversations), it is much larger: 24–28% for ALLaM and Jais-adapted.
- **Translated "dialect" understates the real cost.** The FLORES+ Najdi set scores as MSA (penalty 1.003 for every tokenizer), and the other dialect translations sit close to MSA too.
- **Moroccan is the worst dialect for every tokenizer** on FLORES+ (7–18% more tokens per character).
- **Algiers and Annaba are the worst** in the parallel conversational data (PADIC).
- **The MSA-focused Arabic tokenizers (ALLaM, Jais) lose the most on dialects.** They are still the most efficient on dialect text in absolute terms.

**Lesson:** a tokenizer that is excellent on MSA can be noticeably worse on dialects. Evaluate both.

## 5. The controlled experiment: data beats vocabulary size

To find the cause, I trained my own BPE and Unigram tokenizers. Everything is identical (1B training characters, same algorithm, same vocabulary size) except **whether half the training text is dialect**:

![Dialect penalty for tokenizers trained on MSA only vs MSA plus dialect](https://raw.githubusercontent.com/oraby8/arabic-tokenizer-eval/main/results/blog_figures/3_dialect_training_experiment.png)

On natural text, the dialect penalty **drops by about half** (BPE: +31% → +17%), and MSA efficiency drops by only 2.6–4.2%. The improvement also **carries over to dialects that were never in the training data** (Levantine, Mesopotamian, Tunisian).

Making the vocabulary bigger doesn't do this. From 32K to 128K, MSA efficiency rises a lot, but the dialect penalty stays the same (+16%).

**Lesson:** if you want a tokenizer that handles dialects, put dialect text in the tokenizer's training data. A bigger vocabulary won't fix it.

## 6. Normalization: AraToken was right, but only halfway

AraToken reported that normalization (unifying Alif variants, removing tatweel, mapping digits) reduces fertility. I tested two settings:

- **Normalizing only the input to an existing tokenizer:** tokens usually go **up**. Light normalization adds up to +2.6% on MSA, and aggressive normalization (including ة→ه) adds up to +12%.

  The vocabularies learned "إلى" and "مدرسة" as single tokens, so "الى" and "مدرسه" split into more pieces.
- **Training the tokenizer on normalized text too:** tokens go **down**, but only by **1.3–2.7%**.

Aggressive normalization saves less than 1 point more than light normalization, while erasing real distinctions such as ة/ه and the hamza seats.

**Lesson:** normalize at training time or not at all, and prefer light over aggressive normalization. The efficiency gain is small; the information loss is real.

## 7. Two practical surprises: tweets and diacritics

**Tweets:**

- **URLs alone take 12–18% of the tokens.**
- Removing URLs, mentions, hashtag marks and emoji saves **21–34%**.
- Elongation (جمييييل) appears in 20% of tweets but costs only about 1% of tokens.

**Diacritics multiply the token count by 2–5×** for the models tested, and up to 8× for tokenizers trained without diacritics:

![Extra tokens for fully diacritized text, per tokenizer](https://raw.githubusercontent.com/oraby8/arabic-tokenizer-eval/main/results/blog_figures/4_diacritics_cost.png)

- **Qwen3.5** handles them best: +108%, more than twice as good as Qwen2.5/3 (+239%).
- **Jais-adapted (+377%) and Karnak (+344%)** are among the worst.
- Unicode NFC reorders shadda and fatha, which costs some tokenizers up to 6 more points.

**Lesson:** if your use case involves diacritized text (Quran, poetry, education, TTS), test diacritized text specifically. MSA results won't tell you.

## 8. Warnings about the data

These cost me the most time, and they may save you time:

- **FineWeb-2's "Moroccan" subset (`ary_Arab`) is about 98% MSA news.** Its dialect labels can't be trusted without checking them.
- **The FLORES+ Najdi set reads as MSA** by the dialectness score.
- **Egyptian and Moroccan Wikipedia are about 50% bot-generated templates.** One tokenizer had memorized them, which made Egyptian look *cheaper* than MSA.
- **The Hugging Face repo named "MADAR"** (`IQSeC-Lab/MADAR`) is a malware dataset, not the Arabic dialect corpus.

**Lesson:** check dialectness and duplication before measuring anything.

## 9. Morphology-aware tokenizers: expensive, and most of the gain is extra compute

Two of the papers in my last article (Fanar's MorphBPE, and Unigram with morphological constraints) argue that tokens should respect Arabic morphology: و+ال+كتاب, not arbitrary chunks. So I built both, using the same 1B characters and the same vocabulary sizes as my reference tokenizers. The only change is that no token may cross a morpheme boundary. The boundaries came from CAMeL Tools, with its Egyptian analyzer for Egyptian text and its MSA analyzer for everything else.

**The cost is large.** At 64K:

- **MorphBPE** needs **34% more tokens per MSA word** than plain BPE.
- **Unigram with the morphology constraint** needs **64% more** than plain Unigram.
- **A bigger vocabulary doesn't win the tokens back.** At 128K the Unigram cost grows to +71%, because whole words are exactly what the constraint forbids.
- Clitic-only splitting (و ف ب ل ك س ال and pronoun suffixes) is cheaper: +22% for BPE and +35% for Unigram.

**MorphBPE is only partly morphological.** The constraint applies when the vocabulary is learned, not when text is tokenized. A merge like ال+م is legitimate inside العالم, so it also fires in المكتبة, right across the boundary.

| Tokenizer | How often token boundaries match morpheme boundaries (F1) |
|---|---|
| Unigram + constraint | 0.96 |
| MorphBPE | 0.62 |
| Real LLM tokenizers | 0.09–0.24 |

**The dialect penalty "shrinks", but it's an illusion.** MSA gets much more expensive, while dialects are hardly segmented by the analyzers. The ratio therefore improves, but **every constrained tokenizer is less efficient on dialect text in absolute terms.**

**Does it pay off in the model?** I trained language models on the same 1B characters with each tokenizer, in two setups:

- **From scratch:** a 135M-parameter model.
- **Transplant:** Qwen3-0.6B with its tokenizer swapped out.

I measured **bits per character** on 22 held-out MSA and dialect test sets. Bits per character doesn't depend on the tokenizer, so models with different tokenizers can be compared directly.

![Token cost vs bits-per-character gain of morphology constraints](https://raw.githubusercontent.com/oraby8/arabic-tokenizer-eval/main/results/blog_figures/5_morphology_cost_vs_payoff.png)

**At first sight, Unigram + constraint looks like a clear win.**

- **Equal characters:** it beat plain Unigram on 22 of 22 test sets from scratch (−4.5% bits per character), and on 21 of 22 after the transplant (−3.3%).
- **MorphBPE gained only about 1%** over BPE.

**But there's a catch: more tokens means more training.** Every model saw the same 1B characters. A tokenizer that splits that text into 56% more tokens also gives its model 56% more training steps.

So I ran a compute-matched control: each from-scratch model retrained with its twin's number of steps, in both directions. Training the plain model longer and training the constrained one shorter gave the same answer:

| Comparison (from scratch) | Equal characters | Equal compute |
|---|---|---|
| MorphBPE vs BPE | −1.0% bits per character, 22/22 sets | **+1.8% (worse), 0/22 sets** |
| Unigram + constraint vs Unigram | −4.5%, 22/22 sets | **−0.6% to −1.1%, 17–18/22 sets** |

- **MorphBPE's gain was entirely extra compute.** At the same compute, it is worse than plain BPE on every test set.
- **Unigram + constraint keeps a small real advantage,** about 1% at the same compute. That small gain costs 64% more tokens per word at inference.
- **Plain Unigram beat plain BPE** (−2.5% at roughly equal compute, since both produce almost the same number of tokens). It needs no morphology data at all.
- **A good tokenizer transplant matches continued training with fewer tokens.** Qwen3-0.6B with the plain Unigram tokenizer reached about the same bits per character (1.74 MSA, 2.21 dialects) as the same model trained on the same data with its own tokenizer (1.75, 2.19). The transplant needed 238M training tokens; the model with its own tokenizer needed 403M.

**Benchmarks barely move at this scale.**

| Model | Belebele accuracy (6 Arabic varieties, 25% = chance) |
|---|---|
| Transplanted Qwen3, Unigram + constraint | 33.6% |
| Transplanted Qwen3, plain BPE | 31.4% |
| Original Qwen3-0.6B | 31.9% |

The differences are within about 2 points, and the confidence interval is ±1.2 points.

**Lesson:** when you compare tokenizers by training models, match the compute, not just the data. Otherwise the tokenizer that produces more tokens wins simply by training longer. At equal compute, morphological constraints are a poor deal: MorphBPE is worse than BPE, and Unigram + constraint buys about 1% better modelling for up to 64% more tokens per word. The analyzers also still don't cover most dialects.

## What We Learned

1. **The number of Arabic tokens in the vocabulary predicts efficiency.** Total vocabulary size doesn't. It's a proxy for Arabic training data, not a recipe: Karnak added the most Arabic tokens and is still behind ALLaM and Jais.
2. **Check the tokenizer, not the model's name.** Many Arabic models reuse a general tokenizer.
3. **Dialect coverage comes from the training data**, not from adding vocabulary entries.
4. **Normalization is a training decision,** not a preprocessing trick, and its gain is small.
5. **Verify your data.** Dataset labels, translations and Wikipedias can all mislead.
6. **Morphology constraints mostly don't pay off at equal compute.** MorphBPE is worse than BPE, and Unigram + constraint gains about 1% for up to 64% more tokens per word. Plain Unigram beats plain BPE for free.

## Open Questions

- How do closed models (Claude, Gemini, newer ChatGPT models) compare? Their tokenizers can only be measured through their APIs.
- Would a real dialect morphological analyzer change the picture for dialects? Today only Egyptian has one in CAMeL Tools.
- Do the small bits-per-character gains grow or vanish at larger model scales, where benchmark differences become measurable?

Code, results and the trained tokenizers: https://github.com/oraby8/arabic-tokenizer-eval

If you work on Arabic tokenization or have dialect data, I'd love to hear from you.

#ArabicNLP #NLP #LLM #Tokenization #MachineLearning #AI #Research
