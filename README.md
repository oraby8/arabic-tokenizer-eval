# How efficient are today's LLM tokenizers on Arabic, MSA and dialects?

A controlled comparison of **24 tokenizers from current LLMs** (Llama 3/4, Qwen 2.5/3/3.5, Gemma 3/4, DeepSeek-V3, Aya/Command, gpt-oss, Jais, ALLaM, Fanar, Falcon-H1, Karnak, Atlas-Chat, Nile-Chat, …). They are measured on Modern Standard Arabic and **up to 13 dialects**, with **reference tokenizers trained for this study** to separate the effect of vocabulary size from the effect of training data.

Phases 1–3 are a cost and efficiency study: how many tokens it takes to say the same thing. Phase 4 adds morphology-constrained tokenizers and small language-model experiments (bits per character, Belebele) to test whether a tokenizer change also helps the model.

| Report | What it covers |
|---|---|
| [BLOG_POST.md](BLOG_POST.md) | The write-up of all phases for a general audience |
| [RESULTS_phase1.md](RESULTS_phase1.md) | FLORES+ parallel sentences (English, MSA, 9 dialects): parity vs English, dialect penalty, what explains efficiency |
| [RESULTS_phase2.md](RESULTS_phase2.md) | Natural web, Wikipedia and conversational text. Reference BPE/Unigram tokenizers at 32K–128K, MSA-only vs MSA+dialect training |
| [RESULTS_phase3.md](RESULTS_phase3.md) | Ablations: orthographic normalization (at inference vs at training), social-media cleaning, diacritics |
| [RESULTS_phase4a.md](RESULTS_phase4a.md) | MorphBPE and SentencePiece Unigram with morphological constraints: fertility, dialect penalty, morpheme-boundary alignment |
| [RESULTS_phase4b.md](RESULTS_phase4b.md) | Language models (135M from scratch, Qwen3-0.6B transplants) with those tokenizers: bits per character, Belebele, compute-matched controls |

## Key findings
1. **Arabic costs 0.9–1.6× the tokens of English for the same content**, depending on the tokenizer. Jais-adapted and ALLaM reach parity. Relative to English, Llama-3.x, DeepSeek-V3 and Qwen2.5/3 are no better than raw UTF-8 bytes: Arabic costs them as much extra as it costs a byte-level model.
2. **The number of Arabic tokens in the vocabulary predicts efficiency** (ρ = 0.96 over 15 real tokenizers); vocabulary size does not. Llama-4 (201K vocab, 2.2% Arabic) is barely better than Llama-3. This is a correlation: Karnak has the most Arabic tokens but is less efficient than ALLaM and Jais, so extending a vocabulary isn't the same as training on Arabic.
3. **Several "Arabic" models reuse a general tokenizer unchanged.** Fanar-2 and Nile-Chat use Gemma-3's; Atlas-Chat uses Gemma-2's; Command-R7B-Arabic uses Command-R's. Karnak extends Qwen3's vocabulary, and that extension makes MSA 39% more compact.
4. **Every tokenizer pays a dialect penalty, and Maghrebi dialects are the worst.** On translated FLORES+ text it is small (3–9% on average, 7–18% for Moroccan), while on natural text it reaches 24–28% for ALLaM and Jais-adapted. Tokenizers built mainly on MSA (ALLaM, Jais) lose the most efficiency on dialects. They still beat the global tokenizers on dialect text in absolute terms.
5. **Training on dialect text halves the dialect penalty and costs only 3–4% on MSA.** This is a controlled experiment, and the gain carries over to dialects never seen in training. **Increasing the vocabulary size does not reduce the penalty.**
6. **Normalization only helps if the tokenizer is trained on normalized text, and then by only 1–3%.** Applied only at inference, it *adds* tokens (aggressive normalization adds up to +12%).
7. **Diacritics multiply token counts by 2–8×.** Qwen3.5 handles them best (+108%), while Jais-adapted (+377%) and Karnak (+344%) are among the worst. Unicode NFC reorders shadda/fatha, which costs tokenizers that don't normalize internally up to 6 more points.
8. **Morphology constraints are expensive and mostly don't pay off at equal compute.** They add 34% (MorphBPE) to 64% (Unigram) more tokens per MSA word. At equal compute, MorphBPE models text worse than BPE, and Unigram + constraint is only about 1% better than plain Unigram. Plain Unigram beats plain BPE by 2.5%.
9. **Data warnings for other researchers:**
   - FineWeb-2's `ary_Arab` ("Moroccan") subset is about 98% MSA news.
   - The FLORES+ Najdi set reads as MSA.
   - Egyptian and Moroccan Wikipedia are about 50% bot templates.
   - The Hugging Face repo `IQSeC-Lab/MADAR` is a malware dataset, not the Arabic MADAR corpus.

## Fairness controls
- **Same text for every tokenizer:** parallel corpora (FLORES+, PADIC) wherever possible, and identical preprocessing (NFC + diacritics removed with CAMeL `dediac_ar`).
- **Counting rules:** `add_special_tokens=False` and no chat templates.
- **Tokenizer-independent normalizers:** characters, UTF-8 bytes, and words from one fixed splitter (CAMeL `simple_word_tokenize`).
- **Deduplication:** tokenizers that produce identical token ids on every corpus are counted once.
- **Dialectness checked, not assumed:** ALDi scores filter which documents count as MSA and as dialect, and bot templates are removed.
- **Statistics:** paired bootstrap 95% CIs over sentences, and Wilcoxon tests on parallel data.
- **Reference tokenizers** are trained on the FineWeb-2 *train* split, which is disjoint from the evaluation documents. Three dialects are held out of training entirely.

## Repository layout
```
src/
  registry.py              tokenizers (Hugging Face repo ids) and FLORES varieties
  textnorm.py              shared normalization and word spans
  download_flores.py       Phase 1 data          download_phase2.py   Phase 2 data
  prep_flores.py           prep_natural.py       prep_phase3.py      build evaluation tables
  aldi.py, aldi_train.py   dialectness scoring (GPU)
  count_tokens.py          per-sentence token counts, round-trip, UNK / byte-fallback, Arabic vocabulary use
  train_reference.py       reference BPE / Unigram tokenizers
  analyze.py               Phase 1 metrics (FLORES)    analyze_natural.py   Phase 2 metrics
  ablation.py, summarize_phase3.py, trained_norm.py     Phase 3
  run_phase1.sh, run_phase2.sh, run_phase3.sh           end-to-end pipelines
  morph_segment.py         Phase 4: CAMeL morpheme / clitic boundaries for the training mix
  train_morph.py, train_spm.py, run_spm.sh             MorphBPE, HF Unigram and SentencePiece Unigram (+ constraint)
  morph_eval.py, phase4_summary.py                     morpheme-boundary F1 and the Phase 4a tables
  lm_common.py, lm_data.py, lm_train.py, lm_eval.py    Phase 4b language models: data, training, BPC + Belebele
  run_lm.sh, run_lm_eval.sh, run_lm_ctrl.sh, phase4b_summary.py   Phase 4b queues and summary
results/                   tables (CSV), figures (PNG), tokenizer metadata, ALDi scores
tokenizers/                the reference tokenizers trained here (tokenizer.json, SentencePiece .model)
```
Load a reference tokenizer with `PreTrainedTokenizerFast(tokenizer_file="tokenizers/bpe-mix-64k.json")`.

| Name | Training data |
|---|---|
| `*-mix-*` | 50% MSA + 50% dialect |
| `*-msa-*` | 100% MSA |
| `*-mixlight-*`, `*-mixaggressive-*` | the mix after normalization |
| `bpe-mixmorpheme-*`, `bpe-mixclitic-*` | MorphBPE: the mix, no merge across a morpheme (or clitic) boundary |
| `unigram-mixmorpheme-*`, `unigram-mixclitic-*` | HF Unigram with the same constraint (replication) |
| `spm-unigram-plain-*`, `spm-unigram-{morpheme,clitic}-*` | SentencePiece Unigram without / with the constraint; load with `sentencepiece.SentencePieceProcessor(model_file=...)` |

All of them are trained on 1B characters, with diacritics removed, and cover Arabic only.

## Reproduce
```bash
git clone https://github.com/oraby8/arabic-tokenizer-eval.git && cd arabic-tokenizer-eval
python3.12 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
cp env.sh.example env.sh   # set B to this directory; add a Hugging Face token
bash src/run_phase1.sh     # ~5 min (CPU) + ALDi on 1 GPU
bash src/run_phase2.sh     # ~45 min with 4 GPUs (ALDi), 96 CPU cores (training/counting)
bash src/run_phase3.sh     # ~15 min
# Phase 4a: segment, train constrained tokenizers, count and score them
python src/morph_segment.py && python src/train_spm.py --chunk && bash src/run_spm.sh
for j in {bpe,unigram}-morpheme-{32k,64k,128k} {bpe,unigram}-clitic-64k; do python src/train_morph.py --job $j; done
for c in flores fineweb wiki padic; do python src/count_tokens.py --corpus $c; done
python src/morph_eval.py && python src/phase4_summary.py
# Phase 4b: ~9 h on 4x L40S
python src/lm_data.py && bash src/run_lm.sh && bash src/run_lm_eval.sh && bash src/run_lm_ctrl.sh && python src/phase4b_summary.py
```
**Gated Hugging Face repos:** accept the licenses for FLORES+ (`openlanguagedata/flores_plus`), Llama 3.1/3.2/4, Gemma 2/3/4, Aya Expanse, Command A/R/R7B-Arabic, and Jais-adapted. Everything else is public.


## Data sources
Corpora are downloaded at run time and are not redistributed here. Each is subject to its own license.

| Corpus | Use |
|---|---|
| FLORES+ (`openlanguagedata/flores_plus`, CC BY-SA 4.0) | parallel English, MSA and 9 dialects |
| FineWeb-2 (`HuggingFaceFW/fineweb-2`, ODC-By) | natural web text; train split for the reference tokenizers |
| Wikipedia (`wikimedia/wikipedia` 20231101: ar, arz, ary) | encyclopedic text |
| PADIC (`arbml/PADIC`) | parallel conversational MSA, Algiers, Annaba, Syrian, Palestinian |
| ArSarcasm-v2 (`arbml/ArSarcasm_v2`) | tweets |
| Arabic diacritization benchmark (`arbml/arabic_text_diacritization`, Tashkeela-derived) | diacritized text |
| Sentence-ALDi (`AMR-KELEG/Sentence-ALDi`) | dialectness scores |

## Not covered
- **Falcon-Arabic:** there is no public tokenizer.
- **MADAR:** it requires a request to CAMeL Lab.
- **QADI:** only tweet IDs are released.
- **Closed-model APIs** (Claude, Gemini, Chatgpt): token counting was not run.
- **Model quality:** for example bits per character or Belebele scores. That is the natural next step.

## License
Code, result tables and figures: [MIT](LICENSE).
The reference tokenizers in `tokenizers/` are trained on FineWeb-2 text (ODC-By 1.0). Please also credit FineWeb-2 when you reuse them.
The corpora are not redistributed here and keep their own licenses (see [Data sources](#data-sources)).

## Background
This work follows up the questions in a review of five recent Arabic tokenization papers:

| Paper | Venue |
|---|---|
| [Morphemes without Borders](https://aclanthology.org/2026.lrec-1.923/) | LREC 2026 |
| [SPLINTER](https://aclanthology.org/2025.findings-acl.1151/) | Findings of ACL 2025 |
| [Semitic Root Encoding](https://aclanthology.org/2025.arabicnlp-main.3/) | ArabicNLP 2025 |
| [AraToken](https://arxiv.org/abs/2512.18399) | arXiv preprint |
| [MorphBPE](https://aclanthology.org/2026.findings-acl.2068/) | Findings of ACL 2026 |

Its open question was *"How efficient are today's popular LLM tokenizers on MSA and dialects, compared fairly?"*
