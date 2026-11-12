# Phase 4b: do morphology-constrained tokenizers help the model?

Run date: 2026-11-11 · Code: `src/lm_common.py`, `src/lm_data.py`, `src/lm_train.py`, `src/lm_eval.py`, `src/run_lm.sh`, `src/run_lm_eval.sh`, `src/run_lm_ctrl.sh`, `src/phase4b_summary.py` · Outputs: `results/phase4/lm/`, `results/phase4/lm_train/`, `results/phase4/phase4b_*.csv`

## Setup
- **Tokenizers:** four 64K reference tokenizers from Phase 4a, all trained on the same 1B characters.

  | Tokenizer | Training tokens | Chars/token |
  |---|---|---|
  | BPE | 236.8M | 4.22 |
  | MorphBPE (morpheme constraint) | 299.5M | 3.34 |
  | SentencePiece Unigram | 238.0M | 4.20 |
  | SentencePiece Unigram + morpheme constraint | 370.3M | 2.70 |
  | Qwen3 (own tokenizer, control) | 402.9M | 2.48 |

- **Training data:** every model gets one epoch over the same 1B-character mix (50% MSA, 50% dialect), so the models are compared at **equal characters**.
- **From scratch:** a Llama-style model with 135M parameters (12 layers, d = 768, tied embeddings), lr 6e-4, batch 128 × 1,024 tokens, cosine schedule.
- **Transplant:** Qwen3-0.6B-Base with its tokenizer replaced.
  - Each new embedding starts as the mean of Qwen3's embeddings for the same bytes.
  - The body is frozen for the first 5% of steps.
  - lr 1e-4 for the body and 5e-4 for the embeddings.
- **Controls:**
  - Qwen3-0.6B trained on the same data with its own tokenizer.
  - Raw Qwen3-0.6B, with no training.
  - **Compute-matched from-scratch runs:** each model retrained with its twin's number of optimizer steps, in both directions (see below).
- **Bits per character (BPC):** total −log₂ p over a document divided by its number of characters, so it doesn't depend on the tokenizer.
  - 22 held-out sets: FLORES+ (MSA and 9 dialects), PADIC (MSA and 4 dialects), and FineWeb-2 test documents (MSA with ALDi < 0.1, dialects with ALDi ≥ 0.3).
  - Every model is scored on exactly the same 69,283 documents: those that fit in 1,024 tokens under all five tokenizers.
- **Belebele:** zero-shot, on 6 Arabic varieties with 900 questions each.
  - Each answer option is scored by its log-likelihood divided by its number of characters, so the four options compete fairly whatever the tokenizer.
  - The 95% CI is the binomial approximation.

## Bits per character
MSA is the arb_Arab / padic_msa set. Dialects is the unweighted mean over the dialect sets. Lower is better.

| Model | FLORES MSA | FLORES dialects | PADIC MSA | PADIC dialects | FineWeb MSA | FineWeb dialects | All MSA | All dialects | Train tokens | GPU-h |
|---|---|---|---|---|---|---|---|---|---|---|
| Raw Qwen3-0.6B | 1.914 | 2.428 | 2.914 | 3.927 | 1.436 | 2.548 | 2.088 | 2.781 | – | – |
| Qwen3-0.6B, own tokenizer, trained | 1.592 | 1.911 | 2.291 | 2.836 | 1.367 | 2.176 | 1.750 | 2.189 | 403M | 5.4 |
| Scratch BPE | 1.869 | 2.181 | 2.497 | 3.036 | 1.765 | 2.662 | 2.044 | 2.513 | 237M | 0.6 |
| Scratch MorphBPE | 1.855 | 2.157 | 2.478 | 3.018 | 1.732 | 2.636 | 2.022 | 2.490 | 300M | 0.7 |
| Scratch SPM-Unigram | 1.811 | 2.125 | 2.414 | 2.951 | 1.711 | 2.610 | 1.978 | 2.452 | 238M | 0.6 |
| Scratch SPM-Unigram + morph | 1.738 | 2.046 | 2.369 | 2.912 | 1.567 | 2.414 | 1.891 | 2.344 | 370M | 1.0 |
| Transplant BPE | 1.594 | 1.916 | 2.312 | 2.863 | 1.374 | 2.258 | 1.760 | 2.223 | 237M | 2.9 |
| Transplant MorphBPE | 1.591 | 1.907 | 2.291 | 2.859 | 1.353 | 2.217 | 1.745 | 2.206 | 300M | 3.6 |
| Transplant SPM-Unigram | 1.571 | 1.893 | 2.272 | 2.818 | 1.374 | 2.272 | 1.739 | 2.207 | 238M | 2.9 |
| Transplant SPM-Unigram + morph | **1.537** | **1.859** | **2.250** | **2.801** | **1.280** | **2.101** | **1.689** | **2.134** | 370M | 4.6 |

## Equal characters vs equal compute
At equal characters, a tokenizer that produces more tokens also gives its model more optimizer steps: 56% more for SPM-Unigram + morph, and 26% more for MorphBPE. The compute-matched controls retrain each from-scratch model with its twin's step count:
- **Base trained longer:** the unconstrained model gets the constrained model's step count. That means 1.26–1.56 epochs, so some text is repeated.
- **Variant trained shorter:** the constrained model gets the unconstrained model's step count, so it sees only 64–79% of the characters.

| Comparison (from scratch) | Setting | Mean BPC change | MSA | Dialects | Sets where the variant is better | Belebele change |
|---|---|---|---|---|---|---|
| MorphBPE vs BPE | equal characters (main run) | −0.98% | −1.13% | −0.96% | 22/22 | 0.0 pts |
| MorphBPE vs BPE | equal steps: BPE trained longer | **+1.78%** | +2.14% | +1.72% | **0/22** | −0.8 pts |
| MorphBPE vs BPE | equal steps: MorphBPE trained shorter | **+1.76%** | +2.23% | +1.69% | **0/22** | +0.1 pts |
| SPM + morph vs SPM | equal characters (main run) | −4.47% | −4.76% | −4.42% | 22/22 | −0.4 pts |
| SPM + morph vs SPM | equal steps: SPM trained longer | **−1.07%** | −1.00% | −1.08% | 17/22 | −0.8 pts |
| SPM + morph vs SPM | equal steps: SPM + morph trained shorter | **−0.64%** | −0.13% | −0.72% | 18/22 | −0.7 pts |
| SPM-Unigram vs BPE | equal characters (~equal compute: 238M vs 237M tokens) | −2.55% | −3.18% | −2.45% | 22/22 | +0.9 pts |

The transplant runs aren't compute-matched. Their equal-character gains (MorphBPE −0.8%, SPM + morph −3.3%) carry the same confound.

## Belebele (accuracy, 900 questions per variety; chance = 25%)

| Model | MSA | Egyptian | Moroccan | Najdi | Levantine | Mesopotamian | Mean (95% CI) |
|---|---|---|---|---|---|---|---|
| Raw Qwen3-0.6B | 36.0 | 31.1 | 28.8 | 32.0 | 32.7 | 31.1 | 31.9 ± 1.2 |
| Qwen3-0.6B, own tokenizer, trained | 34.3 | 30.2 | 30.6 | 31.4 | 31.4 | 31.7 | 31.6 ± 1.2 |
| Transplant BPE | 33.1 | 30.3 | 31.1 | 30.4 | 32.0 | 31.4 | 31.4 ± 1.2 |
| Transplant MorphBPE | 32.6 | 30.1 | 29.2 | 29.8 | 30.1 | 31.6 | 30.6 ± 1.2 |
| Transplant SPM-Unigram | 34.0 | 31.8 | 30.7 | 32.1 | 33.7 | 32.6 | 32.5 ± 1.2 |
| Transplant SPM-Unigram + morph | **36.2** | **33.0** | **31.4** | **33.7** | **34.7** | **32.8** | **33.6 ± 1.3** |
| From-scratch models (all 8) | 26–28 | | | | | | 26.8–28.1 |

## What this shows
1. **Most of the morphology gain is extra compute.**
   - At equal characters, both constrained tokenizers lower BPC on every test set.
   - At equal compute, **MorphBPE is worse than BPE on all 22 sets (+1.8%)**, in both control directions.
   - **SPM-Unigram + morph keeps a small advantage (−0.6% to −1.1%, 17–18/22 sets).** It costs 64% more tokens per MSA word (Phase 4a).
2. **Plain Unigram beats plain BPE at about equal compute** (−2.5%, 22/22 sets), with no morphology data. This matches Phase 4a, where plain Unigram already aligned better with morpheme boundaries than BPE.
3. **The effects are the same size on MSA and on dialects.** That's notable, because the constraint mostly binds on MSA (Phase 4a). Whatever helps, it isn't specific to the morphology the analyzer finds.
4. **A transplanted reference tokenizer matches continued training with Qwen3's own tokenizer, using fewer tokens.**
   - Transplant SPM-Unigram reaches 1.739 / 2.207 (MSA / dialects) with 238M training tokens.
   - The Qwen3 control reaches 1.750 / 2.189 with 403M.
   - Qwen3's own tokenizer is 2.48 chars/token on this data, against 4.20 for the reference tokenizer.
5. **Belebele doesn't separate the tokenizers at this scale.**
   - The from-scratch models are at chance.
   - The transplants are 30.6–33.6%, within about 3 points of each other and of raw Qwen3 (31.9%), against a ±1.2-point CI.
   - Transplant SPM + morph is highest on every variety, but its edge over transplant SPM-Unigram (1.1 points) is within noise, and this comparison is not compute-matched.
   - Continued training on 1B characters did not improve raw Qwen3's Belebele score.

## Caveats
- **No paired tests.** `lm_eval.py` saved BPC sums per test set and Belebele accuracy per variety, not per-document or per-question scores. Consistency is therefore reported as the number of the 22 sets on which a model wins, and Belebele with an unpaired binomial CI.
- **One seed per configuration**, a small scale (135M from scratch, 0.6B transplant) and one epoch over 1B characters. Differences of about 1% in BPC are near what a different seed could produce. The compute-matched SPM result (17–18/22 sets) should be read as weak evidence.
- **The compute-matched controls trade one confound for another:** repeated data in one direction, less data in the other. The two directions agree, which is the main reason to trust them.
- **The transplants are not compute-matched,** so their equal-character gains include the extra-step effect.
- **The morphology analyzer is weak on dialects** (Phase 4a). A better dialect segmenter could change the dialect results.
- **Two transplant runs (MorphBPE and SPM-Unigram) ran out of GPU memory at start-up**, because an evaluation shared the GPU. They were restarted from step 0 with identical settings, so the results are unaffected.
