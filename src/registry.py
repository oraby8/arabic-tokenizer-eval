"""Tokenizers under evaluation.

group: "global" = general multilingual LLMs, "arabic" = Arabic-centric LLMs.
core=True marks the user's selected list (used by default in analysis);
core=False entries are kept as extra reference points (run with TOKSET=all).
pending=True: gated repo without access yet; skipped until it works.
Not available: Falcon-Arabic (no public Hugging Face repo as of 2026-11-10).
"""

TOKENIZERS = [
    # --- global open models (selected) ---
    dict(name="Llama-3.1", group="global", kind="hf", repo="meta-llama/Llama-3.1-8B", core=True),
    dict(name="Llama-3.2", group="global", kind="hf", repo="meta-llama/Llama-3.2-1B", core=True),
    dict(name="Llama-4", group="global", kind="hf", repo="meta-llama/Llama-4-Scout-17B-16E", core=True),
    dict(name="Qwen2.5", group="global", kind="hf", repo="Qwen/Qwen2.5-7B", core=True),
    dict(name="Qwen3", group="global", kind="hf", repo="Qwen/Qwen3-8B", core=True),
    dict(name="Qwen3.5", group="global", kind="hf", repo="Qwen/Qwen3.5-9B", core=True),
    dict(name="Gemma-3", group="global", kind="hf", repo="google/gemma-3-12b-pt", core=True),
    dict(name="Gemma-4", group="global", kind="hf", repo="google/gemma-4-31B-it", core=True),
    dict(name="DeepSeek-V3", group="global", kind="hf", repo="deepseek-ai/DeepSeek-V3", core=True),
    dict(name="Aya-Expanse", group="global", kind="hf", repo="CohereLabs/aya-expanse-8b", core=True),
    dict(name="Command-A", group="global", kind="hf", repo="CohereLabs/c4ai-command-a-03-2025", core=True),
    dict(name="Command-R", group="global", kind="hf", repo="CohereLabs/c4ai-command-r-08-2024", core=True),
    dict(name="Command-R7B-Arabic", group="arabic", kind="hf", repo="CohereLabs/c4ai-command-r7b-arabic-02-2025", core=True),
    dict(name="gpt-oss", group="global", kind="hf", repo="openai/gpt-oss-20b", core=True),
    # --- Arabic-centric models (selected) ---
    dict(name="Jais-3-0.9B", group="arabic", kind="hf", repo="inception42/Jais-3-0.9B", core=True),
    dict(name="Jais-adapted", group="arabic", kind="hf", repo="inceptionai/jais-adapted-7b", core=True),
    dict(name="ALLaM-7B", group="arabic", kind="hf", repo="humain-ai/ALLaM-7B-Instruct-preview", core=True),
    dict(name="Fanar-1", group="arabic", kind="hf", repo="QCRI/Fanar-1-9B", core=True),
    dict(name="Fanar-2", group="arabic", kind="hf", repo="QCRI/Fanar-2-27B-Instruct", core=True),
    dict(name="Falcon-H1", group="global", kind="hf", repo="tiiuae/Falcon-H1-7B-Base", core=True),
    dict(name="Karnak-6B", group="arabic", kind="hf", repo="Applied-Innovation-Center/Karnak-6B-v1.0", core=True),
    dict(name="Karnak-40B", group="arabic", kind="hf", repo="Applied-Innovation-Center/Karnak-40B-v1.0", core=True),
    dict(name="Atlas-Chat", group="arabic", kind="hf", repo="MBZUAI-Paris/Atlas-Chat-9B", core=True),
    dict(name="Nile-Chat", group="arabic", kind="hf", repo="MBZUAI-Paris/Nile-Chat-12B", core=True),
    # --- reference tokenizers trained in this project (src/train_reference.py) ---
    dict(name="Ref-BPE-mix-32K", group="reference", kind="local", repo="tokenizers/bpe-mix-32k.json", core=True),
    dict(name="Ref-BPE-mix-64K", group="reference", kind="local", repo="tokenizers/bpe-mix-64k.json", core=True),
    dict(name="Ref-BPE-mix-128K", group="reference", kind="local", repo="tokenizers/bpe-mix-128k.json", core=True),
    dict(name="Ref-BPE-msa-64K", group="reference", kind="local", repo="tokenizers/bpe-msa-64k.json", core=True),
    dict(name="Ref-Unigram-mix-32K", group="reference", kind="local", repo="tokenizers/unigram-mix-32k.json", core=True),
    dict(name="Ref-Unigram-mix-64K", group="reference", kind="local", repo="tokenizers/unigram-mix-64k.json", core=True),
    dict(name="Ref-Unigram-mix-128K", group="reference", kind="local", repo="tokenizers/unigram-mix-128k.json", core=True),
    dict(name="Ref-Unigram-msa-64K", group="reference", kind="local", repo="tokenizers/unigram-msa-64k.json", core=True),
    # --- Phase 4: morphology-constrained reference tokenizers (same 1B-char mix and sizes) ---
    # MorphBPE: HF BPE, no merges across CAMeL morpheme boundaries (src/train_morph.py)
    dict(name="Ref-MorphBPE-morpheme-32K", group="reference", kind="local", repo="tokenizers/bpe-mixmorpheme-32k.json", core=True, phase=4),
    dict(name="Ref-MorphBPE-morpheme-64K", group="reference", kind="local", repo="tokenizers/bpe-mixmorpheme-64k.json", core=True, phase=4),
    dict(name="Ref-MorphBPE-morpheme-128K", group="reference", kind="local", repo="tokenizers/bpe-mixmorpheme-128k.json", core=True, phase=4),
    dict(name="Ref-MorphBPE-clitic-64K", group="reference", kind="local", repo="tokenizers/bpe-mixclitic-64k.json", core=True, phase=4),
    # SentencePiece Unigram: plain baseline and with pretokenization_delimiter at morpheme boundaries (src/train_spm.py)
    dict(name="Ref-SPM-Unigram-plain-32K", group="reference", kind="spm", repo="tokenizers/spm-unigram-plain-32k.model", core=True, phase=4),
    dict(name="Ref-SPM-Unigram-plain-64K", group="reference", kind="spm", repo="tokenizers/spm-unigram-plain-64k.model", core=True, phase=4),
    dict(name="Ref-SPM-Unigram-plain-128K", group="reference", kind="spm", repo="tokenizers/spm-unigram-plain-128k.model", core=True, phase=4),
    dict(name="Ref-SPM-Unigram-morpheme-32K", group="reference", kind="spm", repo="tokenizers/spm-unigram-morpheme-32k.model", core=True, phase=4),
    dict(name="Ref-SPM-Unigram-morpheme-64K", group="reference", kind="spm", repo="tokenizers/spm-unigram-morpheme-64k.model", core=True, phase=4),
    dict(name="Ref-SPM-Unigram-morpheme-128K", group="reference", kind="spm", repo="tokenizers/spm-unigram-morpheme-128k.model", core=True, phase=4),
    dict(name="Ref-SPM-Unigram-clitic-64K", group="reference", kind="spm", repo="tokenizers/spm-unigram-clitic-64k.model", core=True, phase=4),
    # HF-tokenizers Unigram with the same constraint: replication of the SPM variant with a second trainer; appendix only
    dict(name="Ref-HFMorphUnigram-morpheme-32K", group="reference", kind="local", repo="tokenizers/unigram-mixmorpheme-32k.json", phase=4),
    dict(name="Ref-HFMorphUnigram-morpheme-64K", group="reference", kind="local", repo="tokenizers/unigram-mixmorpheme-64k.json", phase=4),
    dict(name="Ref-HFMorphUnigram-morpheme-128K", group="reference", kind="local", repo="tokenizers/unigram-mixmorpheme-128k.json", phase=4),
    dict(name="Ref-HFMorphUnigram-clitic-64K", group="reference", kind="local", repo="tokenizers/unigram-mixclitic-64k.json", phase=4),
    # --- extra reference points from Phase 1 (not in the selected list) ---
    dict(name="Jais-13B", group="arabic", kind="hf", repo="inceptionai/jais-13b"),
    dict(name="Jais-3-375B", group="arabic", kind="hf", repo="inception42/Jais-3-375B-A23B"),
    dict(name="Gemma-2", group="global", kind="hf", repo="google/gemma-2-9b"),
    dict(name="Mistral-7B-v0.1", group="global", kind="hf", repo="mistralai/Mistral-7B-v0.1"),
    dict(name="Mistral-Nemo", group="global", kind="hf", repo="mistralai/Mistral-Nemo-Base-2407"),
    dict(name="Phi-4", group="global", kind="hf", repo="microsoft/phi-4"),
    dict(name="BLOOM", group="global", kind="hf", repo="bigscience/bloom-560m"),
    dict(name="mT5", group="global", kind="hf", repo="google/mt5-base"),
    dict(name="Falcon3", group="global", kind="hf", repo="tiiuae/Falcon3-7B-Base"),
    dict(name="Apertus", group="global", kind="hf", repo="swiss-ai/Apertus-8B-2509"),
    dict(name="AceGPT-13B", group="arabic", kind="hf", repo="FreedomIntelligence/AceGPT-13B"),
    dict(name="SILMA", group="arabic", kind="hf", repo="silma-ai/SILMA-9B-Instruct-v1.0"),
    dict(name="AraGPT2", group="arabic", kind="hf", repo="aubmindlab/aragpt2-base"),
    dict(name="Hala", group="arabic", kind="hf", repo="hammh0a/Hala-9B"),
    dict(name="GPT-4 (cl100k)", group="global", kind="tiktoken", repo="cl100k_base"),
    dict(name="GPT-4o (o200k)", group="global", kind="tiktoken", repo="o200k_base"),
]

VARIETIES = {
    "eng_Latn": "English",
    "arb_Arab": "MSA",
    "arz_Arab": "Egyptian",
    "apd_Arab": "Sudanese",
    "apc_Arab_nort3139": "N. Levantine",
    "apc_Arab_sout3123": "S. Levantine",
    "acm_Arab": "Mesopotamian",
    "ars_Arab": "Najdi",
    "acq_Arab": "Ta'izzi-Adeni",
    "ary_Arab": "Moroccan",
    "aeb_Arab": "Tunisian",
}
