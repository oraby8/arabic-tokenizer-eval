"""Phase 4b: encode the 1B-char training mix with each LM tokenizer -> data/lm/<name>.bin (uint32).

Documents (lines of mix.txt) are separated by the tokenizer's EOS id. Every tokenizer sees exactly
the same characters, so the models are compared at equal *data*, not equal tokens.
"""
import os, sys
import numpy as np
from lm_common import B, Encoder, LM_TOKENIZERS, save_json

names = sys.argv[1:] or list(LM_TOKENIZERS)
lines = [l.rstrip("\n") for l in open(f"{B}/data/train_text/mix.txt")]
n_chars = sum(len(l) for l in lines)
os.makedirs(f"{B}/data/lm", exist_ok=True)
for name in names:
    enc = Encoder(name)
    out = open(f"{B}/data/lm/{name}.bin", "wb")
    n_tok = 0
    for i in range(0, len(lines), 20000):
        ids = enc.encode(lines[i:i + 20000])
        flat = np.fromiter((t for doc in ids for t in (*doc, enc.eos)), dtype=np.uint32)
        flat.tofile(out); n_tok += len(flat)
    out.close()
    save_json(dict(name=name, n_tokens=n_tok, n_chars=n_chars, chars_per_token=n_chars / n_tok,
                   eos=enc.eos, vocab_size=enc.vocab_size), f"{B}/data/lm/{name}.json")
    print(f"{name}: {n_tok:,} tokens, {n_chars / n_tok:.3f} chars/token", flush=True)
