"""Shared pieces for Phase 4b language-model experiments.

LM_TOKENIZERS: the four 64K reference tokenizers (+ Qwen3's own, for the continued-pretraining control).
Encoder: uniform encode(list[str]) -> list[list[int]] plus eos id, vocab size and per-token raw bytes
(the bytes are used to initialise transplanted embeddings from Qwen3's embeddings).
"""
import json, os

B = os.environ["B"]
QWEN = "Qwen/Qwen3-0.6B-Base"
LM_TOKENIZERS = {
    "BPE": ("local", "tokenizers/bpe-mix-64k.json"),
    "MorphBPE": ("local", "tokenizers/bpe-mixmorpheme-64k.json"),
    "SPM-Unigram": ("spm", "tokenizers/spm-unigram-plain-64k.model"),
    "SPM-Unigram-morph": ("spm", "tokenizers/spm-unigram-morpheme-64k.model"),
    "Qwen3": ("hf", QWEN),
}


def _byte_decoder():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


class Encoder:
    def __init__(self, name):
        kind, path = LM_TOKENIZERS[name]
        self.name, self.kind = name, kind
        if kind == "spm":
            import sentencepiece as spm
            self.sp = spm.SentencePieceProcessor(model_file=f"{B}/{path}")
            self.vocab_size = self.sp.get_piece_size()
            self.eos = self.sp.eos_id()
        else:
            from tokenizers import Tokenizer
            if kind == "local":
                self.tk = Tokenizer.from_file(f"{B}/{path}")
                self.eos = self.tk.token_to_id("<|endoftext|>")
            else:
                from huggingface_hub import hf_hub_download
                self.tk = Tokenizer.from_file(hf_hub_download(path, "tokenizer.json"))
                self.eos = self.tk.token_to_id("<|endoftext|>")
            self.vocab_size = self.tk.get_vocab_size()

    def encode(self, texts, threads=32):
        if self.kind == "spm":
            return self.sp.encode(texts, num_threads=threads)
        return [e.ids for e in self.tk.encode_batch(texts, add_special_tokens=False)]

    def encode_offsets(self, text):
        if self.kind == "spm":
            proto = self.sp.encode(text, return_type="proto")
            b2c, n = {}, 0
            for i, ch in enumerate(text):
                b2c[n] = i; n += len(ch.encode("utf-8"))
            b2c[n] = len(text)
            return [p.id for p in proto.pieces], [(b2c.get(p.begin, len(text)), b2c.get(p.end, len(text))) for p in proto.pieces]
        e = self.tk.encode(text, add_special_tokens=False)
        return e.ids, e.offsets

    def token_bytes(self):
        """Raw bytes of every token id (None for special / control tokens)."""
        out = []
        if self.kind == "spm":
            for i in range(self.vocab_size):
                if self.sp.is_control(i) or self.sp.is_unknown(i):
                    out.append(None)
                elif self.sp.is_byte(i):
                    out.append(bytes([int(self.sp.id_to_piece(i)[3:5], 16)]))
                else:
                    out.append(self.sp.id_to_piece(i).replace("▁", " ").encode("utf-8"))
            return out
        dec = _byte_decoder()
        inv = {v: k for k, v in self.tk.get_vocab().items()}
        special = {t.content for t in self.tk.get_added_tokens_decoder().values()} if hasattr(self.tk, "get_added_tokens_decoder") else set()
        for i in range(self.vocab_size):
            t = inv.get(i)
            if t is None or t in special:
                out.append(None)
            else:
                out.append(bytes(dec[c] for c in t) if all(c in dec for c in t) else t.encode("utf-8"))
        return out


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, "w"), indent=1)
