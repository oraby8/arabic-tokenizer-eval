"""Tokenize the prepared corpus with every tokenizer and store per-sentence counts.

Rules applied identically to all tokenizers:
- add_special_tokens=False, no chat template
- input text is the same NFC string; no extra normalization
- "continued word": a word span overlapped by 2+ tokens (via character offsets)

Outputs (in $B/results/<corpus>/):
  per_sentence/<name>.parquet  per-sentence counts
  tokenizers.json              metadata: vocab size, Arabic vocab share, round-trip, dedup hash
"""
import argparse, hashlib, json, os, re, sys, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from registry import TOKENIZERS

B = os.environ["B"]
ARABIC = re.compile("[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
BYTE_FALLBACK = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")


class TT:
    """tiktoken wrapper exposing the few methods used below."""

    def __init__(self, enc_name):
        import tiktoken
        self.enc = tiktoken.get_encoding(enc_name)
        self.vocab_size = self.enc.n_vocab

    def encode_offsets(self, text):
        ids = self.enc.encode_ordinary(text)
        _, starts = self.enc.decode_with_offsets(ids)
        ends = starts[1:] + [len(text)]
        return ids, list(zip(starts, ends))

    def decode(self, ids):
        return self.enc.decode(ids)

    def token_strings(self):
        for i in range(self.enc.n_vocab):
            try:
                yield i, self.enc.decode_single_token_bytes(i).decode("utf-8", errors="replace")
            except KeyError:
                continue

    def is_unk(self, ids):
        return [False] * len(ids)

    def is_byte_fallback(self, ids):
        return [False] * len(ids)


class HF:
    def __init__(self, repo, local=False):
        if local:  # reference tokenizer trained in this project (tokenizer.json path)
            from transformers import PreTrainedTokenizerFast
            self.tok = PreTrainedTokenizerFast(tokenizer_file=repo, unk_token="<unk>" if "unigram" in repo else None)
        else:
            from transformers import AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(repo, use_fast=True)
        if not self.tok.is_fast:
            raise RuntimeError("no fast tokenizer (offsets unavailable)")
        self.vocab_size = len(self.tok)
        self.unk = self.tok.unk_token_id
        toks = self.tok.convert_ids_to_tokens(list(range(len(self.tok))))
        self.bf = {i for i, t in enumerate(toks) if t and BYTE_FALLBACK.match(t)}

    def encode_offsets(self, text):
        e = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return e["input_ids"], e["offset_mapping"]

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def token_strings(self):
        for i in range(len(self.tok)):
            yield i, self.tok.decode([i])

    def is_unk(self, ids):
        return [i == self.unk for i in ids] if self.unk is not None else [False] * len(ids)

    def is_byte_fallback(self, ids):
        return [i in self.bf for i in ids]


class SPM:
    """SentencePiece model (Phase 4); offsets come from the immutable proto (byte -> char)."""

    def __init__(self, path):
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor(model_file=path)
        self.vocab_size = self.sp.get_piece_size()

    def encode_offsets(self, text):
        proto = self.sp.encode(text, return_type="proto")
        b2c, n = {}, 0
        for i, ch in enumerate(text):
            b2c[n] = i
            n += len(ch.encode("utf-8"))
        b2c[n] = len(text)
        ids, offs = [], []
        for p in proto.pieces:
            ids.append(p.id)
            offs.append((b2c.get(p.begin, len(text)), b2c.get(p.end, len(text))))
        return ids, offs

    def decode(self, ids):
        return self.sp.decode(ids)

    def token_strings(self):
        for i in range(self.vocab_size):
            yield i, self.sp.id_to_piece(i).replace("\u2581", " ")

    def is_unk(self, ids):
        return [self.sp.is_unknown(i) for i in ids]

    def is_byte_fallback(self, ids):
        return [self.sp.is_byte(i) for i in ids]


def load_tokenizer(cfg):
    if cfg["kind"] == "tiktoken":
        return TT(cfg["repo"])
    if cfg["kind"] == "spm":
        return SPM(f"{B}/{cfg['repo']}")
    return HF(f"{B}/{cfg['repo']}" if cfg["kind"] == "local" else cfg["repo"], local=cfg["kind"] == "local")


def continued_words(spans, offsets):
    """Count word spans covered by 2+ non-empty tokens."""
    offs = sorted((s, e) for s, e in offsets if e > s)
    n, j = 0, 0
    for ws, we in spans:
        while j < len(offs) and offs[j][1] <= ws:
            j += 1
        k, c = j, 0
        while k < len(offs) and offs[k][0] < we:
            c += 1
            k += 1
        n += c >= 2
    return n


def run_one(cfg, corpus):
    out_dir = f"{B}/results/{corpus}"
    t = load_tokenizer(cfg)
    df = pd.read_parquet(f"{B}/data/prepared/{corpus}.parquet")
    rows, h = [], hashlib.sha256()
    rt_fail = 0
    used = set()
    for r in df.itertuples():
        ids, offs = t.encode_offsets(r.text)
        h.update(repr(ids).encode())
        if r.variety != "eng_Latn":
            used.update(ids)
        dec = t.decode(ids)
        ok = dec.strip() == r.text.strip()
        rt_fail += not ok
        rows.append(dict(
            sid=r.sid, variety=r.variety, n_tokens=len(ids),
            n_unk=sum(t.is_unk(ids)), n_byte_fallback=sum(t.is_byte_fallback(ids)),
            n_continued=continued_words(list(r.spans), offs), roundtrip_ok=ok,
        ))
    out = pd.DataFrame(rows)
    out.to_parquet(f"{out_dir}/per_sentence/{cfg['name'].replace('/', '_')}.parquet")
    strings = list(t.token_strings())
    ar_ids = {i for i, s in strings if ARABIC.search(s)}
    n_ar = len(ar_ids)
    return dict(
        name=cfg["name"], group=cfg["group"], repo=cfg["repo"], vocab_size=t.vocab_size,
        arabic_vocab=n_ar, arabic_vocab_share=n_ar / max(len(strings), 1),
        arabic_vocab_used=len(ar_ids & used),  # Arabic-script tokens seen on this corpus's Arabic text
        roundtrip_fail=rt_fail, output_hash=h.hexdigest()[:16],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="flores")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    os.makedirs(f"{B}/results/{a.corpus}/per_sentence", exist_ok=True)
    meta_path = f"{B}/results/{a.corpus}/tokenizers.json"
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    cfgs = [c for c in TOKENIZERS if (a.only and c["name"] in a.only) or (not a.only and not c.get("pending"))]
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(run_one, c, a.corpus): c for c in cfgs}
        for f in as_completed(futs):
            c = futs[f]
            try:
                m = f.result()
                meta[c["name"]] = m
                print(f"OK   {c['name']:18s} vocab={m['vocab_size']:>7} ar_share={m['arabic_vocab_share']:.3f} rt_fail={m['roundtrip_fail']}", flush=True)
            except Exception as e:
                print(f"FAIL {c['name']:18s} {type(e).__name__}: {str(e)[:200]}", flush=True)
                traceback.print_exc(file=sys.stderr)
    json.dump(meta, open(meta_path, "w"), indent=1)


if __name__ == "__main__":
    main()
