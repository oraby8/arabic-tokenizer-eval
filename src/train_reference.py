"""Train Arabic reference tokenizers on FineWeb-2 *train* splits (disjoint from the eval docs).

Training sets (same normalization as eval: NFC + dediac), each ~1B characters:
  mix : 50% MSA (arb) + 50% dialect docs with ALDi >= 0.3 from arz/ary/ars train splits
        (FineWeb-2 dialect labels alone are unreliable: the ary subset is mostly MSA news)
  msa : 100% MSA (arb)
apc/acm/aeb are never used for training (they act as unseen dialects).

Tokenizers: byte-level BPE and Unigram (HF tokenizers), vocab 32K/64K/128K on mix, 64K on msa.
"""
import argparse, glob, os, random
import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from textnorm import normalize

B = os.environ["B"]
RAW, OUT = f"{B}/data/raw", f"{B}/data/train_text"
TARGET = 1_000_000_000


def stream(code):
    for f in sorted(glob.glob(f"{RAW}/fw2.{code}.train.*.parquet")):
        for batch in pq.ParquetFile(f).iter_batches(columns=["text"], batch_size=4096):
            for t in batch.column(0).to_pylist():
                yield normalize(t)


def build_text(name, budget):
    path = f"{OUT}/{name}.txt"
    if os.path.exists(path):
        return path
    os.makedirs(OUT, exist_ok=True)
    with open(path + ".tmp", "w") as f:
        for code, chars in budget.items():
            n = 0
            for t in stream(code):
                f.write(t.replace("\n", " ") + "\n")
                n += len(t)
                if n >= chars:
                    break
            print(f"{name}: {code} {n:,} chars", flush=True)
    os.rename(path + ".tmp", path)
    return path


def build_mix_text(path_name="mix", min_aldi=0.3):
    """50% MSA + 50% dialect docs selected by ALDi (scores from aldi_train.py), balanced over
    arz/ary/ars as far as each has enough dialectal text; any shortfall is filled by the others."""
    path = f"{OUT}/{path_name}.txt"
    if os.path.exists(path):
        return path
    import pandas as pd
    sc = pd.concat([pd.read_parquet(f) for f in glob.glob(f"{B}/data/train_aldi/shard*.parquet")])
    sc = sc[sc.aldi >= min_aldi].sample(frac=1, random_state=0)
    avail = sc.groupby("code").n_chars.sum().to_dict()
    want, left, codes = {}, TARGET // 2, sorted(avail, key=avail.get)
    for i, c in enumerate(codes):  # smallest first, so leftovers flow to larger pools
        want[c] = min(avail[c], left // (len(codes) - i))
        left -= want[c]
    with open(path + ".tmp", "w") as f:
        n = 0
        for t in stream("arb_Arab"):
            f.write(t.replace("\n", " ") + "\n"); n += len(t)
            if n >= TARGET // 2:
                break
        print(f"{path_name}: arb_Arab {n:,} chars", flush=True)
        for c in codes:
            n = 0
            for fname, rows in sc[sc.code == c].groupby("file").row:
                texts = pq.read_table(f"{RAW}/{fname}", columns=["text"]).column(0).to_pylist()
                for r in rows:
                    t = normalize(texts[r]); f.write(t.replace("\n", " ") + "\n"); n += len(t)
                    if n >= want[c]:
                        break
                if n >= want[c]:
                    break
            print(f"{path_name}: {c} {n:,} chars (ALDi>={min_aldi}, available {avail[c]:,})", flush=True)
    os.rename(path + ".tmp", path)
    return path


def train(kind, vocab, text_path, out):
    if kind == "bpe":
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        tr = trainers.BpeTrainer(vocab_size=vocab, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                 special_tokens=["<|endoftext|>"], show_progress=False)
    else:
        tok = Tokenizer(models.Unigram())
        tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always")
        tok.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")
        tr = trainers.UnigramTrainer(vocab_size=vocab, special_tokens=["<unk>"], unk_token="<unk>",
                                     max_piece_length=16, show_progress=False)
    tok.train([text_path], tr)
    tok.save(out)
    print("saved", out, tok.get_vocab_size(), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-only", action="store_true")
    ap.add_argument("--job")  # e.g. bpe-mix-64k
    a = ap.parse_args()
    msa = build_text("msa", {"arb_Arab": TARGET})
    mix = build_mix_text()
    if a.job:
        kind, data, v = a.job.split("-")
        os.makedirs(f"{B}/tokenizers", exist_ok=True)
        train(kind, int(v[:-1]) * 1024, mix if data == "mix" else msa, f"{B}/tokenizers/{a.job}.json")
