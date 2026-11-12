"""Phase 4a: morphology-constrained reference tokenizers (MorphBPE and Morph-Unigram).

Same 1B-char mix, same trainers and vocab sizes as Ref-*-mix-*; the only change is that
the training text carries morpheme boundaries (U+E000, from morph_segment.py) and the
*training* pre-tokenizer splits on them, so:
  - MorphBPE   : no merge can cross a morpheme boundary (MorphBPE, Fanar report, Algorithm 1)
  - MorphUnigram: no vocabulary piece can span a boundary
After training the boundary split is removed: the saved tokenizer is a standard byte-level BPE /
Metaspace Unigram that runs on plain text, like the paper's "drop-in" design.
Usage: python train_morph.py --job bpe-morpheme-64k   (kind-scheme-size)
"""
import argparse, os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

B = os.environ["B"]
MARK = ""


def train(kind, vocab, text_path, out):
    split = pre_tokenizers.Split(pattern=MARK, behavior="removed")
    if kind == "bpe":
        tok = Tokenizer(models.BPE())
        final = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.pre_tokenizer = pre_tokenizers.Sequence([split, pre_tokenizers.ByteLevel(add_prefix_space=False)])
        tok.decoder = decoders.ByteLevel()
        tr = trainers.BpeTrainer(vocab_size=vocab, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                 special_tokens=["<|endoftext|>"], show_progress=False)
    else:
        tok = Tokenizer(models.Unigram())
        final = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always")
        # Metaspace first (spaces -> ▁, one piece per word), then split words at morpheme marks
        tok.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always"), split])
        tok.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")
        tr = trainers.UnigramTrainer(vocab_size=vocab, special_tokens=["<unk>"], unk_token="<unk>",
                                     max_piece_length=16, show_progress=False)
    tok.train([text_path], tr)
    tok.pre_tokenizer = final
    tok.save(out)
    mark_bytes = pre_tokenizers.ByteLevel(add_prefix_space=False).pre_tokenize_str(MARK)[0][0]
    bad = [t for t in tok.get_vocab() if MARK in t or mark_bytes in t]
    print("saved", out, tok.get_vocab_size(), "tokens containing the marker:", len(bad), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    a = ap.parse_args()
    kind, scheme, size = a.job.split("-")
    os.makedirs(f"{B}/tokenizers", exist_ok=True)
    train(kind, int(size[:-1]) * 1024, f"{B}/data/train_text/mix_{scheme}.txt",
          f"{B}/tokenizers/{kind}-mix{scheme}-{size}.json")
