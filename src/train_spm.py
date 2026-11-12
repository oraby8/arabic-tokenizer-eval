"""Phase 4a: SentencePiece Unigram, plain and morphology-constrained.

Same 1B-char mix as the other reference tokenizers. The morph variants are trained on the
U+E000-marked text with --pretokenization_delimiter=U+E000, so no piece can cross a morpheme
boundary; the delimiter is removed and the model runs on plain text.
SentencePiece skips sentences longer than max_sentence_length, so every training file is first
cut into chunks of <= 2,000 characters at sentence punctuation (same rule for all variants).
normalization_rule_name=identity: no NFKC or whitespace changes (text is already NFC + dediac).
Usage: python train_spm.py --chunk            (writes *.spm.txt once)
       python train_spm.py --job plain-64k | morpheme-32k | clitic-64k
"""
import argparse, os, re
import sentencepiece as spm

B = os.environ["B"]
TXT = f"{B}/data/train_text"
MARK = ""
SENT = re.compile(r"(?<=[.!?؟])\s+")
SRC = {"plain": "mix.txt", "morpheme": "mix_morpheme.txt", "clitic": "mix_clitic.txt"}


def chunk(src, dst, max_chars=2000):
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            buf = ""
            for s in SENT.split(line.strip()):
                while len(s) > max_chars:  # very long "sentences": cut at the last space
                    cut = s.rfind(" ", 0, max_chars)
                    cut = cut if cut > 0 else max_chars
                    fout.write(s[:cut].strip() + "\n"); s = s[cut:]
                if len(buf) + len(s) + 1 > max_chars and buf:
                    fout.write(buf + "\n"); buf = ""
                buf = f"{buf} {s}".strip()
            if buf:
                fout.write(buf + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", action="store_true")
    ap.add_argument("--job")
    ap.add_argument("--threads", type=int, default=24)
    a = ap.parse_args()
    if a.chunk:
        for k, f in SRC.items():
            chunk(f"{TXT}/{f}", f"{TXT}/{k}.spm.txt")
            print("chunked", k, flush=True)
    if a.job:
        variant, size = a.job.split("-")
        extra = {"pretokenization_delimiter": MARK} if variant != "plain" else {}
        os.makedirs(f"{B}/tokenizers", exist_ok=True)
        spm.SentencePieceTrainer.train(
            input=f"{TXT}/{variant}.spm.txt", model_prefix=f"{B}/tokenizers/spm-unigram-{variant}-{size}",
            model_type="unigram", vocab_size=int(size[:-1]) * 1024, character_coverage=0.9999,
            byte_fallback=True, normalization_rule_name="identity", remove_extra_whitespaces=False,
            max_sentence_length=16384, input_sentence_size=0, shuffle_input_sentence=True,
            num_threads=a.threads, train_extremely_large_corpus=True, minloglevel=1, **extra)
        sp = spm.SentencePieceProcessor(model_file=f"{B}/tokenizers/spm-unigram-{variant}-{size}.model")
        bad = sum(MARK in sp.id_to_piece(i) for i in range(sp.get_piece_size()))
        print("saved", a.job, sp.get_piece_size(), "pieces containing the delimiter:", bad, flush=True)
