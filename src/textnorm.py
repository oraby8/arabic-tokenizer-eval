"""Text normalization and word spans shared by every corpus (same rules as FLORES)."""
import re
import unicodedata
from camel_tools.utils.dediac import dediac_ar
from camel_tools.tokenizers.word import simple_word_tokenize

DIAC = re.compile("[ً-ْٰ]")


def normalize(text, arabic=True):
    text = unicodedata.normalize("NFC", text)
    return dediac_ar(text) if arabic else text


def truncate(text, max_chars):
    if len(text) <= max_chars:
        return text
    cut = text.rfind(" ", 0, max_chars)
    return text[: cut if cut > max_chars // 2 else max_chars]


def word_spans(text):
    spans, pos = [], 0
    for w in simple_word_tokenize(text):
        i = text.find(w, pos)
        if i < 0:
            raise ValueError(f"word not found: {w!r}")
        spans.append((i, i + len(w)))
        pos = i + len(w)
    return spans
