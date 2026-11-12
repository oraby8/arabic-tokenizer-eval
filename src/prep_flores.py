"""Build the aligned FLORES+ table used by every tokenizer.

Main condition: NFC + diacritics removed (CAMeL dediac_ar).
Kept alongside: the raw NFC text (diacritized ablation, Phase 3).
Word spans come from one fixed splitter (CAMeL simple_word_tokenize) for all varieties.
"""
import json, os, re, unicodedata
import pandas as pd
from camel_tools.utils.dediac import dediac_ar
from camel_tools.tokenizers.word import simple_word_tokenize
from registry import VARIETIES

B = os.environ["B"]
DIAC = re.compile("[ً-ْٰ]")


def word_spans(text):
    spans, pos = [], 0
    for w in simple_word_tokenize(text):
        i = text.find(w, pos)
        if i < 0:
            raise ValueError(f"word not found: {w!r}")
        spans.append((i, i + len(w)))
        pos = i + len(w)
    return spans


rows = []
for split in ["dev", "devtest"]:
    for code in VARIETIES:
        for line in open(f"{B}/data/flores/{split}.{code}.jsonl"):
            r = json.loads(line)
            raw = unicodedata.normalize("NFC", r["text"])
            main = dediac_ar(raw) if code != "eng_Latn" else raw
            rows.append(dict(
                sid=f"{split}-{r['id']}", variety=code, text_raw=raw, text=main,
                n_diac_raw=len(DIAC.findall(raw)), spans=word_spans(main),
            ))

df = pd.DataFrame(rows)
df["n_chars"] = df.text.str.len()
df["n_bytes"] = df.text.map(lambda s: len(s.encode("utf-8")))
df["n_words"] = df.spans.map(len)
assert df.groupby("variety").sid.nunique().nunique() == 1, "varieties not aligned"
os.makedirs(f"{B}/data/prepared", exist_ok=True)
df.to_parquet(f"{B}/data/prepared/flores.parquet")
print(df.groupby("variety")[["n_chars", "n_bytes", "n_words", "n_diac_raw"]].sum())
