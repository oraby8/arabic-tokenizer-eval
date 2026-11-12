"""Build Phase 2 evaluation corpora: fineweb (natural web), wiki, padic (parallel).

- FineWeb-2: test split only for varieties whose train split feeds the reference tokenizers
  (arb, arz, ary, ars). apc/acm/aeb are never used for training, so test+train are both eval.
- Docs shorter than 200 chars are dropped; docs are truncated to 2,000 chars at a word boundary
  so long pages do not dominate the sums.
- PADIC is Buckwalter-transliterated; converted to Arabic script with CAMeL bw2ar.
"""
import glob, os
import pandas as pd
from camel_tools.utils.charmap import CharMapper
from textnorm import normalize, truncate, word_spans, DIAC

B = os.environ["B"]
RAW = f"{B}/data/raw"
N_DOCS, MAX_CHARS, MIN_CHARS = 3000, 2000, 200
TRAIN_FED = {"arb_Arab", "arz_Arab", "ary_Arab", "ars_Arab"}


def finish(rows, name):
    df = pd.DataFrame(rows)
    df["n_diac_raw"] = df.text_raw.map(lambda s: len(DIAC.findall(s)))
    df["spans"] = df.text.map(word_spans)
    df["n_chars"] = df.text.str.len()
    df["n_bytes"] = df.text.map(lambda s: len(s.encode("utf-8")))
    df["n_words"] = df.spans.map(len)
    df = df[df.n_words > 0]
    df.to_parquet(f"{B}/data/prepared/{name}.parquet")
    print(name, df.groupby("variety").agg(docs=("sid", "size"), chars=("n_chars", "sum"), words=("n_words", "sum")).to_string(), sep="\n")


def sample_docs(files, n, seed=0):
    df = pd.concat([pd.read_parquet(f, columns=["text"]) for f in files], ignore_index=True)
    df = df[df.text.str.len() >= MIN_CHARS]
    return df.sample(min(n, len(df)), random_state=seed).text.tolist()


rows = []
for code in ["arb_Arab", "arz_Arab", "ary_Arab", "ars_Arab", "apc_Arab", "acm_Arab", "aeb_Arab"]:
    files = sorted(glob.glob(f"{RAW}/fw2.{code}.test.*.parquet"))
    if code not in TRAIN_FED:
        files += sorted(glob.glob(f"{RAW}/fw2.{code}.train.*.parquet"))
    for i, t in enumerate(sample_docs(files, N_DOCS)):
        raw = truncate(normalize(t, arabic=False), MAX_CHARS)
        rows.append(dict(sid=f"{code}-{i}", variety=code, text_raw=raw, text=normalize(raw)))
finish(rows, "fineweb")

rows = []
for w in ["ar", "arz", "ary"]:
    for i, t in enumerate(sample_docs(sorted(glob.glob(f"{RAW}/wiki.{w}.*.parquet")), N_DOCS)):
        raw = truncate(normalize(t, arabic=False), MAX_CHARS)
        rows.append(dict(sid=f"wiki_{w}-{i}", variety=f"wiki_{w}", text_raw=raw, text=normalize(raw)))
finish(rows, "wiki")

bw2ar = CharMapper.builtin_mapper("bw2ar")
p = pd.read_parquet(f"{RAW}/padic.parquet")
cols = {"MODERN-STANDARD-ARABIC": "padic_msa", "ALGIERS": "padic_algiers", "ANNABA": "padic_annaba",
        "SYRIAN": "padic_syrian", "PALESTINIAN": "padic_palestinian"}
p = p[list(cols)].apply(lambda s: s.str.strip())
p = p[(p != "").all(axis=1)].reset_index(drop=True)
rows = []
for i, r in p.iterrows():
    for c, v in cols.items():
        raw = normalize(bw2ar(r[c]), arabic=False)
        rows.append(dict(sid=f"padic-{i}", variety=v, text_raw=raw, text=normalize(raw)))
finish(rows, "padic")
