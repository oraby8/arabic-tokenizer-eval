"""Phase 4a: morphological alignment of token boundaries on held-out evaluation text.

Gold boundaries: CAMeL MLE segmentation (same segmenter as used for MorphBPE training, so the
morph tokenizers have a built-in advantage on this metric; compare them with that in mind).
  MSA sets  : FLORES MSA + PADIC MSA + FineWeb-2 MSA (ALDi < 0.1), msa db
  EGY sets  : FineWeb-2 Egyptian (ALDi >= 0.3) + PADIC-free; egy db (morpheme scheme only)
For each word with >= 1 gold boundary, token boundaries strictly inside the word are compared
with gold boundaries:  precision = hits / token boundaries, recall = hits / gold boundaries.
Also: share of words where every gold boundary is a token boundary ("boundary-respecting").
"""
import os, re
from multiprocessing import Pool
import numpy as np
import pandas as pd
from morph_segment import WORD, seg_init, seg_chunk
from count_tokens import load_tokenizer
from registry import TOKENIZERS

B = os.environ["B"]
OUT = f"{B}/results/phase4"
os.makedirs(OUT, exist_ok=True)


def load_sets():
    P = f"{B}/data/prepared"
    rows = []
    f = pd.read_parquet(f"{P}/flores.parquet", columns=["sid", "variety", "text"])
    rows += [("msa", t) for t in f[f.variety == "arb_Arab"].text]
    p = pd.read_parquet(f"{P}/padic.parquet", columns=["variety", "text"])
    rows += [("msa", t) for t in p[p.variety == "padic_msa"].text]
    w = pd.read_parquet(f"{P}/fineweb.parquet", columns=["sid", "variety", "text"]).merge(
        pd.read_parquet(f"{B}/results/fineweb/aldi.parquet"), on=["sid", "variety"])
    rows += [("msa", t) for t in w[(w.variety == "arb_Arab") & (w.aldi < 0.1)].text]
    rows += [("egy", t) for t in w[(w.variety == "arz_Arab") & (w.aldi >= 0.3)].text]
    return pd.DataFrame(rows, columns=["group", "text"])


def gold_for(group, words):
    seg_init(group)
    out = {"clitic": {}, "morpheme": {}}
    for i in range(0, len(words), 2000):
        r = seg_chunk(words[i:i + 2000])
        for s in out:
            out[s].update(r[s])
    return out


def score(cfg, df, gold):
    tok = load_tokenizer(cfg)
    acc = {}
    for text, g in zip(df.text, df.group):
        _, offs = tok.encode_offsets(text)
        starts = {s for s, e in offs if e > s}
        for m in WORD.finditer(text):
            ws, we, w = m.start(), m.end(), m.group(0)
            inner = {x - ws for x in starts if ws < x < we}
            for scheme in ("clitic", "morpheme"):
                if g == "egy" and scheme == "clitic":
                    continue
                gb = set(gold[g][scheme].get(w, ()))
                if not gb:
                    continue
                a = acc.setdefault((g, scheme), [0, 0, 0, 0, 0])
                a[0] += len(inner & gb); a[1] += len(inner); a[2] += len(gb); a[3] += gb <= inner; a[4] += 1
    rows = []
    for (g, scheme), (hit, ntok, ngold, full, nw) in acc.items():
        p, r = hit / max(ntok, 1), hit / max(ngold, 1)
        rows.append(dict(tokenizer=cfg["name"], group=g, gold=scheme, precision=p, recall=r,
                         f1=2 * p * r / max(p + r, 1e-9), words_fully_respected=full / nw, n_words=nw))
    return rows


if __name__ == "__main__":
    df = load_sets()
    gold = {}
    for g in ["msa", "egy"]:
        words = sorted({w for t in df[df.group == g].text for w in WORD.findall(t)})
        gold[g] = gold_for(g, words)
        print(g, len(words), "eval word types;", {s: len(v) for s, v in gold[g].items()}, "segmented", flush=True)
    canon = pd.read_csv(f"{B}/results/phase2/core/summary.csv", index_col=0).index.tolist()
    cfgs = [c for c in TOKENIZERS if c["name"] in canon or c.get("phase") == 4]
    cfgs = [c for c in cfgs if c["kind"] != "tiktoken"]
    with Pool(24) as p:
        res = p.starmap(score, [(c, df, gold) for c in cfgs])
    R = pd.DataFrame([r for rr in res for r in rr])
    R.to_csv(f"{OUT}/morph_alignment.csv", index=False)
    piv = R.pivot_table(index="tokenizer", columns=["group", "gold"], values=["f1", "words_fully_respected"])
    print(piv.sort_values(("f1", "msa", "morpheme"), ascending=False).round(3).to_string())
