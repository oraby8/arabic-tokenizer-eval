"""Phase 3 ablations: how much each text transform changes token counts, per tokenizer.

ratio = tokens(transformed text) / tokens(base text) on the same sentences (< 1 = fewer tokens).
95% CI: paired bootstrap over sentences. changed_% = share of sentences the transform alters.

Normalization (on the main NFC + dediac text):
  alif     أ إ آ ٱ -> ا          yaa   ى -> ي            taa   ة -> ه
  tatweel  remove ـ              hamza ؤ -> و, ئ -> ي     digits ٠-٩ -> 0-9
  light      = alif + yaa + tatweel + digits   (AraToken-style)
  aggressive = light + taa + hamza
Social cleaning (tweets): urls, mentions, hashtag marks, emoji, elongation (3+ repeats -> 2), clean_all.
Diacritics (tashkeela): diac_asis / diac_nfc vs undiacritized base.
"""
import json, os, re
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from count_tokens import TT, HF
from registry import TOKENIZERS

B = os.environ["B"]
OUT = f"{B}/results/phase3"
os.makedirs(OUT, exist_ok=True)

ALIF, DIGITS = str.maketrans("أإآٱ", "اااا"), str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
HAMZA = str.maketrans("ؤئ", "وي")
EMOJI = re.compile("[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F900-\U0001F9FF️‍]")
URL, MENTION = re.compile(r"https?://\S+|www\.\S+"), re.compile(r"@\w+")
ELONG = re.compile(r"(.)\1{2,}")
SPACES = re.compile(r"\s{2,}")


def sp(s):
    return SPACES.sub(" ", s).strip()


T = {
    "alif": lambda s: s.translate(ALIF),
    "yaa": lambda s: s.replace("ى", "ي"),
    "taa": lambda s: s.replace("ة", "ه"),
    "tatweel": lambda s: s.replace("ـ", ""),
    "hamza": lambda s: s.translate(HAMZA),
    "digits": lambda s: s.translate(DIGITS),
}
T["light"] = lambda s: T["digits"](T["tatweel"](T["yaa"](T["alif"](s))))
T["aggressive"] = lambda s: T["hamza"](T["taa"](T["light"](s)))
SOCIAL = {
    "urls": lambda s: sp(URL.sub(" ", s)),
    "mentions": lambda s: sp(MENTION.sub(" ", s)),
    "hashtags": lambda s: sp(re.sub(r"#(\w+)", lambda m: m.group(1).replace("_", " "), s)),
    "emoji": lambda s: sp(EMOJI.sub(" ", s)),
    "elongation": lambda s: ELONG.sub(r"\1\1", s),
}
SOCIAL["clean_all"] = lambda s: T["tatweel"](SOCIAL["elongation"](SOCIAL["emoji"](SOCIAL["hashtags"](SOCIAL["mentions"](SOCIAL["urls"](s))))))
NORM_SETS = ("flores:", "padic:", "fineweb:", "tweets:")


def variants(df):
    """yield (transform, set_mask, list_of_texts) — the base is always df.text."""
    norm = df.set.str.startswith(NORM_SETS)
    for name, fn in T.items():
        yield name, norm, [fn(s) for s in df.text[norm]]
    tw = df.set.str.startswith("tweets:")
    for name, fn in SOCIAL.items():
        yield name, tw, [fn(s) for s in df.text[tw]]
    tk = df.set == "tashkeela"
    for col in ["diac_asis", "diac_nfc"]:
        yield col, tk, df[col][tk].tolist()


def count(tok, texts):
    if isinstance(tok, TT):
        return np.array([len(x) for x in tok.enc.encode_ordinary_batch(texts, num_threads=4)])
    return np.array([len(x) for x in tok.tok(texts, add_special_tokens=False)["input_ids"]])


def run(cfg):
    tok = TT(cfg["repo"]) if cfg["kind"] == "tiktoken" else HF(
        f"{B}/{cfg['repo']}" if cfg["kind"] == "local" else cfg["repo"], local=cfg["kind"] == "local")
    df = pd.read_parquet(f"{B}/data/prepared/phase3.parquet")
    base = count(tok, df.text.tolist())
    out = [pd.DataFrame({"tokenizer": cfg["name"], "set": df.set, "sid": df.sid, "transform": "base", "n": base, "changed": False})]
    for name, mask, texts in variants(df):
        out.append(pd.DataFrame({"tokenizer": cfg["name"], "set": df.set[mask].values, "sid": df.sid[mask].values,
                                 "transform": name, "n": count(tok, texts),
                                 "changed": np.array(texts) != df.text[mask].values}))
    pd.concat(out).to_parquet(f"{OUT}/per_sentence_{cfg['name']}.parquet")
    return cfg["name"]


def analyze(names):
    rng = np.random.default_rng(0)
    rows = []
    for n in names:
        d = pd.read_parquet(f"{OUT}/per_sentence_{n}.parquet")
        b = d[d["transform"] == "base"].set_index(["set", "sid"]).n
        for (s, t), g in d[d["transform"] != "base"].groupby(["set", "transform"]):
            x = g.set_index(["set", "sid"]).n.values
            y = b.loc[list(zip(g.set, g.sid))].values
            ix = rng.integers(0, len(x), (1000, len(x)))
            bs = x[ix].sum(1) / y[ix].sum(1)
            rows.append(dict(tokenizer=n, set=s, transform=t, ratio=x.sum() / y.sum(),
                             lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                             changed_pct=100 * g.changed.mean(), n_sent=len(x)))
    L = pd.DataFrame(rows)
    L.to_csv(f"{OUT}/ablation_long.csv", index=False)
    return L


if __name__ == "__main__":
    # canonical tokenizers = the deduplicated Phase 2 core set (aliases give identical counts)
    canon = pd.read_csv(f"{B}/results/phase2/core/summary.csv", index_col=0).index.tolist()
    cfgs = [c for c in TOKENIZERS if c["name"] in canon]
    with ProcessPoolExecutor(24) as ex:
        for f in as_completed([ex.submit(run, c) for c in cfgs]):
            print("done", f.result(), flush=True)
    L = analyze([c["name"] for c in cfgs])
    print(L.groupby("transform").ratio.describe()[["mean", "min", "max"]].round(3).to_string())
