"""Phase 3b: does normalization help when the tokenizer is *trained* on normalized text?

Applying normalization only at inference (ablation.py) mostly increases tokens, because the
vocabularies were learned on unnormalized spelling. AraToken normalized the training data too.
Here: same mix corpus as the reference tokenizers, transformed with T["light"] or T["aggressive"],
then BPE / Unigram 64K trained exactly like Ref-*-mix-64K.

Comparison on the same sentences:
  ratio = tokens(normalized tokenizer, normalized text) / tokens(Ref-*-mix-64K, original text)
Usage: python trained_norm.py --data | --job bpe-light | --eval
"""
import argparse, os
import numpy as np
import pandas as pd
from ablation import T
from train_reference import train, OUT as TXT

B = os.environ["B"]
JOBS = {"bpe-light": ("bpe", "light"), "bpe-aggressive": ("bpe", "aggressive"), "unigram-light": ("unigram", "light")}
BASE = {"bpe": "tokenizers/bpe-mix-64k.json", "unigram": "tokenizers/unigram-mix-64k.json"}


def group_of(s):
    if s in ("flores:arb_Arab", "padic:padic_msa", "fineweb:arb_Arab"):
        return "MSA"
    if s.startswith(("flores:", "padic:", "fineweb:")) and s != "flores:ars_Arab":
        return "dialect"
    if s.startswith("tweets:"):
        return "tweets"
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="store_true")
    ap.add_argument("--job")
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    if a.data:
        for norm in ["light", "aggressive"]:
            with open(f"{TXT}/mix.txt") as fin, open(f"{TXT}/mix_{norm}.txt", "w") as fout:
                for line in fin:
                    fout.write(T[norm](line))
            print("built", norm, flush=True)
    if a.job:
        kind, norm = JOBS[a.job]
        train(kind, 64 * 1024, f"{TXT}/mix_{norm}.txt", f"{B}/tokenizers/{kind}-mix{norm}-64k.json")
    if a.eval:
        from transformers import PreTrainedTokenizerFast
        df = pd.read_parquet(f"{B}/data/prepared/phase3.parquet", columns=["set", "sid", "text"])
        df["group"] = df.set.map(group_of)
        df = df[df.group.notna()].reset_index(drop=True)
        rng = np.random.default_rng(0)

        def ntok(path, texts):
            t = PreTrainedTokenizerFast(tokenizer_file=f"{B}/{path}", unk_token="<unk>" if "unigram" in path else None)
            return np.array([len(x) for x in t(texts, add_special_tokens=False)["input_ids"]])

        rows = []
        for job, (kind, norm) in JOBS.items():
            base = ntok(BASE[kind], df.text.tolist())
            new = ntok(f"tokenizers/{kind}-mix{norm}-64k.json", [T[norm](s) for s in df.text])
            base_inf = ntok(BASE[kind], [T[norm](s) for s in df.text])  # same normalization, inference only
            for g in ["MSA", "dialect", "tweets"]:
                m = (df.group == g).values
                ix = rng.integers(0, m.sum(), (1000, m.sum()))
                for label, x in [("trained+inference", new[m]), ("inference only", base_inf[m])]:
                    bs = x[ix].sum(1) / base[m][ix].sum(1)
                    rows.append(dict(tokenizer=f"{kind}-64K", normalization=norm, setting=label, group=g,
                                     pct_change=100 * (x.sum() / base[m].sum() - 1),
                                     lo=100 * (np.percentile(bs, 2.5) - 1), hi=100 * (np.percentile(bs, 97.5) - 1)))
        R = pd.DataFrame(rows)
        os.makedirs(f"{B}/results/phase3", exist_ok=True)
        R.to_csv(f"{B}/results/phase3/trained_normalization.csv", index=False)
        print(R.round(2).to_string(index=False))
