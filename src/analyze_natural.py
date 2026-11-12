"""Phase 2 analysis: natural corpora (fineweb, wiki) and the parallel PADIC set.

For every tokenizer x corpus x variety:
  chars/token, fertility, continued-word share, UNK + byte-fallback rate, with 95% bootstrap CIs.
Dialect penalty = chars/token(MSA of the same corpus) / chars/token(variety) (>1 = dialect costs more).
  - fineweb/wiki: independent bootstrap over docs (different texts per variety)
  - padic: paired bootstrap over sentences + Wilcoxon on per-sentence tokens (same meaning)
Dialectness filter (main condition): MSA docs need ALDi < 0.1, dialect docs ALDi >= 0.3.
The unfiltered numbers are kept in the long table (subset="all").

Outputs: $B/results/phase2/<tokset>/
"""
import json, os
from collections import Counter
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from registry import TOKENIZERS

B = os.environ["B"]
tokset = os.environ.get("TOKSET", "core")
OUT = f"{B}/results/phase2/{tokset}"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)
N_BOOT = 1000

CORPORA = {
    "fineweb": dict(msa="arb_Arab", parallel=False, names={
        "arb_Arab": "MSA", "arz_Arab": "Egyptian", "ary_Arab": "Moroccan", "ars_Arab": "Najdi",
        "apc_Arab": "Levantine", "acm_Arab": "Mesopotamian", "aeb_Arab": "Tunisian"}),
    "wiki": dict(msa="wiki_ar", parallel=False, names={"wiki_ar": "MSA", "wiki_arz": "Egyptian", "wiki_ary": "Moroccan"}),
    "padic": dict(msa="padic_msa", parallel=True, names={
        "padic_msa": "MSA", "padic_algiers": "Algiers", "padic_annaba": "Annaba",
        "padic_syrian": "Syrian", "padic_palestinian": "Palestinian"}),
}
UNSEEN = {"apc_Arab", "acm_Arab", "aeb_Arab"}  # never used to train reference tokenizers

order = [c["name"] for c in TOKENIZERS if tokset == "all" or c.get("core")]
group = {c["name"]: c["group"] for c in TOKENIZERS}


def templated_flags(texts, n=8, min_df=5, frac=0.01):
    """True for docs sharing an n-word sequence with many other docs (bot templates, boilerplate).
    Dialect Wikipedias are ~50% templated (e.g. auto-generated galaxy stubs in arz)."""
    grams = []
    for t in texts:
        w = t.split()
        grams.append({" ".join(w[i:i + n]) for i in range(max(len(w) - n + 1, 0))})
    df = Counter(g for s in grams for g in s)
    thr = max(min_df, frac * len(texts))
    return np.array([any(df[g] >= thr for g in s) for s in grams])


# tokenizers are aliases only if they give identical ids on *every* corpus
metas = {c: json.load(open(f"{B}/results/{c}/tokenizers.json")) for c in CORPORA}
sig = {n: tuple(metas[c][n]["output_hash"] for c in CORPORA) for n in order if all(n in metas[c] for c in CORPORA)}
canon, aliases_all, seen = [], {}, {}
for n, s in sig.items():
    if s in seen:
        aliases_all.setdefault(seen[s], set()).add(n)
    else:
        seen[s] = n
        canon.append(n)

long_rows = []
for corpus, cfg in CORPORA.items():
    R = f"{B}/results/{corpus}"
    meta = metas[corpus]
    docs = pd.read_parquet(f"{B}/data/prepared/{corpus}.parquet", columns=["sid", "variety", "text", "n_chars", "n_words"])
    docs = docs.merge(pd.read_parquet(f"{R}/aldi.parquet")[["sid", "variety", "aldi"]], on=["sid", "variety"], how="left")
    docs["templated"] = False
    for v, g in docs.groupby("variety"):
        docs.loc[g.index, "templated"] = templated_flags(g.text.tolist())
    docs = docs.drop(columns="text")
    keep = np.where(docs.variety == cfg["msa"], docs.aldi < 0.1, docs.aldi >= 0.3) & ~docs.templated.values
    docs["filtered"] = keep if not cfg["parallel"] else True  # PADIC is parallel: keep all rows aligned
    for name in canon:
        d = pd.read_parquet(f"{R}/per_sentence/{name.replace('/', '_')}.parquet").merge(docs, on=["sid", "variety"])
        for subset in ["filtered", "all"]:
            ds = d[d.filtered] if subset == "filtered" else d
            msa = ds[ds.variety == cfg["msa"]]
            if cfg["parallel"]:
                msa = msa.set_index("sid")
            for v, vname in cfg["names"].items():
                x = ds[ds.variety == v]
                if len(x) < 30:
                    continue
                cpt = x.n_chars.sum() / x.n_tokens.sum()
                ix = rng.integers(0, len(x), (N_BOOT, len(x)))
                cpt_bs = x.n_chars.values[ix].sum(1) / x.n_tokens.values[ix].sum(1)
                row = dict(corpus=corpus, subset=subset, tokenizer=name, group=group[name], variety=v, variety_name=vname,
                           n_docs=len(x), aldi_mean=x.aldi.mean(), templated_share=x.templated.mean(), chars_per_token=cpt,
                           cpt_lo=np.percentile(cpt_bs, 2.5), cpt_hi=np.percentile(cpt_bs, 97.5),
                           fertility=x.n_tokens.sum() / x.n_words.sum(),
                           continued_share=x.n_continued.sum() / x.n_words.sum(),
                           unk_or_fallback_rate=(x.n_unk.sum() + x.n_byte_fallback.sum()) / x.n_tokens.sum(),
                           unseen_by_reference=v in UNSEEN)
                if v != cfg["msa"]:
                    if cfg["parallel"]:
                        xx = x.set_index("sid").loc[msa.index]
                        jx = rng.integers(0, len(xx), (N_BOOT, len(xx)))
                        pen_bs = (msa.n_chars.values[jx].sum(1) / msa.n_tokens.values[jx].sum(1)) / (xx.n_chars.values[jx].sum(1) / xx.n_tokens.values[jx].sum(1))
                        row["token_ratio_same_meaning"] = xx.n_tokens.sum() / msa.n_tokens.sum()
                        row["wilcoxon_p"] = wilcoxon(xx.n_tokens.values, msa.n_tokens.values).pvalue
                    else:
                        jm = rng.integers(0, len(msa), (N_BOOT, len(msa)))
                        pen_bs = (msa.n_chars.values[jm].sum(1) / msa.n_tokens.values[jm].sum(1)) / cpt_bs
                    msa_cpt = msa.n_chars.sum() / msa.n_tokens.sum()
                    row.update(len_adj_penalty=msa_cpt / cpt, penalty_lo=np.percentile(pen_bs, 2.5), penalty_hi=np.percentile(pen_bs, 97.5))
                long_rows.append(row)
        m = meta[name]
        long_rows.append(dict(corpus=corpus, subset="vocab", tokenizer=name, group=group[name],
                              arabic_vocab=m["arabic_vocab"], arabic_vocab_used=m.get("arabic_vocab_used")))

L = pd.DataFrame(long_rows)
L.to_csv(f"{OUT}/metrics_long.csv", index=False)
F = L[L.subset == "filtered"]

# ---- penalty matrix: tokenizer x (corpus: variety), main condition ----
pen = F[F.len_adj_penalty.notna()].assign(col=lambda d: d.corpus + ": " + d.variety_name)
pen_tbl = pen.pivot(index="tokenizer", columns="col", values="len_adj_penalty")
msa_cpt = F[(F.corpus == "fineweb") & (F.variety == "arb_Arab")].set_index("tokenizer").chars_per_token
pen_tbl = pen_tbl.loc[msa_cpt.sort_values(ascending=False).index]
pen_tbl.to_csv(f"{OUT}/dialect_penalty_all_corpora.csv")

# ---- vocab utilisation on natural MSA+dialect web text ----
V = L[(L.subset == "vocab") & (L.corpus == "fineweb")].set_index("tokenizer")
summary = pd.DataFrame({
    "group": pd.Series(group),
    "MSA_web_chars/tok": msa_cpt,
    "MSA_wiki_chars/tok": F[(F.corpus == "wiki") & (F.variety == "wiki_ar")].set_index("tokenizer").chars_per_token,
    "MSA_padic_chars/tok": F[(F.corpus == "padic") & (F.variety == "padic_msa")].set_index("tokenizer").chars_per_token,
    "penalty_web_seen": pen[(pen.corpus == "fineweb") & ~pen.unseen_by_reference.astype(bool)].groupby("tokenizer").len_adj_penalty.mean(),
    "penalty_web_unseen": pen[(pen.corpus == "fineweb") & pen.unseen_by_reference.astype(bool)].groupby("tokenizer").len_adj_penalty.mean(),
    "penalty_wiki": pen[pen.corpus == "wiki"].groupby("tokenizer").len_adj_penalty.mean(),
    "penalty_padic": pen[pen.corpus == "padic"].groupby("tokenizer").len_adj_penalty.mean(),
    "penalty_all_mean": pen.groupby("tokenizer").len_adj_penalty.mean(),
    "worst": pen.loc[pen.groupby("tokenizer").len_adj_penalty.idxmax()].set_index("tokenizer").col,
    "worst_penalty": pen.groupby("tokenizer").len_adj_penalty.max(),
    "arabic_vocab": V.arabic_vocab, "arabic_vocab_used_%": 100 * V.arabic_vocab_used / V.arabic_vocab,
    "aliases": pd.Series({k: ", ".join(sorted(v)) for k, v in aliases_all.items()}),
}).loc[msa_cpt.sort_values(ascending=False).index]
summary.to_csv(f"{OUT}/summary.csv")

# ---- ALDi of the eval subsets ----
aldi_tbl = F[F.tokenizer == F.tokenizer.iloc[0]].groupby(["corpus", "variety_name"])[["n_docs", "aldi_mean"]].first()
A = L[(L.subset == "all") & (L.tokenizer == F.tokenizer.iloc[0])].groupby(["corpus", "variety_name"])[["n_docs", "aldi_mean", "templated_share"]].first()
aldi_tbl = aldi_tbl.join(A, rsuffix="_unfiltered")
aldi_tbl.to_csv(f"{OUT}/eval_subsets.csv")

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(aldi_tbl.round(3).to_string(), "\n")
print(summary.drop(columns=["aliases"]).round(3).to_string(), "\n")
print("ALIASES:", {k: sorted(v) for k, v in aliases_all.items()})
print(pen_tbl.round(3).to_string())

# ---- figures ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
os.makedirs(f"{OUT}/figures", exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=0.85)
plt.figure(figsize=(13, 10))
sns.heatmap(pen_tbl, annot=True, fmt=".2f", cmap="vlag", center=1.0, vmin=0.8, vmax=1.6,
            cbar_kws={"label": "MSA chars/token ÷ dialect chars/token (>1 = dialect costs more)"})
plt.title("Dialect penalty per character on natural (FineWeb-2, Wikipedia) and parallel (PADIC) text\n"
          "dialectal docs only (ALDi ≥ 0.3), bot templates removed; rows sorted by MSA efficiency")
plt.xlabel(""); plt.ylabel(""); plt.tight_layout(); plt.savefig(f"{OUT}/figures/dialect_penalty_all_corpora.png", dpi=160); plt.close()

dial_cpt = F[F.len_adj_penalty.notna()].groupby("tokenizer").chars_per_token.mean()
pal = {"global": "#4C72B0", "arabic": "#DD8452", "reference": "#55A868"}
plt.figure(figsize=(9, 7))
for t in summary.index:
    plt.scatter(summary.loc[t, "MSA_web_chars/tok"], dial_cpt[t], c=pal[group[t]], s=50, edgecolor="k", lw=.4)
    plt.annotate(t, (summary.loc[t, "MSA_web_chars/tok"], dial_cpt[t]), fontsize=7, xytext=(4, 2), textcoords="offset points")
lim = [2.0, 4.9]
plt.plot(lim, lim, "k--", lw=.8, label="dialect = MSA (no penalty)")
plt.xlim(lim); plt.ylim(2.0, 4.2); plt.legend(loc="upper left")
plt.xlabel("MSA chars/token (FineWeb-2 web)"); plt.ylabel("dialect chars/token (mean over all dialect sets)")
plt.title("MSA vs dialect efficiency (blue = global, orange = Arabic-centric, green = reference)")
plt.tight_layout(); plt.savefig(f"{OUT}/figures/msa_vs_dialect_efficiency.png", dpi=160); plt.close()
