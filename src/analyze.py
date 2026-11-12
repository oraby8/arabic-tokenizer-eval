"""Aggregate per-sentence counts into metrics, CIs, tests, tables and plots."""
import glob, json, os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from registry import VARIETIES, TOKENIZERS

B = os.environ["B"]
corpus = os.environ.get("CORPUS", "flores")
tokset = os.environ.get("TOKSET", "core")  # core = user's selected list, all = everything counted
R = f"{B}/results/{corpus}"
OUT = f"{R}/{tokset}"
os.makedirs(f"{OUT}/figures", exist_ok=True)
rng = np.random.default_rng(0)
N_BOOT = 2000

meta = json.load(open(f"{R}/tokenizers.json"))
order = [c["name"] for c in TOKENIZERS if tokset == "all" or c.get("core")]
groups_by_name = {c["name"]: c["group"] for c in TOKENIZERS}
meta = {n: {**meta[n], "group": groups_by_name[n]} for n in order if n in meta}
corpus_df = pd.read_parquet(f"{B}/data/prepared/{corpus}.parquet")
aldi = pd.read_parquet(f"{R}/aldi.parquet").groupby("variety").aldi.mean()

# ---- dedup: tokenizers producing identical ids on the whole corpus are one tokenizer ----
groups = {}
for name, m in meta.items():
    groups.setdefault(m["output_hash"], []).append(name)
canonical, aliases = {}, {}
for h, names in groups.items():
    names = sorted(names, key=order.index)
    canonical[names[0]] = h
    aliases[names[0]] = names[1:]

frames = []
for name in canonical:
    d = pd.read_parquet(f"{R}/per_sentence/{name.replace('/', '_')}.parquet")
    d["tokenizer"] = name
    frames.append(d)
# tokenizer-independent baselines
for unit, col in [("[bytes]", "n_bytes"), ("[chars]", "n_chars"), ("[words]", "n_words")]:
    frames.append(corpus_df[["sid", "variety"]].assign(n_tokens=corpus_df[col], n_unk=0, n_byte_fallback=0,
                                                         n_continued=0, roundtrip_ok=True, tokenizer=unit))
tok = pd.concat(frames).merge(corpus_df[["sid", "variety", "n_chars", "n_bytes", "n_words"]], on=["sid", "variety"])

# ---- core metrics ----
g = tok.groupby(["tokenizer", "variety"])[["n_tokens", "n_chars", "n_bytes", "n_words", "n_continued", "n_unk", "n_byte_fallback"]].sum()
g["fertility"] = g.n_tokens / g.n_words
g["chars_per_token"] = g.n_chars / g.n_tokens
g["bytes_per_token"] = g.n_bytes / g.n_tokens
g["continued_share"] = g.n_continued / g.n_words
g["unk_rate"] = g.n_unk / g.n_tokens
g["tokens_per_1k_words"] = 1000 * g.fertility
g["words_in_128k_ctx"] = 128_000 / g.fertility
g = g.reset_index()

# ---- paired bootstrap for ratios of sums (sentences are shared across varieties) ----
piv = tok.pivot_table(index=["tokenizer", "sid"], columns="variety", values="n_tokens")
sids = corpus_df.sid.unique()
boot_idx = rng.integers(0, len(sids), size=(N_BOOT, len(sids)))


def ratio_ci(mat_num, mat_den):
    point = mat_num.sum() / mat_den.sum()
    bs = mat_num[boot_idx].sum(1) / mat_den[boot_idx].sum(1)
    return point, *np.percentile(bs, [2.5, 97.5])


rows = []
for t in piv.index.get_level_values(0).unique():
    m = piv.loc[t].reindex(sids)
    for v in VARIETIES:
        if v == "eng_Latn":
            continue
        p, lo, hi = ratio_ci(m[v].values, m["eng_Latn"].values)
        d, dlo, dhi = ratio_ci(m[v].values, m["arb_Arab"].values)
        pval = np.nan if v == "arb_Arab" or t.startswith("[") else wilcoxon(m[v].values, m["arb_Arab"].values).pvalue
        rows.append(dict(tokenizer=t, variety=v, parity_vs_en=p, parity_lo=lo, parity_hi=hi,
                         dialect_penalty=d, penalty_lo=dlo, penalty_hi=dhi, wilcoxon_p_vs_msa=pval))
ratios = pd.DataFrame(rows)
full = g.merge(ratios, on=["tokenizer", "variety"], how="left")
full["variety_name"] = full.variety.map(VARIETIES)
full["aldi_mean"] = full.variety.map(aldi)
full["group"] = full.tokenizer.map(lambda t: meta[t]["group"] if t in meta else "baseline")
full["vocab_size"] = full.tokenizer.map(lambda t: meta[t]["vocab_size"] if t in meta else np.nan)
full["arabic_vocab_share"] = full.tokenizer.map(lambda t: meta[t]["arabic_vocab_share"] if t in meta else np.nan)
# reference tokenizers are trained on Arabic only, so English is shredded and parity vs English is meaningless
full.loc[full.group == "reference", ["parity_vs_en", "parity_lo", "parity_hi"]] = np.nan
full.to_csv(f"{OUT}/metrics_long.csv", index=False)

# ---- tables ----
var_order = ["arb_Arab"] + sorted([v for v in VARIETIES if v not in ("eng_Latn", "arb_Arab")], key=lambda v: aldi[v])
names = {v: VARIETIES[v] for v in VARIETIES}
msa = full[full.variety == "arb_Arab"].set_index("tokenizer")
eng = full[full.variety == "eng_Latn"].set_index("tokenizer")
dial = full[~full.variety.isin(["eng_Latn", "arb_Arab", "ars_Arab"])]  # Najdi excluded: ALDi ~ MSA
summary = pd.DataFrame({
    "group": msa.group, "vocab": msa.vocab_size, "ar_vocab_%": 100 * msa.arabic_vocab_share,
    "fert_EN": eng.fertility, "fert_MSA": msa.fertility, "chars/tok_MSA": msa.chars_per_token,
    "parity_MSA": msa.parity_vs_en,
    "parity_MSA_CI": msa.apply(lambda r: f"[{r.parity_lo:.2f}, {r.parity_hi:.2f}]" if pd.notna(r.parity_lo) else "", axis=1),
    "parity_dialects_mean": dial.groupby("tokenizer").parity_vs_en.mean(),
    "penalty_dialects_mean": dial.groupby("tokenizer").dialect_penalty.mean(),
    "penalty_max": dial.groupby("tokenizer").dialect_penalty.max(),
    "worst_dialect": dial.loc[dial.groupby("tokenizer").dialect_penalty.idxmax()].set_index("tokenizer").variety_name,
    # penalty after removing the length difference: >1 means the tokenizer handles dialect text worse per character
    "len_adj_penalty_mean": (dial.tokenizer.map(msa.chars_per_token) / dial.chars_per_token).groupby(dial.tokenizer).mean(),
    "cont_words_%_MSA": 100 * msa.continued_share,
    "MSA_words_in_128k": msa.words_in_128k_ctx,
    "aliases": pd.Series({k: ", ".join(v) for k, v in aliases.items()}),
}).sort_values(["parity_MSA", "chars/tok_MSA"], ascending=[True, False])
summary.to_csv(f"{OUT}/summary.csv")
parity_tbl = full.pivot(index="tokenizer", columns="variety", values="parity_vs_en")[var_order].rename(columns=names)
parity_tbl = parity_tbl.loc[summary.index]
penalty_tbl = full.pivot(index="tokenizer", columns="variety", values="dialect_penalty")[var_order[1:]].rename(columns=names).loc[summary.index]
fert_tbl = full.pivot(index="tokenizer", columns="variety", values="fertility")[["eng_Latn"] + var_order].rename(columns=names).loc[summary.index]
parity_tbl.to_csv(f"{OUT}/parity_vs_english.csv")
penalty_tbl.to_csv(f"{OUT}/dialect_penalty_vs_msa.csv")
fert_tbl.to_csv(f"{OUT}/fertility.csv")

# ---- RQ4: what explains MSA efficiency? ----
# real LLM tokenizers only: the reference tokenizers are mostly Arabic by design and would inflate the correlation
real = summary[summary.group.isin(["arabic", "global"])].dropna(subset=["vocab"])
rq4 = {
    "spearman_charsPerTok_vs_logVocab": spearmanr(np.log(real.vocab), real["chars/tok_MSA"]),
    "spearman_charsPerTok_vs_arabicShare": spearmanr(real["ar_vocab_%"], real["chars/tok_MSA"]),
    "spearman_charsPerTok_vs_arabicCount": spearmanr(real["ar_vocab_%"] * real.vocab, real["chars/tok_MSA"]),
}
X = np.column_stack([np.ones(len(real)), np.log(real.vocab), np.log(real["ar_vocab_%"] * real.vocab / 100)])
coef, res, *_ = np.linalg.lstsq(X, real["chars/tok_MSA"].values, rcond=None)
pred = X @ coef
r2 = 1 - ((real["chars/tok_MSA"] - pred) ** 2).sum() / ((real["chars/tok_MSA"] - real["chars/tok_MSA"].mean()) ** 2).sum()
with open(f"{OUT}/rq4.txt", "w") as f:
    for k, v in rq4.items():
        f.write(f"{k}: rho={v.statistic:.3f} p={v.pvalue:.4f}\n")
    f.write(f"OLS chars/tok ~ log(vocab) + log(#arabic tokens): coef={np.round(coef, 3).tolist()} R2={r2:.3f} n={len(real)}\n")

# ---- figures ----
sns.set_theme(style="whitegrid", font_scale=0.9)
plt.figure(figsize=(11, 10))
sns.heatmap(parity_tbl, annot=True, fmt=".2f", cmap="rocket_r", vmin=0.8, vmax=max(2.0, np.ceil(parity_tbl.max().max() * 2) / 2), cbar_kws={"label": "tokens(Arabic) / tokens(English)"})
plt.title(f"Token parity vs English on FLORES+ (1.0 = same as English)\nvarieties ordered by dialectness (ALDi)")
plt.ylabel(""); plt.xlabel(""); plt.tight_layout(); plt.savefig(f"{OUT}/figures/parity_heatmap.png", dpi=160); plt.close()

plt.figure(figsize=(11, 10))
sns.heatmap(penalty_tbl, annot=True, fmt=".2f", cmap="vlag", center=1.0, vmin=0.85, vmax=1.25, cbar_kws={"label": "tokens(dialect) / tokens(MSA)"})
plt.title("Dialect penalty: tokens for the same sentence vs MSA"); plt.ylabel(""); plt.xlabel("")
plt.tight_layout(); plt.savefig(f"{OUT}/figures/dialect_penalty_heatmap.png", dpi=160); plt.close()

plt.figure(figsize=(9, 6))
pal = {"global": "#4C72B0", "arabic": "#DD8452", "reference": "#55A868"}
for _, r in real.iterrows():
    plt.scatter(r.vocab, r["chars/tok_MSA"], s=30 + 6 * r["ar_vocab_%"], c=pal[r.group], alpha=.8, edgecolor="k", lw=.4)
    plt.annotate(r.name, (r.vocab, r["chars/tok_MSA"]), fontsize=7, xytext=(4, 2), textcoords="offset points")
plt.xscale("log"); plt.xlabel("vocabulary size (log)"); plt.ylabel("MSA characters per token (higher = more efficient)")
plt.title("MSA compression vs vocabulary size (size = % Arabic tokens; blue = global, orange = Arabic-centric)")
plt.tight_layout(); plt.savefig(f"{OUT}/figures/vocab_vs_efficiency.png", dpi=160); plt.close()

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(summary.drop(columns=["aliases"]).round(2).to_string())
adj = full[full.variety != "eng_Latn"].pivot(index="tokenizer", columns="variety", values="chars_per_token")
adj = (adj[["arb_Arab"]].values / adj[var_order[1:]]).rename(columns=names).loc[summary.index]
adj.to_csv(f"{OUT}/dialect_penalty_length_adjusted.csv")
print("\nLength-adjusted dialect penalty (MSA chars/token ÷ dialect chars/token):")
print(adj.round(3).to_string())
print("\nALIASES (identical outputs):", {k: v for k, v in aliases.items() if v})
print(open(f"{OUT}/rq4.txt").read())
print("UNK/byte-fallback:", g[g.n_unk + g.n_byte_fallback > 0].groupby("tokenizer")[["n_unk", "n_byte_fallback"]].sum().to_dict("index"))
