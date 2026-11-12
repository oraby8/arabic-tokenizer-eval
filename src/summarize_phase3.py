"""Pool Phase 3 ablation counts into groups and draw the summary tables/figures.

Groups (same sentences before/after each transform; ratio of summed tokens, paired bootstrap CI):
  MSA      flores:arb_Arab + padic:padic_msa + fineweb:arb_Arab
  dialect  all other flores (Najdi excluded, ALDi ~ MSA), padic dialects, fineweb dialects
  tweets   ArSarcasm-v2, all labels
  tashkeela  diacritized Classical/MSA text
"""
import glob, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

B = os.environ["B"]
OUT = f"{B}/results/phase3"
rng = np.random.default_rng(0)


def group_of(s):
    if s.startswith("tweets:"):
        return "tweets"
    if s == "tashkeela":
        return "tashkeela"
    if s in ("flores:arb_Arab", "padic:padic_msa", "fineweb:arb_Arab"):
        return "MSA"
    if s == "flores:ars_Arab":
        return None
    return "dialect"


rows = []
for f in sorted(glob.glob(f"{OUT}/per_sentence_*.parquet")):
    d = pd.read_parquet(f)
    name = d.tokenizer.iloc[0]
    d["group"] = d.set.map(group_of)
    d = d[d.group.notna()]
    base = d[d["transform"] == "base"].set_index(["set", "sid"]).n
    for (g, t), x in d[d["transform"] != "base"].groupby(["group", "transform"]):
        num = x.n.values
        den = base.loc[list(zip(x.set, x.sid))].values
        ix = rng.integers(0, len(num), (1000, len(num)))
        bs = num[ix].sum(1) / den[ix].sum(1)
        rows.append(dict(tokenizer=name, group=g, transform=t, pct_change=100 * (num.sum() / den.sum() - 1),
                         lo=100 * (np.percentile(bs, 2.5) - 1), hi=100 * (np.percentile(bs, 97.5) - 1),
                         changed_pct=100 * x.changed.mean(), n_sent=len(num)))
S = pd.DataFrame(rows)
S.to_csv(f"{OUT}/ablation_grouped.csv", index=False)

norm_cols = ["alif", "yaa", "taa", "tatweel", "hamza", "digits", "light", "aggressive"]
soc_cols = ["urls", "mentions", "hashtags", "emoji", "elongation", "clean_all"]
order = pd.read_csv(f"{B}/results/phase2/core/summary.csv", index_col=0).index.tolist()
tables = {
    "norm_MSA": S[(S.group == "MSA") & S["transform"].isin(norm_cols)].pivot(index="tokenizer", columns="transform", values="pct_change")[norm_cols],
    "norm_dialect": S[(S.group == "dialect") & S["transform"].isin(norm_cols)].pivot(index="tokenizer", columns="transform", values="pct_change")[norm_cols],
    "social_tweets": S[(S.group == "tweets") & S["transform"].isin(soc_cols)].pivot(index="tokenizer", columns="transform", values="pct_change")[soc_cols],
    "diacritics": S[S.group == "tashkeela"].pivot(index="tokenizer", columns="transform", values="pct_change")[["diac_asis", "diac_nfc"]],
}
for k, t in tables.items():
    t = t.loc[[o for o in order if o in t.index]]
    tables[k] = t
    t.to_csv(f"{OUT}/{k}.csv")
changed = S[S.tokenizer == S.tokenizer.iloc[0]].pivot(index="transform", columns="group", values="changed_pct")
changed.to_csv(f"{OUT}/share_of_sentences_changed.csv")

sns.set_theme(style="whitegrid", font_scale=0.85)
fig, axes = plt.subplots(1, 3, figsize=(20, 9), gridspec_kw={"width_ratios": [8, 8, 6]})
for ax, (k, title) in zip(axes, [("norm_MSA", "Normalization, MSA"), ("norm_dialect", "Normalization, dialects"),
                                 ("social_tweets", "Social cleaning, tweets")]):
    sns.heatmap(tables[k], annot=True, fmt=".1f", cmap="vlag", center=0, vmin=-6, vmax=6, ax=ax, cbar=False)
    ax.set_title(title + "\n% change in tokens (negative = fewer tokens)"); ax.set_xlabel(""); ax.set_ylabel("")
    if ax is not axes[0]:
        ax.set_yticklabels([])
plt.tight_layout(); plt.savefig(f"{OUT}/figures_ablation_heatmaps.png", dpi=150); plt.close()

plt.figure(figsize=(8, 8))
dt = tables["diacritics"].sort_values("diac_asis")
plt.barh(dt.index, dt.diac_asis, color="#4C72B0", label="diacritized, as written")
plt.scatter(dt.diac_nfc, dt.index, color="#C44E52", zorder=3, label="diacritized, after NFC")
plt.xlabel("% more tokens than the same text without diacritics"); plt.legend(loc="lower right")
plt.title("Cost of full diacritization (Tashkeela test set, 2,500 lines)")
plt.tight_layout(); plt.savefig(f"{OUT}/figures_diacritics.png", dpi=150); plt.close()

pd.set_option("display.width", 250)
for k, t in tables.items():
    print(f"\n== {k} (% change in tokens)\n{t.round(2).to_string()}")
print("\n== % of sentences changed by each transform\n", changed.round(1).to_string())
