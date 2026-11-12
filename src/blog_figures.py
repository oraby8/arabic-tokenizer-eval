"""Four clean figures for the write-up, drawn from the result CSVs (no server needed).

Usage: B=<repo root> python src/blog_figures.py   -> results/blog_figures/*.png
Colors: validated categorical slots 1-3 (blue = global, orange = Arabic-centric, aqua = reference);
every mark is direct-labeled (aqua is below 3:1 contrast on the light surface).
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = os.environ.get("B", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(B, "src"))
from registry import TOKENIZERS

OUT = f"{B}/results/blog_figures"
os.makedirs(OUT, exist_ok=True)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COL = {"global": "#2a78d6", "arabic": "#eb6834", "reference": "#1baf7a"}
GROUP = {c["name"]: c["group"] for c in TOKENIZERS}
LABEL = {  # canonical tokenizer -> label naming every model that shares it
    "Gemma-3": "Gemma-3/4 · Fanar-2 · Nile-Chat", "Aya-Expanse": "Aya · Command-A/R/R7B-Arabic",
    "Llama-3.1": "Llama-3.1/3.2", "Qwen2.5": "Qwen2.5/3", "Karnak-6B": "Karnak (6B/40B)",
    "gpt-oss": "gpt-oss (= GPT-4o)", "Atlas-Chat": "Atlas-Chat (= Gemma-2)",
}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def finish(ax, fig, title, subtitle, name, source):
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK, y=0.985)
    fig.text(0.01, 0.925, subtitle, ha="left", fontsize=9.5, color=INK2)
    fig.text(0.01, 0.012, source, ha="left", fontsize=8, color=INK2)
    fig.savefig(f"{OUT}/{name}.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def legend(ax, groups, loc=None):
    names = {"global": "Global model", "arabic": "Arabic-centric model", "reference": "Reference (trained in this study)"}
    h = [plt.Rectangle((0, 0), 1, 1, color=COL[g]) for g in groups]
    ax.figure.legend(h, [names[g] for g in groups], loc="upper right", bbox_to_anchor=(0.99, 0.905),
                     ncol=len(groups), frameon=False, fontsize=9)


# 1. Arabic vs English token cost (FLORES+, MSA)
s = pd.read_csv(f"{B}/results/flores/core/summary.csv", index_col=0)
s = s[s.group.isin(["global", "arabic"])].sort_values("parity_MSA", ascending=False)
fig, ax = plt.subplots(figsize=(9, 6.4))
fig.subplots_adjust(left=0.33, right=0.97, top=0.86, bottom=0.11)
y = np.arange(len(s))
ax.barh(y, s.parity_MSA, height=0.68, color=[COL[GROUP[t]] for t in s.index])
for i, v in enumerate(s.parity_MSA):
    ax.text(v + 0.02, i, f"{v:.2f}×", va="center", fontsize=9, color=INK)
ax.set_yticks(y, [LABEL.get(t, t) for t in s.index])
ax.axvline(1.0, color=INK, lw=1, ls="--")
ax.text(1.0, len(s) - 0.35, " same cost as English", fontsize=8.5, color=INK, va="bottom")
ax.axvline(1.58, color=INK2, lw=1, ls=":")
ax.text(1.58, len(s) - 0.35, " raw UTF-8 bytes", fontsize=8.5, color=INK2, va="bottom")
ax.set_xlim(0, 1.95); ax.set_ylim(-0.6, len(s) + 0.4)
ax.set_xlabel("Arabic (MSA) tokens ÷ English tokens for the same sentences")
ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
legend(ax, ["global", "arabic"])
finish(ax, fig, "How much more does Arabic cost than English?",
       "Same 2,009 FLORES+ sentences in both languages · lower is better", "1_arabic_vs_english",
       "Source: github.com/oraby8/arabic-tokenizer-eval · FLORES+ dev+devtest")

# 2. Arabic tokens in the vocabulary predict efficiency
import json
meta = json.load(open(f"{B}/results/flores/tokenizers.json"))
s2 = s.copy()
s2["n_ar"] = [meta[t]["arabic_vocab"] for t in s2.index]
fig, ax = plt.subplots(figsize=(9, 6.2))
fig.subplots_adjust(left=0.09, right=0.97, top=0.86, bottom=0.12)
offsets = {  # (dx, dy) in points; negative dx = label to the left of the point
    "ALLaM-7B": (8, 4), "Jais-adapted": (-8, -14), "Karnak-6B": (-8, 6), "Jais-3-0.9B": (8, 2), "Falcon-H1": (8, -4),
    "Qwen3.5": (8, 2), "gpt-oss": (8, -6), "Gemma-3": (8, -2), "Aya-Expanse": (-8, 6), "Atlas-Chat": (8, -14),
    "Fanar-1": (-8, -14), "Llama-4": (8, 2), "Qwen2.5": (8, -1), "Llama-3.1": (-8, -8), "DeepSeek-V3": (-8, 4)}
for t, r in s2.iterrows():
    ax.scatter(r.n_ar, r["chars/tok_MSA"], s=70, color=COL[GROUP[t]], edgecolor=SURFACE, lw=2, zorder=3)
    dx, dy = offsets.get(t, (8, 2))
    ax.annotate(f"{LABEL.get(t, t)} · {r.vocab / 1000:.0f}K", (r.n_ar, r["chars/tok_MSA"]), xytext=(dx, dy),
                textcoords="offset points", fontsize=8, color=INK, ha="left" if dx > 0 else "right", va="center")
ax.set_xscale("log")
ax.set_xticks([2000, 3000, 5000, 10000, 20000, 40000], ["2K", "3K", "5K", "10K", "20K", "40K"])
ax.minorticks_off()
ax.set_xlabel("Number of Arabic-script tokens in the vocabulary (log scale)")
ax.set_ylabel("MSA characters per token (higher = more efficient)")
ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.set_xlim(1500, 80000); ax.set_ylim(2.5, 4.3)
ax.text(0.99, 0.02, "label: model · total vocabulary size", transform=ax.transAxes, ha="right", fontsize=8, color=INK2)
legend(ax, ["global", "arabic"])
finish(ax, fig, "Arabic tokens in the vocabulary predict efficiency, total vocabulary size doesn't",
       "Spearman ρ = 0.96 with the number of Arabic tokens vs ρ = −0.28 with vocabulary size (15 distinct tokenizers)",
       "2_arabic_tokens_vs_efficiency", "Source: github.com/oraby8/arabic-tokenizer-eval · FLORES+ MSA")

# 3. Controlled experiment: MSA-only vs MSA+dialect training (BPE and Unigram, 64K)
p2 = pd.read_csv(f"{B}/results/phase2/core/summary.csv", index_col=0)
metrics = [("penalty_all_mean", "All dialect sets\n(12 sets, 3 corpora)"),
           ("penalty_web_unseen", "Dialects never seen\nin training (web)"),
           ("penalty_padic", "Parallel sentences,\nsame meaning (PADIC)")]
fig, axes = plt.subplots(1, 2, figsize=(10, 5.4), sharey=True)
fig.subplots_adjust(left=0.08, right=0.98, top=0.80, bottom=0.17, wspace=0.08)
for ax, algo in zip(axes, ["BPE", "Unigram"]):
    x = np.arange(len(metrics))
    for k, (cfg, color, lab) in enumerate([("msa", "#2a78d6", "Trained on MSA only"),
                                           ("mix", "#eb6834", "Trained on 50% MSA + 50% dialect")]):
        vals = [100 * (p2.loc[f"Ref-{algo}-{cfg}-64K", m] - 1) for m, _ in metrics]
        bars = ax.bar(x + (k - 0.5) * 0.36, vals, width=0.34, color=color, label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"+{v:.0f}%", ha="center", fontsize=9, color=INK)
    msa_cost = 100 * (p2.loc[f"Ref-{algo}-mix-64K", "MSA_web_chars/tok"] / p2.loc[f"Ref-{algo}-msa-64K", "MSA_web_chars/tok"] - 1)
    ax.set_title(f"{algo}, 64K vocabulary · MSA efficiency {msa_cost:+.1f}%", fontsize=10, color=INK, loc="left")
    ax.set_xticks(x, [l for _, l in metrics], fontsize=8.5)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("Extra tokens per character on dialect text vs MSA")
axes[0].set_ylim(0, 62)
axes[0].legend(loc="upper left", frameon=False, fontsize=9)
fig.suptitle("Adding dialect text to tokenizer training halves the dialect penalty", x=0.01, ha="left",
             fontsize=13, fontweight="bold", color=INK, y=0.985)
fig.text(0.01, 0.905, "Same algorithm, vocabulary size and 1B training characters; only the training mix changes",
         ha="left", fontsize=9.5, color=INK2)
fig.text(0.01, 0.012, "Source: github.com/oraby8/arabic-tokenizer-eval · FineWeb-2, Wikipedia, PADIC (ALDi-filtered)",
         ha="left", fontsize=8, color=INK2)
fig.savefig(f"{OUT}/3_dialect_training_experiment.png", dpi=200, facecolor=SURFACE)
plt.close(fig)

# 4. Diacritics cost
d = pd.read_csv(f"{B}/results/phase3/diacritics.csv", index_col=0)
d = d[~d.index.str.startswith("Ref-") & (d.index != "Gemma-4")].sort_values("diac_asis")
fig, ax = plt.subplots(figsize=(9, 6.2))
fig.subplots_adjust(left=0.33, right=0.95, top=0.86, bottom=0.11)
y = np.arange(len(d))
ax.barh(y, d.diac_asis, height=0.68, color=[COL[GROUP[t]] for t in d.index])
for i, v in enumerate(d.diac_asis):
    ax.text(v + 5, i, f"+{v:.0f}%  ({1 + v / 100:.1f}×)", va="center", fontsize=9, color=INK)
ax.set_yticks(y, [LABEL.get(t, t) for t in d.index])
ax.invert_yaxis()
ax.set_xlim(0, 480)
ax.set_xlabel("Extra tokens for fully diacritized text vs the same text without diacritics")
ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
legend(ax, ["global", "arabic"])
finish(ax, fig, "Diacritics multiply the token count by 2–5× for these models",
       "2,500 fully diacritized sentences (Tashkeela test set, mostly Classical Arabic) · lower is better",
       "4_diacritics_cost", "Source: github.com/oraby8/arabic-tokenizer-eval · arbml/arabic_text_diacritization")
print("wrote", sorted(os.listdir(OUT)))


# 5. Morphology constraints: token cost vs language-modelling payoff (64K, Phase 4)
p4 = pd.read_csv(f"{B}/results/phase4/phase4a_summary.csv")
tw = pd.read_csv(f"{B}/results/phase4/phase4b_twins.csv").set_index(["mode", "variant"])
pairs = [("MorphBPE", "Ref-MorphBPE-morpheme-64K", "Ref-BPE-mix-64K", "MorphBPE\nvs BPE"),
         ("SPM-Unigram-morph", "Ref-SPM-Unigram-morpheme-64K", "Ref-SPM-Unigram-plain-64K", "Unigram + morph\nvs Unigram")]
fert = p4.set_index("tokenizer")["fert_MSA_flores"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.8))
fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.17, wspace=0.3)
x = np.arange(len(pairs))
cost = [100 * (fert[v] / fert[b] - 1) for _, v, b, _ in pairs]
bars = a1.bar(x, cost, width=0.5, color="#52514e")
for b_, v in zip(bars, cost):
    a1.text(b_.get_x() + b_.get_width() / 2, v + 1.5, f"+{v:.0f}%", ha="center", fontsize=10, color=INK)
a1.set_xticks(x, [p[3] for p in pairs]); a1.set_ylim(0, max(cost) * 1.25)
a1.set_title("Cost: more tokens per MSA word", fontsize=10.5, color=INK, loc="left")
a1.grid(axis="y", color=GRID, lw=0.8); a1.set_axisbelow(True)
cm = pd.read_csv(f"{B}/results/phase4/phase4b_compute_matched.csv")
eq_steps = {v: -cm[(cm.variant == v) & cm.setting.str.startswith("equal steps")].mean_rel_bpc_change_pct.mean()
            for v in ("MorphBPE", "SPM-Unigram-morph")}
series = [("From scratch, equal characters", "#2a78d6", [-tw.loc[("scratch", p[0]), "mean_rel_bpc_change_pct"] for p in pairs]),
          ("From scratch, equal compute", "#eb6834", [eq_steps[p[0]] for p in pairs]),
          ("Qwen3-0.6B transplant, equal characters", "#1baf7a", [-tw.loc[("transplant", p[0]), "mean_rel_bpc_change_pct"] for p in pairs])]
for k, (lab, color, vals) in enumerate(series):
    bs = a2.bar(x + (k - 1) * 0.26, vals, width=0.24, color=color, label=lab)
    for b_, v in zip(bs, vals):
        a2.text(b_.get_x() + b_.get_width() / 2, v + (0.1 if v >= 0 else -0.1), f"{v:+.1f}%", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=9, color=INK)
a2.axhline(0, color=INK2, lw=1)
a2.set_xticks(x, [p[3] for p in pairs]); a2.set_ylim(-5.0, 5.6)
a2.set_title("Payoff: bits-per-character reduction (above 0 = better)", fontsize=10.5, color=INK, loc="left")
a2.legend(frameon=False, fontsize=8.5, loc="lower right", bbox_to_anchor=(1.0, 0.0), borderaxespad=0.2)
a2.grid(axis="y", color=GRID, lw=0.8); a2.set_axisbelow(True)
for ax in (a1, a2):
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0); ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
fig.suptitle("Morphology constraints: a big token cost; at equal compute, a gain of ~1% at best", x=0.01, ha="left",
             fontsize=13, fontweight="bold", color=INK, y=0.985)
fig.text(0.01, 0.885, "64K vocabularies, same 1B training characters. BPC = mean over 22 MSA and dialect test sets. "
         "Equal compute = same number of training steps.", ha="left", fontsize=9.5, color=INK2)
fig.text(0.01, 0.012, "Source: FLORES+, PADIC, FineWeb-2 (held out) · github.com/oraby8/arabic-tokenizer-eval",
         ha="left", fontsize=8, color=INK2)
fig.savefig(f"{OUT}/5_morphology_cost_vs_payoff.png", dpi=200, facecolor=SURFACE)
plt.close(fig)
