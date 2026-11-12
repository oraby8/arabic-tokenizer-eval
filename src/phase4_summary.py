"""Phase 4a summary: each morphology-constrained tokenizer next to its unconstrained twin.

Pairs (same 1B-char mix, same vocab size, same trainer):
  Ref-BPE-mix-{V}            vs  Ref-MorphBPE-{morpheme,clitic}-{V}
  Ref-SPM-Unigram-plain-{V}  vs  Ref-SPM-Unigram-{morpheme,clitic}-{V}
Metrics: FLORES MSA fertility and chars/token; web MSA chars/token; dialect penalty (all sets,
PADIC parallel, unseen dialects); absolute dialect chars/token; morphological boundary F1.
"""
import os
import pandas as pd

B = os.environ["B"]
OUT = f"{B}/results/phase4"
os.makedirs(OUT, exist_ok=True)
fl = pd.read_csv(f"{B}/results/flores/core/summary.csv", index_col=0)
p2 = pd.read_csv(f"{B}/results/phase2/core/summary.csv", index_col=0)
L = pd.read_csv(f"{B}/results/phase2/core/metrics_long.csv")
F = L[(L.subset == "filtered") & L.len_adj_penalty.notna()]
dial_cpt = F.groupby(["tokenizer", "corpus"]).chars_per_token.mean().unstack().mean(1)
ma = pd.read_csv(f"{OUT}/morph_alignment.csv")
f1 = ma.pivot_table(index="tokenizer", columns=["group", "gold"], values="f1")
f1.columns = [f"F1_{g}_{s}" for g, s in f1.columns]

rows = []
for V in [32, 64, 128]:
    pairs = [("BPE", f"Ref-BPE-mix-{V}K", "none"), ("BPE", f"Ref-MorphBPE-morpheme-{V}K", "morpheme"),
             ("SPM-Unigram", f"Ref-SPM-Unigram-plain-{V}K", "none"), ("SPM-Unigram", f"Ref-SPM-Unigram-morpheme-{V}K", "morpheme")]
    if V == 64:
        pairs += [("BPE", "Ref-MorphBPE-clitic-64K", "clitic"), ("SPM-Unigram", "Ref-SPM-Unigram-clitic-64K", "clitic")]
    for algo, t, c in pairs:
        rows.append(dict(vocab=f"{V}K", algorithm=algo, constraint=c, tokenizer=t,
                         fert_MSA_flores=fl.loc[t, "fert_MSA"], cpt_MSA_flores=fl.loc[t, "chars/tok_MSA"],
                         cpt_MSA_web=p2.loc[t, "MSA_web_chars/tok"], cpt_dialect_abs=dial_cpt[t],
                         penalty_all=p2.loc[t, "penalty_all_mean"], penalty_padic=p2.loc[t, "penalty_padic"],
                         penalty_unseen=p2.loc[t, "penalty_web_unseen"],
                         F1_msa_morpheme=f1.loc[t, "F1_msa_morpheme"], F1_msa_clitic=f1.loc[t, "F1_msa_clitic"],
                         F1_egy_morpheme=f1.loc[t, "F1_egy_morpheme"]))
S = pd.DataFrame(rows)
base = S[S.constraint == "none"].set_index(["vocab", "algorithm"])
for col in ["fert_MSA_flores", "cpt_MSA_web", "cpt_dialect_abs"]:
    S[f"{col}_vs_twin_%"] = [100 * (r[col] / base.loc[(r.vocab, r.algorithm), col] - 1) for _, r in S.iterrows()]
S.to_csv(f"{OUT}/phase4a_summary.csv", index=False)

# appendix: HF-tokenizers Unigram with the same constraint (over-segments)
app = pd.DataFrame([dict(tokenizer=t, fert_MSA_flores=fl.loc[t, "fert_MSA"] if t in fl.index else None,
                         cpt_MSA_web=p2.loc[t, "MSA_web_chars/tok"] if t in p2.index else None)
                    for t in ["Ref-Unigram-mix-64K", "Ref-HFMorphUnigram-morpheme-64K", "Ref-HFMorphUnigram-clitic-64K"]])
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(S.drop(columns="tokenizer").round(3).to_string(index=False))
print("\nreal tokenizers, F1 (msa morpheme / msa clitic / egy morpheme):")
print(f1.loc[[t for t in ["ALLaM-7B", "Jais-adapted", "Gemma-3", "Llama-3.1", "Qwen3.5", "gpt-oss", "Karnak-6B"] if t in f1.index]].round(3).to_string())
print("\nappendix (HF Unigram + constraint; not in phase2/flores core if absent):")
print(app.to_string(index=False))
