"""Phase 4b summary: BPC (MSA vs dialects, per source) and Belebele for every trained model.

Reads results/phase4/lm/<run>/{bpc,belebele}.csv and results/phase4/lm_train/<run>.json.
Writes results/phase4/phase4b_bpc.csv, phase4b_belebele.csv, phase4b_twins.csv, phase4b_compute_matched.csv.
Compute-matched controls (run_lm_ctrl.sh): each scratch model retrained with its twin's optimizer-step count.
BPC per source: MSA = the arb_Arab / padic_msa set; dialects = unweighted mean over that source's dialect sets.
"""
import json, os
import pandas as pd

R = os.path.join(os.path.dirname(__file__), "..", "results", "phase4")
RUNS = ["base-Qwen3", "qwen-Qwen3"] + [f"{m}-{t}" for m in ("scratch", "transplant")
                                       for t in ("BPE", "MorphBPE", "SPM-Unigram", "SPM-Unigram-morph")]
RUNS += ["scratch-BPE-steps2285", "scratch-MorphBPE-steps1807", "scratch-SPM-Unigram-steps2825", "scratch-SPM-Unigram-morph-steps1816"]
# (base, variant, base retrained with the variant's steps, variant retrained with the base's steps)
CONTROLS = [("BPE", "MorphBPE", "BPE-steps2285", "MorphBPE-steps1807"),
            ("SPM-Unigram", "SPM-Unigram-morph", "SPM-Unigram-steps2825", "SPM-Unigram-morph-steps1816")]
TWINS = [("BPE", "MorphBPE"), ("SPM-Unigram", "SPM-Unigram-morph"), ("BPE", "SPM-Unigram")]


def is_msa(s):
    return s.endswith("arb_Arab") or s.endswith("padic_msa")


rows, bel, per_set = [], [], {}
for run in RUNS:
    b = pd.read_csv(f"{R}/lm/{run}/bpc.csv")
    per_set[run] = b.set_index("set").bpc
    rec = dict(run=run)
    tl = f"{R}/lm_train/{run}.json"
    if os.path.exists(tl):
        t = json.load(open(tl))
        rec.update(train_tokens_M=round(t["tokens"] / 1e6, 1), gpu_hours=round(t["hours"], 2))
    b["src"] = b.set.str.split(":").str[0]
    for src, g in b.groupby("src"):
        m = g[g.set.map(is_msa)].bpc.mean()
        d = g[~g.set.map(is_msa)].bpc.mean()
        rec[f"{src}_msa"], rec[f"{src}_dia"], rec[f"{src}_gap"] = m, d, d - m
    rec["all_msa"] = b[b.set.map(is_msa)].bpc.mean()
    rec["all_dia"] = b[~b.set.map(is_msa)].bpc.mean()
    rows.append(rec)
    e = pd.read_csv(f"{R}/lm/{run}/belebele.csv")
    e["run"] = run
    bel.append(e)

bpc = pd.DataFrame(rows)
bel = pd.concat(bel)
bw = bel.pivot(index="run", columns="lang", values="acc").reindex(RUNS)
bw["dialect_mean"] = bw.drop(columns="arb_Arab").mean(1)
bw["all_mean"] = bel.groupby("run").acc.mean().reindex(RUNS)
# CI of a 5400-question mean (all six languages pooled)
n = bel.groupby("run").n.sum().reindex(RUNS)
bw["all_ci95"] = 1.96 * (bw.all_mean * (1 - bw.all_mean) / n) ** 0.5

twins = []
for mode in ("scratch", "transplant"):
    for a, c in TWINS:
        x, y = per_set[f"{mode}-{a}"], per_set[f"{mode}-{c}"]
        diff = (y - x) / x
        twins.append(dict(mode=mode, base=a, variant=c, sets_variant_better=int((diff < 0).sum()), sets=len(diff),
                          mean_rel_bpc_change_pct=round(100 * diff.mean(), 2),
                          msa_rel_pct=round(100 * diff[diff.index.map(is_msa)].mean(), 2),
                          dia_rel_pct=round(100 * diff[~diff.index.map(is_msa)].mean(), 2),
                          belebele_change_pts=round(100 * (bw.all_mean[f"{mode}-{c}"] - bw.all_mean[f"{mode}-{a}"]), 1)))
twins = pd.DataFrame(twins)


def compare(x, y):
    d = (per_set[y] - per_set[x]) / per_set[x]
    return dict(sets_variant_better=int((d < 0).sum()), mean_rel_bpc_change_pct=round(100 * d.mean(), 2),
                msa_rel_pct=round(100 * d[d.index.map(is_msa)].mean(), 2),
                dia_rel_pct=round(100 * d[~d.index.map(is_msa)].mean(), 2),
                belebele_change_pts=round(100 * (bw.all_mean[y] - bw.all_mean[x]), 1))


ctrl = []
for base, var, base_long, var_short in CONTROLS:
    for setting, x, y in [("equal characters (main run)", base, var),
                          ("equal steps: base trained as long as the variant", base_long, var),
                          ("equal steps: variant trained as short as the base", base, var_short)]:
        ctrl.append(dict(base=base, variant=var, setting=setting, **compare(f"scratch-{x}", f"scratch-{y}")))
ctrl = pd.DataFrame(ctrl)

bpc.round(3).to_csv(f"{R}/phase4b_bpc.csv", index=False)
bw.round(3).to_csv(f"{R}/phase4b_belebele.csv")
twins.to_csv(f"{R}/phase4b_twins.csv", index=False)
ctrl.to_csv(f"{R}/phase4b_compute_matched.csv", index=False)
pd.set_option("display.width", 250)
print(bpc.round(3).to_string(index=False)); print(); print(bw.round(3).to_string()); print(); print(twins.to_string(index=False)); print(); print(ctrl.to_string(index=False))
