"""Phase 4b evaluation: bits-per-character and Belebele for each trained model.

BPC (tokenizer-independent): sum of -log2 p over all tokens of a document (conditioned on a leading
EOS) / number of characters. Held-out sets never used for tokenizer or LM training:
  FLORES+ (MSA + 9 dialects), PADIC (MSA + 4), FineWeb-2 *test* (MSA ALDi < 0.1, dialects ALDi >= 0.3).
Belebele (facebook/belebele: arb, arz, ary, ars, apc, acm; 900 questions each), zero-shot:
  context "P: {passage}\nQ: {question}\nA:" + " {option}"; score = log p(option) / len(option chars)
  (per-character normalisation keeps tokenizers comparable); 95% CI = normal approx. of binomial.
Text gets the same normalisation as training (NFC + diacritics removed).
Usage: CUDA_VISIBLE_DEVICES=0 python lm_eval.py --run scratch-BPE   (or --base for raw Qwen3-0.6B)
"""
import argparse, json, math, os
import numpy as np
import pandas as pd
import torch
from lm_common import B, QWEN, Encoder, save_json
from textnorm import normalize

ap = argparse.ArgumentParser()
ap.add_argument("--run")
ap.add_argument("--base", action="store_true", help="evaluate Qwen3-0.6B-Base without training")
ap.add_argument("--max-docs", type=int, default=0)
a = ap.parse_args()
from transformers import AutoModelForCausalLM

if a.base:
    name, tokname, path = "base-Qwen3", "Qwen3", QWEN
else:
    info = json.load(open(f"{B}/lm_runs/{a.run}/train_log.json"))
    name, tokname, path = a.run, info["tok"], f"{B}/lm_runs/{a.run}/model"
enc = Encoder(tokname)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
MAXLEN = 1024
BS = 8


@torch.no_grad()
def token_logprobs(batch_ids):
    """batch of id lists (each starting with EOS) -> list of per-position log p(token_t | <t), t>=1."""
    L = min(max(len(x) for x in batch_ids), MAXLEN)
    x = torch.full((len(batch_ids), L), enc.eos, dtype=torch.long)
    for i, ids in enumerate(batch_ids):
        ids = ids[:L]; x[i, :len(ids)] = torch.tensor(ids)
    x = x.cuda()
    logits = model(input_ids=x).logits
    out = []
    for i, ids in enumerate(batch_ids):  # per row: avoids materialising a float32 [B, L, V] tensor
        n = min(len(ids), L) - 1
        lp = -torch.nn.functional.cross_entropy(logits[i, :n].float(), x[i, 1:n + 1], reduction="none")
        out.append(lp.cpu().numpy())
    return out


def bpc_sets():
    P = f"{B}/data/prepared"
    out = []
    f = pd.read_parquet(f"{P}/flores.parquet", columns=["variety", "text"])
    out += [(f"flores:{v}", t) for v, t in zip(f.variety, f.text) if v != "eng_Latn"]
    p = pd.read_parquet(f"{P}/padic.parquet", columns=["variety", "text"])
    out += [(f"padic:{v}", t) for v, t in zip(p.variety, p.text)]
    w = pd.read_parquet(f"{P}/fineweb.parquet", columns=["sid", "variety", "text"]).merge(
        pd.read_parquet(f"{B}/results/fineweb/aldi.parquet"), on=["sid", "variety"])
    w = w[((w.variety == "arb_Arab") & (w.aldi < 0.1)) | ((w.variety != "arb_Arab") & (w.aldi >= 0.3))]
    out += [(f"fineweb:{v}", t) for v, t in zip(w.variety, w.text)]
    return pd.DataFrame(out, columns=["set", "text"])


def shared_keep(df):
    """Docs that fit in MAXLEN tokens under *every* LM tokenizer, so all models see the same text."""
    path = f"{B}/results/phase4/lm/bpc_keep.npy"
    if a.max_docs == 0 and os.path.exists(path):
        return np.load(path)
    from lm_common import LM_TOKENIZERS
    keep = np.ones(len(df), dtype=bool)
    for t in LM_TOKENIZERS:
        keep &= np.array([len(x) + 1 <= MAXLEN for x in Encoder(t).encode(df.text.tolist())])
    if a.max_docs == 0:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, keep)
    return keep


def run_bpc():
    df = bpc_sets()
    if a.max_docs:
        df = df.groupby("set").head(a.max_docs)
    df = df[shared_keep(df)].reset_index(drop=True)
    ids = [[enc.eos] + x for x in enc.encode(df.text.tolist())]
    order = np.argsort([len(x) for x in ids])
    nll = np.zeros(len(ids)); trunc = 0
    for k in range(0, len(order), BS):
        idx = order[k:k + BS]
        for i, lp in zip(idx, token_logprobs([ids[i] for i in idx])):
            nll[i] = -lp.sum() / math.log(2)
            trunc += len(ids[i]) > MAXLEN
    df["bits"], df["chars"] = nll, df.text.str.len()
    assert trunc == 0, "shared_keep should have removed every over-length doc"
    g = df.groupby("set").agg(bits=("bits", "sum"), chars=("chars", "sum"), docs=("text", "size"))
    g["bpc"] = g.bits / g.chars
    return g.reset_index(), trunc


def run_belebele():
    from huggingface_hub import hf_hub_download
    rows = []
    for lang in ["arb_Arab", "arz_Arab", "ary_Arab", "ars_Arab", "apc_Arab", "acm_Arab"]:
        items = [json.loads(l) for l in open(hf_hub_download("facebook/belebele", f"data/{lang}.jsonl", repo_type="dataset"))]
        if a.max_docs:
            items = items[:a.max_docs]
        correct = 0
        for it in items:
            ctx = normalize(f"P: {it['flores_passage']}\nQ: {it['question']}\nA:")
            opts = [normalize(" " + it[f"mc_answer{j}"]) for j in range(1, 5)]
            seqs, spans = [], []
            for o in opts:
                full_ids, offs = enc.encode_offsets(ctx + o)
                first = next((k for k, (s, e) in enumerate(offs) if s >= len(ctx) or e > len(ctx)), len(offs))
                seqs.append([enc.eos] + full_ids); spans.append(first)
            lps = token_logprobs(seqs)
            scores = [lp[first:].sum() / max(len(o.strip()), 1) for lp, first, o in zip(lps, spans, opts)]
            correct += int(np.argmax(scores)) + 1 == int(it["correct_answer_num"])
        acc = correct / len(items)
        rows.append(dict(lang=lang, n=len(items), acc=acc, ci95=1.96 * math.sqrt(acc * (1 - acc) / len(items))))
        print(f"  belebele {lang}: {acc:.3f}", flush=True)
    return pd.DataFrame(rows)


bpc, trunc = run_bpc()
print(name, "BPC", bpc[["set", "bpc"]].round(3).to_string(index=False), flush=True)
bel = run_belebele()
out = f"{B}/results/phase4/lm/{name}" + ("-smoke" if a.max_docs else "")  # smoke runs never shadow real results
os.makedirs(out, exist_ok=True)
bpc.to_csv(f"{out}/bpc.csv", index=False)
bel.to_csv(f"{out}/belebele.csv", index=False)
save_json(dict(run=name, tok=tokname, n_docs=int(bpc.docs.sum())), f"{out}/meta.json")
print("done", name, flush=True)
