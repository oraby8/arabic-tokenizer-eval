"""Score every FineWeb-2 dialect *train* doc (arz, ary, ars) with ALDi, sharded over GPUs.

FineWeb-2's dialect labels are unreliable (the ary subset is ~98% MSA news), so the dialect half
of the reference-tokenizer training mix is selected by ALDi instead of by label.
Doc score = mean ALDi over its first 3 sentences with 3+ words.
Usage: SHARD=i NSHARDS=4 CUDA_VISIBLE_DEVICES=i python aldi_train.py
"""
import glob, os, re
import pandas as pd
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from textnorm import normalize

B = os.environ["B"]
shard, nshards = int(os.environ["SHARD"]), int(os.environ["NSHARDS"])
SPLIT = re.compile(r"(?<=[.!?؟\n])\s+")

docs = []
for code in ["arz_Arab", "ary_Arab", "ars_Arab"]:
    for f in sorted(glob.glob(f"{B}/data/raw/fw2.{code}.train.*.parquet")):
        texts = pq.read_table(f, columns=["text"]).column(0).to_pylist()
        docs += [(code, os.path.basename(f), i, t) for i, t in enumerate(texts) if i % nshards == shard]

units = []
for k, (_, _, _, t) in enumerate(docs):
    sents = [s for s in SPLIT.split(normalize(t[:3000])) if len(s.split()) >= 3][:3] or [t[:500]]
    units += [(k, s) for s in sents]
units.sort(key=lambda u: len(u[1]))

tok = AutoTokenizer.from_pretrained("AMR-KELEG/Sentence-ALDi")
model = AutoModelForSequenceClassification.from_pretrained("AMR-KELEG/Sentence-ALDi", torch_dtype=torch.float16).cuda().eval()
scores = []
with torch.no_grad():
    for i in range(0, len(units), 1024):
        b = tok([s for _, s in units[i:i + 1024]], padding=True, truncation=True, max_length=128, return_tensors="pt").to("cuda")
        scores += model(**b).logits.float().squeeze(-1).clamp(0, 1).tolist()
        if i % (1024 * 500) == 0:
            print(f"shard {shard}: {i:,}/{len(units):,}", flush=True)
s = pd.DataFrame({"k": [k for k, _ in units], "aldi": scores}).groupby("k").aldi.mean()
out = pd.DataFrame([dict(code=c, file=f, row=r, n_chars=len(t)) for c, f, r, t in docs])
out["aldi"] = s.reindex(range(len(out))).values
os.makedirs(f"{B}/data/train_aldi", exist_ok=True)
out.to_parquet(f"{B}/data/train_aldi/shard{shard}.parquet")
print(f"shard {shard} done", out.groupby("code").aldi.agg(["size", "mean", lambda x: (x >= 0.3).mean()]).to_string())
