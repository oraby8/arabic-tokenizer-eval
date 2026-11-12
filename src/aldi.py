"""Dialectness (ALDi, Keleg et al. 2023): 0 = MSA, 1 = strongly dialectal.

Sentence-level model; a document's score is the mean over its first 8 sentences with 3+ words
(FLORES/PADIC rows are single sentences, so they are scored directly).
Usage: CORPUS=fineweb python aldi.py
"""
import os, re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

B = os.environ["B"]
corpus = os.environ.get("CORPUS", "flores")
SPLIT = re.compile(r"(?<=[.!?؟\n])\s+")

df = pd.read_parquet(f"{B}/data/prepared/{corpus}.parquet", columns=["sid", "variety", "text"])
df = df[df.variety != "eng_Latn"].reset_index(drop=True)
units = []
for i, t in enumerate(df.text):
    sents = [s for s in SPLIT.split(t) if len(s.split()) >= 3][:8] or [t]
    units += [(i, s) for s in sents]

tok = AutoTokenizer.from_pretrained("AMR-KELEG/Sentence-ALDi")
model = AutoModelForSequenceClassification.from_pretrained("AMR-KELEG/Sentence-ALDi").cuda().eval()
scores = []
with torch.no_grad():
    for k in range(0, len(units), 512):
        batch = tok([s for _, s in units[k:k + 512]], padding=True, truncation=True, max_length=128, return_tensors="pt").to("cuda")
        scores += model(**batch).logits.squeeze(-1).clamp(0, 1).tolist()
u = pd.DataFrame({"row": [i for i, _ in units], "aldi": scores}).groupby("row").aldi.mean()
df["aldi"] = u.reindex(range(len(df))).values
os.makedirs(f"{B}/results/{corpus}", exist_ok=True)
df[["sid", "variety", "aldi"]].to_parquet(f"{B}/results/{corpus}/aldi.parquet")
print(corpus, df.groupby("variety").aldi.describe(percentiles=[.25, .5, .75])[["mean", "25%", "50%", "75%"]].sort_values("mean").round(3).to_string(), sep="\n")
