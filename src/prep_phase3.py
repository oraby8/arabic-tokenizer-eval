"""Phase 3 inputs: one table of (set, sid, text variants) for the ablations.

Normalization ablation uses the main-condition text (NFC + dediac) of:
  FLORES (MSA + 9 dialects), PADIC, FineWeb-2 (ALDi-filtered: MSA < 0.1, dialects >= 0.3), tweets.
Diacritics ablation: Tashkeela-based diacritization test set (2,500 lines, mostly Classical Arabic):
  diac_asis (as written), diac_nfc (NFC, which canonically reorders e.g. shadda after fatha), undiac.
Social-media ablation: ArSarcasm-v2 tweets (train + test, 15.5K), raw except NFC + dediac.
"""
import os, unicodedata
import pandas as pd
from datasets import load_dataset
from camel_tools.utils.dediac import dediac_ar
from textnorm import normalize

B = os.environ["B"]
P = f"{B}/data/prepared"
rows = []

f = pd.read_parquet(f"{P}/flores.parquet", columns=["sid", "variety", "text"])
for r in f[f.variety != "eng_Latn"].itertuples():
    rows.append(dict(set=f"flores:{r.variety}", sid=r.sid, text=r.text))
p = pd.read_parquet(f"{P}/padic.parquet", columns=["sid", "variety", "text"])
for r in p.itertuples():
    rows.append(dict(set=f"padic:{r.variety}", sid=r.sid, text=r.text))
w = pd.read_parquet(f"{P}/fineweb.parquet", columns=["sid", "variety", "text"]).merge(
    pd.read_parquet(f"{B}/results/fineweb/aldi.parquet"), on=["sid", "variety"])
w = w[((w.variety == "arb_Arab") & (w.aldi < 0.1)) | ((w.variety != "arb_Arab") & (w.aldi >= 0.3))]
for r in w.itertuples():
    rows.append(dict(set=f"fineweb:{r.variety}", sid=r.sid, text=r.text))

t = load_dataset("arbml/ArSarcasm_v2")
tw = pd.concat([t["train"].to_pandas(), t["test"].to_pandas()], ignore_index=True)
for i, r in tw.iterrows():
    rows.append(dict(set=f"tweets:{r.dialect}", sid=f"tw-{i}", text=normalize(r.tweet)))

d = load_dataset("arbml/arabic_text_diacritization")["test"].to_pandas()
for i, r in d.iterrows():
    asis = r.diaritized.strip()
    rows.append(dict(set="tashkeela", sid=f"tk-{i}", text=dediac_ar(unicodedata.normalize("NFC", asis)),
                     diac_asis=asis, diac_nfc=unicodedata.normalize("NFC", asis)))

df = pd.DataFrame(rows)
df = df[df.text.str.strip() != ""]
df.to_parquet(f"{P}/phase3.parquet")
print(df.groupby("set").size().to_string())
