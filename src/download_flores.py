"""Download FLORES+ dev + devtest for English, MSA and 9 Arabic dialects (gated: accept the license on HF)."""
import os, shutil
from huggingface_hub import hf_hub_download
from registry import VARIETIES

B = os.environ["B"]
out = f"{B}/data/flores"
os.makedirs(out, exist_ok=True)
for split in ["dev", "devtest"]:
    for code in VARIETIES:
        p = hf_hub_download("openlanguagedata/flores_plus", f"{split}/{code}.jsonl", repo_type="dataset")
        shutil.copy(p, f"{out}/{split}.{code}.jsonl")
        print(split, code)
