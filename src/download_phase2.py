import os, shutil
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, HfApi
B=os.environ["B"]; api=HfApi()
out=f"{B}/data/raw"; os.makedirs(out,exist_ok=True)
fw=api.list_repo_files("HuggingFaceFW/fineweb-2",repo_type="dataset")
for c in ["arb_Arab","arz_Arab","ary_Arab","ars_Arab","apc_Arab","acm_Arab","aeb_Arab"]:
    files=[f for f in fw if f.startswith(f"data/{c}/test/")]
    files+= [f for f in fw if f.startswith(f"data/{c}/train/")][: (1 if c=="arb_Arab" else 9)]
    for f in files:
        p=hf_hub_download("HuggingFaceFW/fineweb-2",f,repo_type="dataset")
        dst=f"{out}/fw2.{c}.{f.split('/')[-2]}.{f.split('/')[-1]}"
        if not os.path.exists(dst): os.symlink(p,dst)
        print(c,f,pq.ParquetFile(p).metadata.num_rows,"rows",flush=True)
for w,n in [("20231101.ar",1),("20231101.arz",3),("20231101.ary",1)]:
    for f in sorted(x for x in api.list_repo_files("wikimedia/wikipedia",repo_type="dataset") if x.startswith(w+"/"))[:n]:
        p=hf_hub_download("wikimedia/wikipedia",f,repo_type="dataset"); dst=f"{out}/wiki.{w.split('.')[1]}.{f.split('/')[-1]}"
        if not os.path.exists(dst): os.symlink(p,dst)
        print(w,f,pq.ParquetFile(p).metadata.num_rows,flush=True)
p=hf_hub_download("arbml/PADIC","data/train-00000-of-00001-f9284a3c6d306101.parquet",repo_type="dataset")
if not os.path.exists(f"{out}/padic.parquet"): os.symlink(p,f"{out}/padic.parquet")
print(pq.ParquetFile(f"{out}/fw2.arb_Arab.test.000_00000.parquet").schema_arrow)
