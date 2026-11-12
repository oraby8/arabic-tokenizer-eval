"""Phase 4b: train one language model on the 1B-char mix (one epoch, equal characters for all).

--mode scratch     Llama-style ~135M model from random init (12 layers, d=768, tied embeddings).
--mode transplant  Qwen3-0.6B-Base with its tokenizer replaced: each new token's embedding starts as the
                   mean of Qwen3's embeddings for the same bytes (AraToken-style); body frozen for the
                   first 5% of steps, then everything trains.
--mode qwen        control: Qwen3-0.6B-Base with its own tokenizer, same data and schedule.
Usage: CUDA_VISIBLE_DEVICES=0 python lm_train.py --tok BPE --mode scratch [--max-steps N]
"""
import argparse, json, math, os, time
import numpy as np
import torch
from lm_common import B, QWEN, Encoder, save_json

ap = argparse.ArgumentParser()
ap.add_argument("--tok", required=True)
ap.add_argument("--mode", choices=["scratch", "transplant", "qwen"], required=True)
ap.add_argument("--seq", type=int, default=1024)
ap.add_argument("--batch", type=int, default=128, help="sequences per optimizer step")
ap.add_argument("--micro", type=int, default=0, help="micro-batch (default: 32 scratch, 8 otherwise)")
ap.add_argument("--max-steps", type=int, default=0, help="smoke test")
ap.add_argument("--compile", action="store_true")
ap.add_argument("--steps", type=int, default=0, help="full run with this many optimizer steps (compute-matched controls)")
ap.add_argument("--tag", default="", help="suffix for the run name")
a = ap.parse_args()
torch.manual_seed(0); np.random.seed(0)
dev = "cuda"
micro = a.micro or (32 if a.mode == "scratch" else 8)
accum = a.batch // micro
run = f"{a.mode}-{a.tok}" + (f"-{a.tag}" if a.tag else "")
out_dir = f"{B}/lm_runs/{run}"
os.makedirs(out_dir, exist_ok=True)

meta = json.load(open(f"{B}/data/lm/{a.tok}.json"))
data = np.memmap(f"{B}/data/lm/{a.tok}.bin", dtype=np.uint32, mode="r")
n_blocks = (len(data) - 1) // a.seq
order = np.random.default_rng(0).permutation(n_blocks)
steps = a.max_steps or a.steps or n_blocks // a.batch  # --steps beyond one epoch wraps around the same permutation

from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

if a.mode == "scratch":
    cfg = LlamaConfig(vocab_size=meta["vocab_size"], hidden_size=768, intermediate_size=2048, num_hidden_layers=12,
                      num_attention_heads=12, num_key_value_heads=12, max_position_embeddings=a.seq, rope_theta=10000.0,
                      tie_word_embeddings=True, bos_token_id=meta["eos"], eos_token_id=meta["eos"])
    model = LlamaForCausalLM(cfg)
    lr, emb_lr, warm = 6e-4, 6e-4, max(1, int(0.02 * steps))
else:
    model = AutoModelForCausalLM.from_pretrained(QWEN, dtype=torch.float32)
    lr, emb_lr, warm = 1e-4, 5e-4, max(1, int(0.02 * steps))
    if a.mode == "transplant":
        from tokenizers import Tokenizer
        from huggingface_hub import hf_hub_download
        from lm_common import _byte_decoder
        qtok = Tokenizer.from_file(hf_hub_download(QWEN, "tokenizer.json"))
        enc_bytes = {b: c for c, b in _byte_decoder().items()}
        old = model.get_input_embeddings().weight.data
        new = torch.empty(meta["vocab_size"], old.shape[1])
        mean = old.mean(0)
        n_fallback = 0
        for i, bts in enumerate(Encoder(a.tok).token_bytes()):
            if i == meta["eos"]:
                new[i] = old[qtok.token_to_id("<|endoftext|>")]
            elif not bts:
                new[i] = mean; n_fallback += 1
            else:
                ids = [t.id for t in qtok.model.tokenize("".join(enc_bytes[b] for b in bts))]
                new[i] = old[ids].mean(0) if ids else mean
        model.resize_token_embeddings(meta["vocab_size"])
        model.get_input_embeddings().weight.data.copy_(new)
        model.config.eos_token_id = model.config.bos_token_id = meta["eos"]
        print(f"transplanted {meta['vocab_size']} embeddings ({n_fallback} special/unknown -> mean); tied={model.config.tie_word_embeddings}", flush=True)
model.config.use_cache = False
model.to(dev)
n_params = sum(p.numel() for p in model.parameters())
emb_params = [model.get_input_embeddings().weight]
emb_ids = {id(p) for p in emb_params}
decay = [p for n, p in model.named_parameters() if id(p) not in emb_ids and p.dim() >= 2]
no_decay = [p for n, p in model.named_parameters() if id(p) not in emb_ids and p.dim() < 2]
opt = torch.optim.AdamW([dict(params=decay, lr=lr, weight_decay=0.1), dict(params=no_decay, lr=lr, weight_decay=0.0),
                         dict(params=emb_params, lr=emb_lr, weight_decay=0.0)], betas=(0.9, 0.95), eps=1e-8, fused=True)
base_lrs = [g["lr"] for g in opt.param_groups]
freeze_body = int(0.05 * steps) if a.mode == "transplant" else 0
fwd = torch.compile(model) if a.compile else model


def lr_at(s):
    if s < warm:
        return (s + 1) / warm
    p = (s - warm) / max(1, steps - warm)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))


log, t0, tok_seen = [], time.time(), 0
print(f"{run}: {n_params / 1e6:.1f}M params, {len(data):,} tokens, {steps} steps x {a.batch}x{a.seq}, micro {micro}", flush=True)
for s in range(steps):
    body_on = s >= freeze_body
    for g, base in zip(opt.param_groups, base_lrs):
        g["lr"] = base * lr_at(s) * (1.0 if (body_on or g["params"] is emb_params) else 0.0)
    for k in range(accum):
        idx = order[(s * a.batch + k * micro + np.arange(micro)) % n_blocks]
        x = torch.from_numpy(np.stack([data[i * a.seq:(i + 1) * a.seq].astype(np.int64) for i in idx])).to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = fwd(input_ids=x, labels=x).loss / accum
        loss.backward()
        tok_seen += x.numel()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    if s % 25 == 0 or s == steps - 1:
        el = time.time() - t0
        rec = dict(step=s, loss=loss.item() * accum, lr=opt.param_groups[0]["lr"], tok_per_s=tok_seen / el, elapsed_h=el / 3600)
        log.append(rec)
        print(json.dumps(rec), flush=True)
    if (s + 1) % 1000 == 0 and not a.max_steps:
        model.save_pretrained(f"{out_dir}/model")
model.save_pretrained(f"{out_dir}/model")
save_json(dict(run=run, tok=a.tok, mode=a.mode, params=n_params, steps=steps, tokens=tok_seen,
               chars=meta["n_chars"] * min(1.0, tok_seen / meta["n_tokens"]), hours=(time.time() - t0) / 3600,
               final_loss=log[-1]["loss"], log=log), f"{out_dir}/train_log.json")
print("done", run, flush=True)
