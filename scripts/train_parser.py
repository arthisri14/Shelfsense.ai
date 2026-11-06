"""Train the from-scratch mini-GPT query parser on CPU.

Smoke test:  python scripts\\train_parser.py --quick --out models\\parser_quick
Full run:    python scripts\\train_parser.py
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from shelfsense.nlp.bpe import BPETokenizer
from shelfsense.nlp.data import IGNORE, collate, encode_example, iter_batches
from shelfsense.nlp.decode import MiniGPTParser
from shelfsense.nlp.gpt import GPT, GPTConfig
from shelfsense.nlp.synth import make_splits


def lr_at(step: int, total: int, base: float, warmup: int = 100, floor: float = 0.1) -> float:
    if step < warmup:
        return base * step / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p)))


@torch.no_grad()
def eval_loss(model, examples, pad_id: int, bs: int = 128) -> float:
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(examples), bs):
        x, y = collate(examples[i:i + bs], pad_id)
        logits, _ = model(x)
        total += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                                 ignore_index=IGNORE, reduction="sum").item()
        count += int((y != IGNORE).sum())
    model.train()
    return total / max(count, 1)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/parser_v0")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--vocab", type=int, default=1200)
    ap.add_argument("--n-train", type=int, default=30000)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--block", type=int, default=192)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--embd", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--quick", action="store_true", help="small, fast smoke test")
    args = ap.parse_args()
    if args.quick:
        args.steps, args.n_train, args.n_val, args.n_test, args.vocab = 300, 3000, 200, 100, 700

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("generating data...")
    splits = make_splits(args.seed, args.n_train, args.n_val, args.n_test)

    print("training the BPE tokenizer (about a minute)...")
    corpus = [e["text"] for e in splits["train"]] + [e["target"] for e in splits["train"]]
    tok = BPETokenizer().train(corpus, vocab_size=args.vocab, min_freq=2)
    print(f"  vocab size {tok.vocab_size}")

    def encode(exs):
        out = [encode_example(tok, e["text"], e["target"], args.block) for e in exs]
        kept = [o for o in out if o is not None]
        return kept, len(out) - len(kept)

    train, dropped = encode(splits["train"])
    val, _ = encode(splits["val"])
    print(f"  train sequences {len(train)} (dropped {dropped} longer than {args.block} tokens)")
    print(f"  avg length {sum(len(i) for i, _ in train) / len(train):.1f} tokens")

    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=args.block, n_layer=args.layers,
                    n_head=args.heads, n_embd=args.embd)
    model = GPT(cfg)
    print(f"model parameters: {model.n_params():,}")
    pad_id = tok.special_id("<pad>")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    batches = iter_batches(train, args.batch, pad_id, random.Random(args.seed))

    model.train()
    t0 = time.time()
    best_val = float("inf")
    for step in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, args.lr)
        x, y = next(batches)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                               ignore_index=IGNORE)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            per = (time.time() - t0) / step
            print(f"step {step:5d}/{args.steps} loss {loss.item():.4f}  "
                  f"{per:.2f}s/step  eta {per * (args.steps - step) / 60:.1f} min")
        if step % 500 == 0 or step == args.steps:
            best_val = eval_loss(model, val, pad_id)
            print(f"  validation loss {best_val:.4f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()},
              str(out / "model.safetensors"))
    (out / "config.json").write_text(json.dumps(asdict(cfg)), encoding="utf-8")
    tok.save(out / "tokenizer.json")
    (out / "meta.json").write_text(json.dumps({
        "seed": args.seed, "n_train": args.n_train, "n_val": args.n_val, "n_test": args.n_test,
        "steps": args.steps, "val_loss": best_val, "params": model.n_params()}), encoding="utf-8")
    print(f"\nsaved to {out}")

    parser = MiniGPTParser(model, tok)
    print("\nsample parses (validation texts):")
    for e in splits["val"][:6]:
        o = parser.parse(e["text"])
        print(f"  {e['text']!r}\n    gold: {e['target']}\n    gpt : {o.raw}")


if __name__ == "__main__":
    main()