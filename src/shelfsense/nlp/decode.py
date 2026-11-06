"""Greedy generation with a KV cache, plus the parser wrapper with a safe fallback."""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch

from shelfsense.nlp.bpe import BPETokenizer
from shelfsense.nlp.gpt import GPT, GPTConfig
from shelfsense.nlp.schema import canonical, validate


@torch.no_grad()
def greedy_generate(model: GPT, prompts: list[list[int]], eos_id: int,
                    max_new: int = 100) -> list[list[int]]:
    """Greedy decoding for prompts that all have the same length (so no padding is needed)."""
    model.eval()
    length = len(prompts[0])
    if any(len(p) != length for p in prompts):
        raise ValueError("all prompts in a batch must have the same length")
    max_new = max(0, min(max_new, model.cfg.block_size - length))
    outs: list[list[int]] = [[] for _ in prompts]
    done = [False] * len(prompts)
    if max_new == 0:
        return outs
    logits, past = model(torch.tensor(prompts, dtype=torch.long), last_only=True)
    pos = length
    for step in range(max_new):
        nxt = logits[:, -1].argmax(dim=-1)
        for b, tok in enumerate(nxt.tolist()):
            if done[b]:
                continue
            if tok == eos_id:
                done[b] = True
            else:
                outs[b].append(tok)
        if all(done) or step == max_new - 1:
            break
        logits, past = model(nxt.unsqueeze(1), past=past, pos_offset=pos)
        pos += 1
    return outs


@dataclass
class Outcome:
    raw: str                 # exactly what the model wrote
    obj: dict | None         # validated, canonical parse (None if unusable)
    error: str | None


def finish(raw: str) -> Outcome:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return Outcome(raw, None, "invalid JSON")
    err = validate(obj)
    if err:
        return Outcome(raw, None, err)
    return Outcome(raw, canonical(obj), None)


class MiniGPTParser:
    def __init__(self, model: GPT, tok: BPETokenizer, max_new: int = 110):
        self.model, self.tok, self.max_new = model, tok, max_new

    @classmethod
    def load(cls, path) -> "MiniGPTParser":
        from safetensors.torch import load_file
        p = Path(path)
        cfg = GPTConfig(**json.loads((p / "config.json").read_text(encoding="utf-8")))
        model = GPT(cfg)
        model.load_state_dict(load_file(str(p / "model.safetensors")))
        model.eval()
        return cls(model, BPETokenizer.load(p / "tokenizer.json"))

    def _prompt(self, text: str) -> list[int]:
        return [self.tok.special_id("<bos>")] + self.tok.encode(text) + [self.tok.special_id("<sep>")]

    def parse_many(self, texts: list[str], batch_size: int = 64) -> list[Outcome]:
        prompts = [self._prompt(t) for t in texts]
        results: list[Outcome | None] = [None] * len(texts)
        buckets: dict[int, list[int]] = defaultdict(list)
        limit = self.model.cfg.block_size - 8
        for i, p in enumerate(prompts):
            if len(p) >= limit:
                results[i] = Outcome("", None, "prompt too long")
            else:
                buckets[len(p)].append(i)
        eos = self.tok.special_id("<eos>")
        for length, idxs in buckets.items():
            for s in range(0, len(idxs), batch_size):
                chunk = idxs[s:s + batch_size]
                gens = greedy_generate(self.model, [prompts[i] for i in chunk], eos, self.max_new)
                for i, g in zip(chunk, gens):
                    results[i] = finish(self.tok.decode(g))
        return results  # type: ignore[return-value]

    def parse(self, text: str) -> Outcome:
        return self.parse_many([text])[0]


def hybrid_parse(parser: MiniGPTParser, texts: list[str]) -> list[dict]:
    """GPT first; if its output is unusable, fall back to the rule-based parser."""
    from shelfsense.nlp import baseline
    outs = parser.parse_many(texts)
    return [o.obj if o.obj is not None else baseline.parse(t) for o, t in zip(outs, texts)]