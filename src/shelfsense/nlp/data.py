"""Turn (text, JSON target) pairs into training tensors."""
from __future__ import annotations

import random

import torch

IGNORE = -100


def encode_example(tok, text: str, target: str, block_size: int):
    """Sequence: <bos> query <sep> json <eos>. Only the JSON part (and <eos>) is trained on."""
    prompt = [tok.special_id("<bos>")] + tok.encode(text) + [tok.special_id("<sep>")]
    answer = tok.encode(target) + [tok.special_id("<eos>")]
    ids = prompt + answer
    if len(ids) > block_size:
        return None
    return ids, [IGNORE] * len(prompt) + answer


def collate(batch, pad_id: int):
    """Right-pad; inputs are ids[:-1], targets are labels[1:] (next-token prediction)."""
    width = max(len(ids) for ids, _ in batch) - 1
    x = torch.full((len(batch), width), pad_id, dtype=torch.long)
    y = torch.full((len(batch), width), IGNORE, dtype=torch.long)
    for i, (ids, labels) in enumerate(batch):
        n = len(ids) - 1
        x[i, :n] = torch.tensor(ids[:-1], dtype=torch.long)
        y[i, :n] = torch.tensor(labels[1:], dtype=torch.long)
    return x, y


def iter_batches(examples, batch_size: int, pad_id: int, rng: random.Random):
    """Endless shuffled batches."""
    while True:
        order = list(range(len(examples)))
        rng.shuffle(order)
        for s in range(0, len(order), batch_size):
            yield collate([examples[i] for i in order[s:s + batch_size]], pad_id)