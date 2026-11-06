import random

import pytest

from shelfsense.nlp import baseline
from shelfsense.nlp.metrics import score_split
from shelfsense.nlp.schema import canonical, dumps_target, validate
from shelfsense.nlp.synth import (
    TEST_UNSEEN, add_noise, make_splits, static_template_words,
)


@pytest.fixture(scope="module")
def splits():
    return make_splits(seed=5, n_train=300, n_val=30, n_test=60)


# ------------------------------------------------------------------ schema
def test_schema_accepts_good_and_rejects_bad():
    good = {"intent": "order", "items": [{"name": "milk", "qty": 2, "size": "500ml"}]}
    assert validate(good) is None
    assert canonical(good) == good
    assert validate({"intent": "order", "items": [{"name": "milk", "qty": 0}]}) is not None
    assert validate({"intent": "nope", "items": []}) is not None
    assert validate({"intent": "order", "items": [{"name": "milk", "qty": True}]}) is not None
    assert validate({"intent": "order", "items": [], "extra": 1}) is not None
    assert validate({"intent": "bargain", "items": [], "offer": -5}) is not None
    assert validate([1, 2]) is not None


# ------------------------------------------------------------------ synthetic data
def test_splits_are_deterministic(splits):
    assert make_splits(seed=5, n_train=300, n_val=30, n_test=60) == splits


def test_every_target_is_valid_and_round_trips(splits):
    import json
    for exs in splits.values():
        for e in exs:
            assert validate(e["obj"]) is None
            assert json.loads(e["target"]) == e["obj"]
            assert e["target"] == dumps_target(e["obj"])


def test_test_sets_do_not_leak_training_queries(splits):
    train_texts = {e["text"] for e in splits["train"] if e["obj"]["items"]}
    for name, exs in splits.items():
        if name.startswith("test"):
            for e in exs:
                if e["obj"]["items"]:
                    assert e["text"] not in train_texts


def test_new_phrasing_and_unseen_items_are_really_held_out(splits):
    train_templates = {e["template"] for e in splits["train"]}
    assert not ({e["template"] for e in splits["test_new_phrasing"]} & train_templates)
    train_names = {i["name"] for e in splits["train"] for i in e["obj"]["items"]}
    assert not (set(TEST_UNSEEN) & train_names)
    unseen_names = {i["name"] for e in splits["test_unseen_items"] for i in e["obj"]["items"]}
    assert unseen_names & set(TEST_UNSEEN)
    assert "reserve" not in static_template_words(False)
    assert "reserve" in static_template_words(True)


def test_noise_keeps_numbers_intact():
    out = add_noise(random.Random(1), "rice 5kg for 300 rupees", 1.0)
    assert "5kg" in out and "300" in out.split()


# ------------------------------------------------------------------ rule-based baseline
def test_baseline_examples():
    assert baseline.parse("do you have milk?") == {
        "intent": "availability", "items": [{"name": "milk", "qty": 1}]}
    assert baseline.parse("anna 2 paal irukka?") == {
        "intent": "availability", "items": [{"name": "milk", "qty": 2}]}
    assert baseline.parse("rice 5kg for 300 rupees?") == {
        "intent": "bargain", "items": [{"name": "rice", "qty": 1, "size": "5kg"}], "offer": 300}
    assert baseline.parse("hello") == {"intent": "greeting", "items": []}
    assert baseline.parse("do you have quinoa")["items"] == [{"name": "quinoa", "qty": 1}]


def test_baseline_output_is_always_valid(splits):
    for exs in splits.values():
        for e in exs:
            assert validate(baseline.parse(e["text"])) is None


# ------------------------------------------------------------------ metrics
def test_metrics():
    gold = [{"intent": "order", "items": [{"name": "milk", "qty": 2}]},
            {"intent": "price", "items": [{"name": "rice", "qty": 1, "size": "5kg"}]}]
    perfect = score_split(gold, gold)
    assert perfect["exact_match"] == 1.0 and perfect["name_f1"] == 1.0
    one_bad = score_split([gold[0], None], gold)
    assert one_bad["valid_rate"] == 0.5 and one_bad["exact_match"] == 0.5
    wrong_qty = score_split([{"intent": "order", "items": [{"name": "milk", "qty": 3}]}, gold[1]],
                            gold)
    assert wrong_qty["qty_acc"] == 0.5 and wrong_qty["name_f1"] == 1.0


# ------------------------------------------------------------------ the GPT (needs torch)
def _tiny_model(dropout=0.0):
    torch = pytest.importorskip("torch")
    from shelfsense.nlp.gpt import GPT, GPTConfig
    torch.manual_seed(0)
    return torch, GPT(GPTConfig(vocab_size=24, block_size=24, n_layer=2, n_head=2, n_embd=48,
                                dropout=dropout))


def test_gpt_shapes():
    torch, m = _tiny_model()
    logits, presents = m(torch.randint(0, 24, (3, 10)))
    assert logits.shape == (3, 10, 24) and len(presents) == 2
    assert m.n_params() > 0
    with pytest.raises(ValueError):
        m(torch.randint(0, 24, (1, 25)))


def test_gpt_is_causal():
    torch, m = _tiny_model()
    m.eval()
    a = torch.randint(0, 24, (1, 10))
    b = a.clone()
    b[0, 7] = (b[0, 7] + 1) % 24                       # change a LATER token
    la, _ = m(a)
    lb, _ = m(b)
    assert torch.allclose(la[:, :7], lb[:, :7], atol=1e-5)


def test_kv_cache_matches_full_forward():
    torch, m = _tiny_model()
    m.eval()
    idx = torch.randint(0, 24, (2, 8))
    full, _ = m(idx)
    _, past = m(idx[:, :5])
    for t in range(5, 8):
        step, past = m(idx[:, t:t + 1], past=past, pos_offset=t)
        assert torch.allclose(step[:, 0], full[:, t], atol=1e-4)


def test_safetensors_round_trip(tmp_path):
    torch, m = _tiny_model()
    st = pytest.importorskip("safetensors.torch")
    from shelfsense.nlp.gpt import GPT
    m.eval()
    path = str(tmp_path / "m.safetensors")
    st.save_file({k: v.contiguous() for k, v in m.state_dict().items()}, path)
    m2 = GPT(m.cfg)
    m2.load_state_dict(st.load_file(path))
    m2.eval()
    x = torch.randint(0, 24, (1, 6))
    assert torch.allclose(m(x)[0], m2(x)[0], atol=1e-6)


def test_tiny_model_memorises_and_generates():
    import torch.nn.functional as F
    from shelfsense.nlp.decode import greedy_generate
    torch, m = _tiny_model()
    eos = 1
    prompts = [[2, 5, 6, 7], [2, 8, 9, 10], [2, 11, 12, 13], [2, 14, 15, 16]]
    answers = [[17, 18, 19, 20, 21], [21, 20, 19, 18, 17], [18, 18, 19, 19, 20], [22, 17, 22, 18, 22]]
    seqs = [p + a + [eos] for p, a in zip(prompts, answers)]
    x = torch.tensor([s[:-1] for s in seqs])
    y = torch.tensor([[-100] * (len(p) - 1) + s[len(p):] for p, s in zip(prompts, seqs)])
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    m.train()
    for _ in range(500):
        logits, _ = m(x)
        loss = F.cross_entropy(logits.reshape(-1, 24), y.reshape(-1), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert greedy_generate(m, prompts, eos_id=eos, max_new=10) == answers


def test_finish_rejects_bad_model_output():
    pytest.importorskip("torch")
    from shelfsense.nlp.decode import finish
    assert finish("not json").obj is None
    assert finish('{"intent":"order","items":[{"name":"milk","qty":0}]}').obj is None
    ok = finish('{"intent":"order","items":[{"name":"milk","qty":2}]}')
    assert ok.obj == {"intent": "order", "items": [{"name": "milk", "qty": 2}]}