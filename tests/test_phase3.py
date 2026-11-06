import json
from datetime import date

import numpy as np
import pytest

from shelfsense.db.connection import get_connection, init_db
from shelfsense.nlp.bpe import BPETokenizer
from shelfsense.nlp.text import pieces, words
from shelfsense.search.aliases import seed_aliases
from shelfsense.search.bm25 import BM25Index, bpe_analyzer
from shelfsense.search.hybrid import rrf_fuse
from shelfsense.search.product_search import Hit, ProductSearch
from shelfsense.search.vector_index import HashingEmbedder, VectorIndex
from shelfsense.simulator.generate import generate

END = date(2026, 11, 2)
SAMPLES = ["aavin milk 500ml", "anna rice 5kg irukka?", "பால் இருக்கா?", "भैया दूध है क्या?",
           "milk 🙂 ok", "  spaced   out\nlines  ", "", "x"]
CORPUS = ["aavin milk 500 ml", "do you have butter?", "anna rice 5kg irukka?",
          "பால் இருக்கா", "bhaiya doodh hai kya?"] * 6


def make_db(days: int = 30):
    c = get_connection(":memory:")
    init_db(c)
    generate(c, end_date=END, days=days, seed=7)
    seed_aliases(c)
    return c


# ------------------------------------------------------------------ text
def test_words_basic():
    assert words("aavin milk 500ml") == ["aavin", "milk", "500", "ml"]
    assert words("5kg") == ["5", "kg"]
    assert words("anna milk irukka?") == ["anna", "milk", "irukka"]


def test_words_keep_tamil_and_hindi_whole():
    assert words("பால் இருக்கா?") == ["பால்", "இருக்கா"]
    assert words("भैया दूध है क्या?") == ["भैया", "दूध", "है", "क्या"]


def test_pieces_are_lossless():
    for s in SAMPLES:
        assert "".join(pieces(s)) == s


def test_pieces_attach_leading_space_to_next_word():
    assert pieces("hi there") == ["hi", " there"]


# ------------------------------------------------------------------ BPE
@pytest.fixture(scope="module")
def tok():
    return BPETokenizer().train(CORPUS, vocab_size=700)


def test_bpe_round_trip_all_languages(tok):
    for s in SAMPLES + ["quinoa ☃ 😀 unseen words"]:
        assert tok.decode(tok.encode(s)) == s


def test_bpe_learns_merges(tok):
    assert tok.vocab_size <= 700
    assert len(tok.encode(" milk")) < len(" milk".encode("utf-8"))


def test_bpe_training_is_deterministic(tok):
    assert BPETokenizer().train(CORPUS, vocab_size=700).merges == tok.merges


def test_bpe_save_and_load(tok, tmp_path):
    p = tmp_path / "tok.json"
    tok.save(p)
    loaded = BPETokenizer.load(p)
    assert loaded.merges == tok.merges
    for s in SAMPLES:
        assert loaded.encode(s) == tok.encode(s)
    assert json.loads(p.read_text(encoding="utf-8"))["version"] == 1


def test_bpe_special_tokens(tok):
    ids = tok.encode("milk", add_special=True)
    assert ids[0] == tok.special_id("<bos>") and ids[-1] == tok.special_id("<eos>")
    assert tok.decode(ids) == "milk"
    assert tok.special_id("<pad>") == 0
    with pytest.raises(ValueError):
        tok.decode([10 ** 6])


def test_bpe_encode_word_has_no_leading_space(tok):
    assert tok.encode_word("milk") == tok.encode("milk")
    assert tok.encode(" milk") != tok.encode("milk")


# ------------------------------------------------------------------ BM25
DOCS = {1: "Aavin Milk Packet 500 ml", 2: "Heritage Milk Packet 500 ml",
        3: "Lakshmi Ponni Boiled Rice 5 kg"}


def test_bm25_ranks_relevant_first():
    idx = BM25Index().build(DOCS)
    assert idx.search("ponni")[0][0] == 3
    assert idx.search("aavin milk")[0][0] == 1


def test_bm25_unknown_and_empty():
    idx = BM25Index().build(DOCS)
    assert idx.search("quinoa") == []
    assert idx.search("") == []
    assert BM25Index().build({}).search("milk") == []


def test_bm25_rare_term_outranks_common_term():
    scores = dict(BM25Index().build(DOCS).search("milk ponni"))
    assert scores[3] > scores[1]


def test_bpe_analyzer_is_case_insensitive(tok):
    an = bpe_analyzer(tok)
    assert an("MILK 5kg") == an("milk 5kg") and len(an("milk")) >= 1


# ------------------------------------------------------------------ vectors
def test_embedder_is_deterministic_and_unit_norm():
    a = HashingEmbedder().embed(["amul butter 500 g"])
    b = HashingEmbedder().embed(["amul butter 500 g"])
    assert np.allclose(a, b)
    assert abs(float(np.linalg.norm(a[0])) - 1.0) < 1e-5


def test_embedder_prefers_typo_to_unrelated():
    v = HashingEmbedder().embed(["butter", "buter", "rice"])
    assert float(v[0] @ v[1]) > float(v[0] @ v[2])


def test_empty_text_embeds_to_zero():
    assert not HashingEmbedder().embed(["???"])[0].any()


def test_vector_index_search_upsert_remove(tmp_path):
    idx = VectorIndex().build({1: "amul butter 500 g", 2: "lakshmi ponni rice 5 kg",
                               3: "aavin milk 500 ml"})
    assert idx.search("buter")[0][0] == 1
    idx.upsert(4, "salted buter")
    assert idx.search("buter")[0][0] == 4
    idx.upsert(1, "plain rice")
    assert 1 in [d for d, _ in idx.search("rice", k=5)]
    assert idx.remove(4) is True and idx.remove(4) is False
    assert 4 not in [d for d, _ in idx.search("buter", k=10)]
    p = tmp_path / "v.npz"
    idx.save(p)
    assert VectorIndex.load(p).search("rice") == idx.search("rice")


def test_rrf_fuse():
    assert [d for d, _ in rrf_fuse([[1, 2, 3], [3, 2, 1]])] == [1, 3, 2]
    assert [d for d, _ in rrf_fuse([[7], [8, 7]])][0] == 7


# ------------------------------------------------------------------ product search
@pytest.fixture(scope="module")
def conn():
    c = make_db()
    yield c
    c.close()


@pytest.fixture(scope="module")
def ps(conn):
    return ProductSearch(conn, store_id=1)


def test_names_aliases_and_scripts_find_milk(ps):
    for q in ["milk", "paal", "பால்", "दूध", "doodh"]:
        res = ps.search(q)
        assert res.status == "found", q
        assert res.hits[0].name == "Milk Packet", q


def test_typo_is_still_found(ps):
    res = ps.search("buter")
    assert res.status == "found"
    assert any(h.name == "Butter" for h in res.hits[:3])


def test_code_mixed_query(ps):
    res = ps.search("anna rice 5kg irukka?")
    assert res.status == "found"
    assert "Rice" in res.hits[0].name


def test_brand_and_pack_query(ps):
    res = ps.search("aavin milk 500ml")
    assert ("Aavin", "Milk Packet") in [(h.brand, h.name) for h in res.hits[:2]]


def test_missing_items_are_not_reported_as_found(ps):
    assert ps.search("millet noodles").status in ("partial", "not_found")
    assert ps.search("peanut butter").status in ("partial", "not_found")
    r = ps.search("do you have quinoa")
    assert r.status == "not_found" and r.unmatched == ("quinoa",)


def test_filler_only_query_is_empty(ps):
    r = ps.search("do you have?")
    assert r.status == "empty" and r.hits == ()


def test_hits_expose_no_sensitive_fields():
    fields = set(Hit.__dataclass_fields__)
    for forbidden in ("floor_price", "unit_cost", "mrp", "qty_on_hand", "reorder_level"):
        assert forbidden not in fields


def test_stock_is_read_live_not_from_the_index():
    c = make_db()
    ps2 = ProductSearch(c, store_id=1)
    pid = c.execute("SELECT id FROM products WHERE name = 'Milk Packet' AND brand = 'Aavin'"
                    ).fetchone()[0]
    c.execute("INSERT INTO stock_batches (store_id, product_id, qty, unit_cost, received_date, "
              "expiry_date) VALUES (1, ?, 5, 10.0, '2026-11-01', '2026-11-10')", (pid,))
    c.commit()
    hit = next(h for h in ps2.search("aavin milk").hits if h.product_id == pid)
    assert hit.in_stock is True
    c.execute("UPDATE stock_batches SET qty = 0 WHERE store_id = 1 AND product_id = ?", (pid,))
    c.commit()
    hit = next(h for h in ps2.search("aavin milk").hits if h.product_id == pid)   # same index
    assert hit.in_stock is False
    c.close()


def test_products_a_store_does_not_sell_are_never_returned(conn):
    row = conn.execute(
        "SELECT sp.product_id, p.name, p.brand FROM store_products sp "
        "JOIN products p ON p.id = sp.product_id WHERE sp.store_id = 2 AND sp.active = 0 LIMIT 1"
    ).fetchone()
    if row is None:
        pytest.skip("store 2 sells every product in this seed")
    res = ProductSearch(conn, store_id=2).search(f"{row['brand']} {row['name']}")
    assert row["product_id"] not in [h.product_id for h in res.hits]