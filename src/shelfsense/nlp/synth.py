"""Synthetic (messy text -> structured JSON) data for the query parser.

Deterministic for a given seed. Test sets deliberately differ from training:
  test_clean         same templates and items as training, no noise
  test_noisy         typos, odd casing, emoji
  test_new_phrasing  templates never used in training
  test_unseen_items  products the model never saw (it must COPY them)
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass

from shelfsense.nlp.schema import INTENTS, build_target, dumps_target
from shelfsense.nlp.text import normalize, words

# canonical name -> ways customers write it (English, transliterated, Tamil/Hindi script)
ITEMS: dict[str, list[str]] = {
    "milk": ["milk", "paal", "doodh", "பால்", "दूध"],
    "curd": ["curd", "thayir", "dahi", "தயிர்", "दही"],
    "rice": ["rice", "arisi", "chawal", "அரிசி", "चावल"],
    "sugar": ["sugar", "sakkarai", "chini", "சர்க்கரை", "चीनी"],
    "salt": ["salt", "uppu", "namak", "உப்பு", "नमक"],
    "toor dal": ["toor dal", "thuvaram paruppu", "arhar dal"],
    "urad dal": ["urad dal", "ulundhu", "உளுந்து"],
    "atta": ["atta", "wheat flour", "godhumai maavu"],
    "oil": ["oil", "cooking oil", "ennai", "tel", "எண்ணெய்", "तेल"],
    "ghee": ["ghee", "nei", "நெய்", "घी"],
    "butter": ["butter", "makkhan", "vennai"],
    "eggs": ["eggs", "egg", "muttai", "anda", "முட்டை", "अंडा"],
    "bread": ["bread", "pav", "double roti"],
    "tea powder": ["tea powder", "tea", "chai", "चाय"],
    "coffee powder": ["coffee powder", "coffee", "kaapi", "காபி"],
    "onion": ["onion", "vengayam", "pyaz", "வெங்காயம்", "प्याज"],
    "tomato": ["tomato", "thakkali", "tamatar", "தக்காளி", "टमाटर"],
    "potato": ["potato", "urulai", "aloo", "உருளைக்கிழங்கு", "आलू"],
    "coconut": ["coconut", "thengai", "nariyal", "தேங்காய்", "नारियल"],
    "turmeric powder": ["turmeric", "turmeric powder", "manjal", "haldi", "மஞ்சள்", "हल्दी"],
    "chilli powder": ["chilli powder", "milagai thool", "mirchi powder"],
    "tamarind": ["tamarind", "puli", "imli", "புளி", "इमली"],
    "mustard": ["mustard", "kadugu", "sarson", "கடுகு"],
    "noodles": ["noodles", "maggi", "நூடுல்ஸ்"],
    "biscuits": ["biscuits", "biscuit", "பிஸ்கட்", "बिस्कुट"],
    "soap": ["soap", "sabun", "சோப்பு", "साबुन"],
    "toothpaste": ["toothpaste", "tooth paste", "பற்பசை"],
    "shampoo": ["shampoo", "hair wash"],
    "detergent": ["detergent", "washing powder"],
    "water": ["water", "thanni", "paani", "தண்ணீர்", "पानी"],
    "cool drink": ["cool drink", "cold drink", "soft drink", "cola"],
    "paneer": ["paneer", "பன்னீர்", "पनीर"],
    "ragi flour": ["ragi flour", "ragi", "kezhvaragu"],
}
KNOWN_NAMES = sorted(ITEMS)

# Items the store does not stock. The model must learn to COPY unfamiliar names.
TRAIN_UNSEEN = ["millet noodles", "ragi biscuits", "peanut butter", "brown rice", "oat milk",
                "multigrain atta", "green tea", "quinoa", "cold pressed coconut oil",
                "protein bar", "jaggery powder", "almond milk"]
# Used ONLY in test_unseen_items.
TEST_UNSEEN = ["chia seeds", "soy milk", "honey", "corn flakes", "dates", "cashew nuts", "tofu",
               "dark chocolate", "olive oil", "cheese slices", "bananas", "apple juice"]

SIZES = ["500ml", "1l", "2l", "1kg", "5kg", "250g", "500g", "200g", "100g"]
QTYS = [2, 2, 3, 3, 4, 5]
JOINERS = {"en": [" and ", ", ", " & "], "ta-en": [" and ", ", "], "hi-en": [" aur ", ", "],
           "ta": [" மற்றும் ", ", "], "hi": [" और ", ", "]}
INTENT_WEIGHTS = [28, 18, 14, 22, 9, 9]   # matches INTENTS order


@dataclass(frozen=True)
class Template:
    intent: str
    lang: str          # en | ta-en | hi-en | ta (Tamil script) | hi (Devanagari)
    text: str          # may contain {items} and {offer}
    holdout: bool = False

    @property
    def has_offer(self) -> bool:
        return "{offer}" in self.text


def _t(intent: str, rows: list[tuple[str, str]], holdout: bool = False) -> list[Template]:
    return [Template(intent, lang, text, holdout) for lang, text in rows]


TEMPLATES: list[Template] = (
    _t("availability", [
        ("en", "do you have {items}?"), ("en", "{items} available?"), ("en", "is {items} in stock?"),
        ("en", "got {items}?"), ("en", "need {items}, available?"),
        ("ta-en", "anna {items} irukka?"), ("ta-en", "{items} kidaikkuma anna?"),
        ("ta-en", "{items} stock irukka?"), ("ta-en", "akka {items} irukka"),
        ("hi-en", "bhaiya {items} hai kya?"), ("hi-en", "{items} milega?"),
        ("hi-en", "{items} available hai?"),
        ("ta", "{items} இருக்கா?"), ("ta", "அண்ணா {items} இருக்கா"),
        ("hi", "भैया {items} है क्या?"), ("hi", "{items} मिलेगा?")])
    + _t("availability", [("en", "any {items} left?"), ("ta-en", "{items} irukkanu sollunga"),
                          ("hi-en", "{items} hai bhaiya?")], holdout=True)
    + _t("price", [
        ("en", "price of {items}?"), ("en", "{items} how much?"), ("en", "{items} rate?"),
        ("en", "what is the rate for {items}?"),
        ("ta-en", "{items} evlo anna?"), ("ta-en", "{items} vila enna?"),
        ("ta-en", "{items} price enna?"),
        ("hi-en", "{items} kitne ka hai?"), ("hi-en", "{items} ka rate kya hai?"),
        ("ta", "{items} விலை என்ன?"), ("hi", "{items} कितने का है?")])
    + _t("price", [("en", "how much is {items}?"), ("ta-en", "{items} rate evlo?"),
                   ("hi-en", "{items} ka price bataiye")], holdout=True)
    + _t("bargain", [
        ("en", "{items} for {offer} rupees?"), ("en", "can you do {items} at {offer}?"),
        ("en", "{items} {offer} rs final?"),
        ("ta-en", "{items} {offer} ku tharuveengala?"), ("ta-en", "{offer} la kudunga {items}"),
        ("hi-en", "{items} {offer} mein dedo"), ("hi-en", "{offer} rupaye mein {items} do"),
        ("en", "can you reduce the price of {items}?"), ("ta-en", "{items} konjam kammi pannunga"),
        ("hi-en", "{items} thoda kam karo")])
    + _t("bargain", [("en", "I'll pay {offer} for {items}"),
                     ("ta-en", "{items} {offer} rupaya ku mudiyuma")], holdout=True)
    + _t("order", [
        ("en", "give me {items}"), ("en", "I want {items}"), ("en", "please send {items}"),
        ("en", "need {items}"),
        ("ta-en", "{items} venum anna"), ("ta-en", "{items} anuppunga"),
        ("ta-en", "anna {items} kudunga"),
        ("hi-en", "{items} chahiye"), ("hi-en", "{items} bhej do"), ("hi-en", "bhaiya {items} do"),
        ("ta", "{items} வேணும்"), ("hi", "{items} चाहिए")])
    + _t("order", [("en", "reserve {items} for me"), ("ta-en", "{items} pack pannunga"),
                   ("hi-en", "{items} rakh dena")], holdout=True)
    + _t("greeting", [
        ("en", "hi"), ("en", "hello"), ("ta-en", "hello anna"), ("en", "hi bro"),
        ("ta-en", "vanakkam"), ("ta-en", "vanakkam anna"), ("hi-en", "namaste"),
        ("hi-en", "namaste bhaiya"), ("en", "good morning"), ("en", "good evening"),
        ("en", "hey"), ("ta", "வணக்கம்"), ("hi", "नमस्ते")])
    + _t("greeting", [("ta-en", "hiii anna"), ("en", "hello sir")], holdout=True)
    + _t("other", [
        ("en", "what time do you close?"), ("ta-en", "shop open ah?"),
        ("hi-en", "kitne baje band hota hai?"), ("en", "do you give home delivery?"),
        ("ta-en", "delivery irukka?"), ("en", "thanks"), ("en", "ok thanks bro"),
        ("ta-en", "upi accept pannuveengala?"), ("hi-en", "payment kaise karun?"),
        ("ta", "நன்றி"), ("hi", "धन्यवाद"), ("en", "where is the shop?")])
    + _t("other", [("en", "are you open on sunday?"), ("ta-en", "sunday leave ah?")], holdout=True)
)


def static_template_words(holdout: bool = False) -> frozenset[str]:
    """Words that appear in template text outside the {slots}. Handy for the rule-based baseline."""
    out: set[str] = set()
    for t in TEMPLATES:
        if t.holdout == holdout:
            out.update(words(normalize(re.sub(r"\{\w+\}", " ", t.text))))
    return frozenset(out)


# ----------------------------------------------------------------------------- building blocks
def _in_range(s: str, lo: int, hi: int) -> bool:
    return any(lo <= ord(c) <= hi for c in s)


def _forms_for(canon: str, lang: str) -> list[str]:
    forms = ITEMS[canon]
    latin = [f for f in forms if f.isascii()]
    if lang == "ta":
        return [f for f in forms if _in_range(f, 0x0B80, 0x0BFF)] or latin
    if lang == "hi":
        return [f for f in forms if _in_range(f, 0x0900, 0x097F)] or latin
    return latin


def _fmt_size(rng: random.Random, size: str) -> str:
    m = re.fullmatch(r"(\d+)([a-z]+)", size)
    num, unit = m.group(1), m.group(2)
    style = rng.choice(["compact", "spaced", "cap"])
    if style == "compact":
        return f"{num}{unit}"
    if style == "spaced":
        return f"{num} {unit}"
    return f"{num} {unit.capitalize()}"


def _item_phrase(rng: random.Random, canon: str, surface: str) -> tuple[str, dict]:
    r = rng.random()
    qty, size = 1, None
    if r < 0.45:
        text = surface
    elif r < 0.70:
        qty = rng.choice(QTYS)
        text = f"{qty} {surface}"
    elif r < 0.78:
        qty = rng.choice(QTYS)
        text = f"{qty} packets {surface}"
    elif r < 0.84:
        qty = rng.choice(QTYS)
        text = f"{surface} {qty} packet"
    elif r < 0.92:
        size = rng.choice(SIZES)
        text = f"{_fmt_size(rng, size)} {surface}"
    else:
        size = rng.choice(SIZES)
        text = f"{surface} {_fmt_size(rng, size)}"
    item = {"name": canon, "qty": qty}
    if size:
        item["size"] = size
    return text, item


def _pick_item(rng: random.Random, mode: str, lang: str, used: set[str]):
    p_test, p_copy = {"unseen": (0.6, 0.0), "known": (0.0, 0.0)}.get(mode, (0.0, 0.2))
    for _ in range(50):
        r = rng.random()
        if r < p_test:
            canon = rng.choice(TEST_UNSEEN)
            surface, known = canon, False
        elif r < p_test + p_copy:
            canon = rng.choice(TRAIN_UNSEEN)
            surface, known = canon, False
        else:
            canon = rng.choice(KNOWN_NAMES)
            surface, known = rng.choice(_forms_for(canon, lang)), True
        if canon not in used:
            used.add(canon)
            return canon, surface, known
    raise RuntimeError("could not pick a distinct item")


def _mutate(rng: random.Random, core: str) -> str:
    op = rng.choice(["case", "drop", "swap", "dup", "vowel"])
    n = len(core)
    if op == "case":
        return core.upper() if rng.random() < 0.5 else core.capitalize()
    if op == "drop" and n >= 5:
        i = rng.randrange(1, n - 1)
        return core[:i] + core[i + 1:]
    if op == "swap" and n >= 4:
        i = rng.randrange(1, n - 2)
        return core[:i] + core[i + 1] + core[i] + core[i + 2:]
    if op == "dup":
        i = rng.randrange(n)
        return core[:i + 1] + core[i] + core[i + 1:]
    if op == "vowel":
        m = re.search(r"(?<=.)[aeiou]", core)
        if m:
            return core[:m.start()] + core[m.end():]
    return core


def add_noise(rng: random.Random, text: str, p: float, protect=frozenset()) -> str:
    """Typos/casing/emoji. Tokens with digits, non-ASCII tokens and `protect`ed words stay intact."""
    if p <= 0:
        return text
    out = []
    for tok in text.split(" "):
        m = re.fullmatch(r"(\W*)(\w+)(\W*)", tok) if tok.isascii() else None
        if (not m or any(ch.isdigit() for ch in tok) or m.group(2).lower() in protect
                or rng.random() >= p):
            out.append(tok)
        else:
            out.append(m.group(1) + _mutate(rng, m.group(2)) + m.group(3))
    s = " ".join(out)
    if rng.random() < p:
        s += rng.choice(["??", "!", " 🙏", " 😊", "..."])
    return s


def build_example(rng: random.Random, *, holdout_templates: bool, item_mode: str,
                  noise_p: float) -> dict:
    intent = rng.choices(INTENTS, weights=INTENT_WEIGHTS)[0]
    tpl = rng.choice([t for t in TEMPLATES if t.intent == intent and t.holdout == holdout_templates])
    items: list[dict] = []
    protect: set[str] = set()
    items_text = ""
    if "{items}" in tpl.text:
        used: set[str] = set()
        phrases = []
        for _ in range(rng.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]):
            canon, surface, known = _pick_item(rng, item_mode, tpl.lang, used)
            phrase, item = _item_phrase(rng, canon, surface)
            phrases.append(phrase)
            items.append(item)
            if not known:
                protect.update(w.lower() for w in canon.split())
        items_text = rng.choice(JOINERS[tpl.lang]).join(phrases)
    offer = rng.choice(range(10, 505, 5)) if tpl.has_offer else None
    text = tpl.text.format(items=items_text, offer=offer)
    if rng.random() < 0.15:
        text = text[:1].upper() + text[1:]
    text = add_noise(rng, text, noise_p, frozenset(protect))
    obj = build_target(intent, items, offer)
    return {"text": text, "target": dumps_target(obj), "obj": obj, "intent": intent,
            "lang": tpl.lang, "template": tpl.text}


def make_splits(seed: int = 1337, n_train: int = 30000, n_val: int = 500,
                n_test: int = 400) -> dict[str, list[dict]]:
    def gen(name, n, *, holdout=False, mode="train", noise_p=0.0, mix_clean=0.0, avoid=None):
        rng = random.Random(f"{seed}:{name}")
        out: list[dict] = []
        attempts = 0
        while len(out) < n and attempts < n * 40:
            attempts += 1
            p = 0.0 if rng.random() < mix_clean else noise_p
            ex = build_example(rng, holdout_templates=holdout, item_mode=mode, noise_p=p)
            # no leakage: test queries with items must not appear verbatim in training
            if avoid is not None and ex["obj"]["items"] and ex["text"] in avoid:
                continue
            out.append(ex)
        return out

    train = gen("train", n_train, noise_p=0.12, mix_clean=0.5)
    avoid = {ex["text"] for ex in train if ex["obj"]["items"]}
    return {
        "train": train,
        "val": gen("val", n_val, noise_p=0.12, mix_clean=0.5, avoid=avoid),
        "test_clean": gen("test_clean", n_test, avoid=avoid),
        "test_noisy": gen("test_noisy", n_test, noise_p=0.25, avoid=avoid),
        "test_new_phrasing": gen("test_new_phrasing", n_test, holdout=True, noise_p=0.05,
                                 avoid=avoid),
        "test_unseen_items": gen("test_unseen_items", n_test, mode="unseen", avoid=avoid),
    }