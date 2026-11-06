"""Alternate names for products (Latin transliterations plus Tamil/Hindi script).

Keyed by substitute_group so every brand/size in a group shares the same aliases.
Review with a native speaker before relying on these.
"""
from __future__ import annotations

import sqlite3

ALIASES: dict[str, list[str]] = {
    "rice_5kg": ["arisi", "chawal", "அரிசி", "चावल"],
    "basmati_1kg": ["basmati", "arisi", "chawal", "அரிசி", "चावल"],
    "toor_dal_1kg": ["thuvaram paruppu", "paruppu", "arhar", "dal", "பருப்பு", "दाल"],
    "urad_dal_1kg": ["ulundhu", "ulutham paruppu", "paruppu", "dal", "உளுந்து", "उड़द", "दाल"],
    "moong_dal_500g": ["pasi paruppu", "paruppu", "green gram", "dal", "பாசிப்பருப்பு", "मूंग", "दाल"],
    "atta_5kg": ["wheat flour", "godhumai maavu", "chapati flour", "கோதுமை மாவு", "आटा"],
    "sugar_1kg": ["sakkarai", "chini", "cheeni", "சர்க்கரை", "சீனி", "चीनी", "शक्कर"],
    "salt_1kg": ["uppu", "namak", "உப்பு", "नमक"],
    "sunoil_1l": ["cooking oil", "oil", "ennai", "tel", "எண்ணெய்", "तेल"],
    "gingelly_oil_500ml": ["sesame oil", "nallennai", "ennai", "oil", "நல்லெண்ணெய்", "तिल का तेल"],
    "ragi_flour_1kg": ["kezhvaragu", "nachni", "கேழ்வரகு", "रागी"],
    "tea_250g": ["chai", "டீ", "चाय"],
    "coffee_200g": ["kaapi", "filter coffee", "காபி", "कॉफी"],
    "health_drink_500g": ["malt drink", "kids health drink"],
    "cola_750ml": ["cool drink", "cold drink", "soft drink", "soda"],
    "water_1l": ["thanni", "paani", "drinking water", "mineral water", "தண்ணீர்", "पानी"],
    "mango_drink_200ml": ["mango juice", "juice", "cool drink"],
    "milk_500ml": ["paal", "doodh", "dudh", "பால்", "दूध"],
    "curd_500g": ["thayir", "dahi", "yogurt", "தயிர்", "दही"],
    "paneer_200g": ["cottage cheese", "பன்னீர்", "पनीर"],
    "butter_500g": ["makkhan", "vennai", "வெண்ணெய்", "मक्खन"],
    "butter_100g": ["makkhan", "vennai", "வெண்ணெய்", "मक्खन"],
    "ghee_500ml": ["nei", "neyy", "நெய்", "घी"],
    "eggs_6": ["egg", "muttai", "anda", "முட்டை", "अंडा"],
    "bread_400g": ["bun", "pav", "double roti", "ப்ரெட்", "ब्रेड"],
    "cookies_200g": ["biscuit", "biscuits", "cookies", "பிஸ்கட்", "बिस्कुट"],
    "glucose_biscuit_250g": ["biscuit", "biscuits", "glucose", "பிஸ்கட்", "बिस्कुट"],
    "marie_250g": ["biscuit", "biscuits", "tea biscuit"],
    "instant_noodles_70g": ["noodles", "instant noodle", "நூடுல்ஸ்", "नूडल्स"],
    "chips_50g": ["chips", "wafers", "snacks"],
    "mixture_200g": ["namkeen", "snack", "மிக்சர்"],
    "detergent_1kg": ["washing powder", "clothes powder", "detergent"],
    "dishwash_bar": ["vessel wash", "dish wash", "dishwash", "bartan"],
    "toilet_cleaner_500ml": ["bathroom cleaner", "toilet wash"],
    "soap_100g": ["soap", "bath soap", "sabun", "சோப்பு", "साबुन"],
    "toothpaste_200g": ["tooth paste", "brush paste", "பற்பசை", "टूथपेस्ट"],
    "shampoo_180ml": ["hair wash", "ஷாம்பு"],
    "turmeric_100g": ["manjal", "haldi", "மஞ்சள்", "हल्दी"],
    "chilli_powder_200g": ["milagai thool", "mirchi", "red chilli", "மிளகாய் தூள்", "मिर्च"],
    "sambar_powder_200g": ["sambhar", "sambar masala", "சாம்பார் பொடி"],
    "mustard_100g": ["kadugu", "sarson", "கடுகு", "सरसों"],
    "tamarind_250g": ["puli", "imli", "புளி", "इमली"],
    "onion_1kg": ["vengayam", "pyaz", "pyaaz", "வெங்காயம்", "प्याज"],
    "tomato_1kg": ["thakkali", "tamatar", "தக்காளி", "टमाटर"],
    "potato_1kg": ["urulai", "aloo", "உருளைக்கிழங்கு", "आलू"],
    "coconut_1pc": ["thengai", "nariyal", "தேங்காய்", "नारियल"],
}


def seed_aliases(conn: sqlite3.Connection) -> int:
    """Insert aliases for every product (safe to run repeatedly). Returns rows attempted."""
    data = []
    for row in conn.execute("SELECT id, substitute_group FROM products").fetchall():
        for alias in ALIASES.get(row["substitute_group"], []):
            data.append((row["id"], alias))
    conn.executemany("INSERT OR IGNORE INTO product_aliases (product_id, alias) VALUES (?, ?)", data)
    conn.commit()
    return len(data)