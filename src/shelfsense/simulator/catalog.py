"""Static catalog and business rules for the synthetic store.

Everything here is made-up demo data. Prices are illustrative, not real market prices.
Festival dates are approximate: edit them freely.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class ProductDef:
    name: str
    brand: str
    category: str
    pack: str
    mrp: float
    shelf_life: int      # days
    popularity: float    # relative sales weight
    group: str           # substitute group

    @property
    def is_perishable(self) -> bool:
        return self.shelf_life <= 30


# name, brand, category, pack, mrp, shelf_life_days, popularity, substitute_group
_RAW = [
    # Staples
    ("Ponni Boiled Rice", "Lakshmi", "Staples", "5 kg", 360, 365, 5.0, "rice_5kg"),
    ("Sona Masoori Rice", "Lakshmi", "Staples", "5 kg", 410, 365, 4.0, "rice_5kg"),
    ("Basmati Rice", "India Gate", "Staples", "1 kg", 175, 365, 1.5, "basmati_1kg"),
    ("Toor Dal", "Tata Sampann", "Staples", "1 kg", 185, 270, 3.5, "toor_dal_1kg"),
    ("Toor Dal", "Fortune", "Staples", "1 kg", 172, 270, 2.5, "toor_dal_1kg"),
    ("Urad Dal", "Tata Sampann", "Staples", "1 kg", 168, 270, 2.5, "urad_dal_1kg"),
    ("Moong Dal", "Tata Sampann", "Staples", "500 g", 92, 270, 1.5, "moong_dal_500g"),
    ("Wheat Atta", "Aashirvaad", "Staples", "5 kg", 295, 120, 3.5, "atta_5kg"),
    ("Wheat Atta", "Pillsbury", "Staples", "5 kg", 278, 120, 2.0, "atta_5kg"),
    ("Sugar", "Parry's", "Staples", "1 kg", 52, 365, 5.0, "sugar_1kg"),
    ("Iodised Salt", "Tata", "Staples", "1 kg", 28, 365, 4.5, "salt_1kg"),
    ("Sunflower Oil", "Fortune", "Staples", "1 L", 148, 270, 4.5, "sunoil_1l"),
    ("Sunflower Oil", "Gold Winner", "Staples", "1 L", 142, 270, 3.0, "sunoil_1l"),
    ("Gingelly Oil", "Idhayam", "Staples", "500 ml", 260, 270, 1.5, "gingelly_oil_500ml"),
    ("Ragi Flour", "Nature Fresh", "Staples", "1 kg", 70, 180, 1.0, "ragi_flour_1kg"),
    # Beverages
    ("Tea Powder", "Tata Tea Gold", "Beverages", "250 g", 150, 365, 3.0, "tea_250g"),
    ("Tea Powder", "Red Label", "Beverages", "250 g", 138, 365, 2.5, "tea_250g"),
    ("Filter Coffee Powder", "Bru", "Beverages", "200 g", 165, 270, 2.0, "coffee_200g"),
    ("Health Drink", "Bournvita", "Beverages", "500 g", 255, 365, 1.2, "health_drink_500g"),
    ("Cola", "Coca-Cola", "Beverages", "750 ml", 40, 180, 2.5, "cola_750ml"),
    ("Cola", "Pepsi", "Beverages", "750 ml", 40, 180, 2.0, "cola_750ml"),
    ("Packaged Water", "Bisleri", "Beverages", "1 L", 20, 365, 4.0, "water_1l"),
    ("Mango Drink", "Frooti", "Beverages", "200 ml", 20, 180, 2.0, "mango_drink_200ml"),
    # Dairy
    ("Milk Packet", "Aavin", "Dairy", "500 ml", 25, 3, 12.0, "milk_500ml"),
    ("Milk Packet", "Heritage", "Dairy", "500 ml", 28, 4, 5.0, "milk_500ml"),
    ("Curd Packet", "Aavin", "Dairy", "500 g", 35, 6, 5.0, "curd_500g"),
    ("Paneer", "Amul", "Dairy", "200 g", 95, 10, 2.5, "paneer_200g"),
    ("Butter", "Amul", "Dairy", "500 g", 285, 90, 1.2, "butter_500g"),
    ("Butter", "Amul", "Dairy", "100 g", 58, 90, 2.5, "butter_100g"),
    ("Ghee", "Aavin", "Dairy", "500 ml", 330, 180, 1.2, "ghee_500ml"),
    ("Eggs", "Farm Fresh", "Dairy", "6 pcs", 48, 14, 5.0, "eggs_6"),
    # Bakery
    ("Sandwich Bread", "Modern", "Bakery", "400 g", 45, 5, 4.0, "bread_400g"),
    ("Sandwich Bread", "Britannia", "Bakery", "400 g", 48, 5, 3.0, "bread_400g"),
    # Snacks
    ("Good Day Butter Cookies", "Britannia", "Snacks", "200 g", 40, 150, 3.0, "cookies_200g"),
    ("Parle-G Biscuits", "Parle", "Snacks", "250 g", 25, 150, 4.0, "glucose_biscuit_250g"),
    ("Marie Gold Biscuits", "Britannia", "Snacks", "250 g", 40, 150, 2.0, "marie_250g"),
    ("Instant Noodles", "Maggi", "Snacks", "70 g", 14, 180, 6.0, "instant_noodles_70g"),
    ("Instant Noodles", "Yippee", "Snacks", "70 g", 14, 180, 2.5, "instant_noodles_70g"),
    ("Potato Chips", "Lays", "Snacks", "52 g", 20, 90, 3.5, "chips_50g"),
    ("Corn Puffs", "Kurkure", "Snacks", "50 g", 20, 90, 2.5, "chips_50g"),
    ("Mixture", "Haldiram's", "Snacks", "200 g", 70, 90, 1.5, "mixture_200g"),
    # Household
    ("Detergent Powder", "Surf Excel", "Household", "1 kg", 235, 720, 2.0, "detergent_1kg"),
    ("Detergent Powder", "Ariel", "Household", "1 kg", 250, 720, 1.2, "detergent_1kg"),
    ("Dishwash Bar", "Vim", "Household", "200 g", 20, 720, 3.0, "dishwash_bar"),
    ("Toilet Cleaner", "Harpic", "Household", "500 ml", 98, 720, 1.0, "toilet_cleaner_500ml"),
    # Personal care
    ("Bath Soap", "Lifebuoy", "Personal Care", "100 g", 35, 1080, 3.5, "soap_100g"),
    ("Bath Soap", "Cinthol", "Personal Care", "100 g", 38, 1080, 2.0, "soap_100g"),
    ("Toothpaste", "Colgate", "Personal Care", "200 g", 110, 720, 3.0, "toothpaste_200g"),
    ("Shampoo", "Clinic Plus", "Personal Care", "180 ml", 125, 720, 1.2, "shampoo_180ml"),
    # Spices
    ("Turmeric Powder", "Eastern", "Spices", "100 g", 42, 300, 2.5, "turmeric_100g"),
    ("Red Chilli Powder", "Aachi", "Spices", "200 g", 85, 300, 2.5, "chilli_powder_200g"),
    ("Sambar Powder", "Aachi", "Spices", "200 g", 90, 300, 3.0, "sambar_powder_200g"),
    ("Mustard Seeds", "Local", "Spices", "100 g", 30, 365, 2.0, "mustard_100g"),
    ("Tamarind", "Local", "Spices", "250 g", 55, 270, 1.5, "tamarind_250g"),
    # Produce
    ("Onion", "Local", "Produce", "1 kg", 45, 21, 6.0, "onion_1kg"),
    ("Tomato", "Local", "Produce", "1 kg", 40, 6, 5.0, "tomato_1kg"),
    ("Potato", "Local", "Produce", "1 kg", 38, 20, 4.0, "potato_1kg"),
    ("Coconut", "Local", "Produce", "1 pc", 30, 10, 4.0, "coconut_1pc"),
]
PRODUCTS = [ProductDef(*row) for row in _RAW]

# category -> (typical margin %, max discount % the bargaining agent may ever give)
CATEGORY_RULES = {
    "Staples": (7.0, 3.0),
    "Beverages": (14.0, 8.0),
    "Dairy": (5.0, 2.0),
    "Bakery": (10.0, 5.0),
    "Snacks": (17.0, 10.0),
    "Household": (15.0, 8.0),
    "Personal Care": (18.0, 10.0),
    "Spices": (14.0, 8.0),
    "Produce": (15.0, 5.0),
}

# name, lead_time_days, categories supplied
SUPPLIERS = [
    ("City Wholesale Traders", 2, ["Staples", "Spices"]),
    ("Southern Dairy Agency", 1, ["Dairy", "Bakery"]),
    ("Metro FMCG Distributors", 3, ["Snacks", "Beverages", "Household", "Personal Care"]),
    ("Fresh Farm Vendors", 1, ["Produce"]),
]

# name, date, ramp_days, peak_multiplier, {category: boost}
FESTIVALS = [
    ("Pongal", date(2026, 1, 14), 6, 1.5, {"Staples": 1.6, "Dairy": 1.4, "Spices": 1.3}),
    ("Tamil New Year", date(2026, 4, 14), 4, 1.3, {"Staples": 1.3, "Snacks": 1.3}),
    ("Vinayagar Chaturthi", date(2026, 9, 14), 4, 1.3, {"Produce": 1.4, "Dairy": 1.2}),
    ("Ayudha Pooja", date(2026, 10, 20), 4, 1.3, {"Produce": 1.5, "Snacks": 1.2}),
    ("Diwali", date(2026, 11, 8), 7, 1.6,
     {"Snacks": 1.8, "Household": 1.5, "Staples": 1.3, "Dairy": 1.3}),
]

STORES = [
    {"name": "Sri Lakshmi Provisions", "base_baskets": 90, "n_customers": 120, "active_share": 1.0},
    {"name": "Green Basket Mart", "base_baskets": 40, "n_customers": 60, "active_share": 0.85},
]

FIRST_NAMES = ["Arun", "Priya", "Karthik", "Divya", "Suresh", "Meena", "Ravi", "Lakshmi", "Vijay",
               "Anitha", "Mohan", "Kavya", "Rahul", "Sneha", "Ganesh", "Deepa", "Imran", "Farah",
               "Sanjay", "Pooja", "Balaji", "Nandhini", "Ajay", "Revathi", "Dinesh", "Swathi"]
LAST_INITIALS = "ABCDEGKMNPRSTV"

# Things customers ask for that the store does not carry (demand-signal fuel).
NOT_IN_CATALOG = ["millet noodles", "ragi biscuits", "peanut butter", "brown rice", "oat milk",
                  "multigrain atta", "green tea", "quinoa", "cold pressed coconut oil",
                  "protein bar", "jaggery powder", "almond milk"]

# Code-mixed phrasing templates: (language, template)
ASK_TEMPLATES = [
    ("en", "do you have {item}?"),
    ("en", "{item} available?"),
    ("en", "need {item}, in stock?"),
    ("ta-en", "anna {item} irukka?"),
    ("ta-en", "{item} kidaikkuma anna?"),
    ("ta-en", "{item} stock irukka?"),
    ("hi-en", "bhaiya {item} hai kya?"),
    ("hi-en", "{item} milega?"),
]