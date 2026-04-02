from __future__ import annotations

import pandas as pd

CATEGORY_MAPPING = {
    "Groceries": ["COOP LOCAL", "SAINSBURY"],
    "Eating": ["DELIVEROO", "JUSTEAT", "HARVESTER", "CHIQUITO", "ASK ITALIAN", "BILL'S"],
    "Shopping": [
        "NEXT", "TOPSHOP", "DOROTHY PERKINS", "MATALAN", "REEBOK", "RIVER ISLAND", "NIKE",
        "ADIDAS", "NEW LOOK", "JD SPORTS", "MILLETS", "NORTH FACE", "UMBRO", "TK MAXX",
        "HOBBY LOBBY", "HOBBYCRAFT", "ETSY", "A LIFE ON CANVAS", "FIVE SENSES ART",
        "CRAFTASTIC", "BRILLIANT BRUSHES", "CASS ART", "STITCH BY STITCH",
        "A YARN STORY", "SPORTS DIRECT", "AMAZON", "MOUNTAIN WAREHOUSE"
    ],
    "Entertainment": [
        "NETFLIX", "BLIZZARD", "XBOX", "MOJANG STUDIOS", "SQUAREONIX", "DISNEY", "GAME",
        "GAMESTATION", "CEX"
    ],
    "Fitness": ["PUREGYM", "GRAND UNION BJJ", "FOOTBALLPITCH"],
    "Finance": ["HALIFAX", "LBG", "PREMIER FINANCE"],
    "Personal Care": ["BOOTS", "LLOYDS PHARMACY", "REMEDY PLUS CARE"],
    "Coffee Shops and Bars": [
        "COFFEE REPUBLIC", "FULL OF BEANS", "AMT COFFEE", "COFFEE #1", "COSTA COFFEE",
        "STARBUCKS", "THE ROYAL OAK", "RED LION", "KINGS ARMS", "ROSE & CROWN",
        "THE CROWN", "WHITE HART"
    ],
    "Bookstore": ["BLACKWELL'S", "WATERSTONES", "FOYLES", "DAUNT BOOKS", "THE WORKS"],
}

CATEGORY_ORDER = [
    "Shopping",
    "Groceries",
    "Entertainment",
    "Eating",
    "Fitness",
    "Bookstore",
    "Finance",
    "Coffee Shops and Bars",
    "Personal Care",
    "General",
]


def classify_spending(name: object) -> str:
    merchant = str(name).upper().strip()

    for category, keywords in CATEGORY_MAPPING.items():
        if any(keyword in merchant for keyword in keywords):
            return category

    return "General"


def add_category_column(
    df: pd.DataFrame,
    merchant_col: str = "Third Party Name",
    output_col: str = "Expenditure categories",
) -> pd.DataFrame:
    out = df.copy()
    out[output_col] = out[merchant_col].apply(classify_spending)
    return out
