from __future__ import annotations

from pathlib import Path

import pandas as pd


SEGMENT_OFFERS = {
    0: [
        {"description": "5%/10% cashback on food (supermarkets, groceries, fast food)", "priority": 0.9},
        {"description": "15% cashback on entertainment (Disney+, Vodafone, etc.)", "priority": 0.7},
        {"description": "2% cashback on all transactions", "priority": 0.5},
    ],
    1: [
        {"description": "15% cashback on entertainment (Disney+, Vodafone, etc.)", "priority": 0.9},
        {"description": "5%/10% cashback on food (supermarkets, groceries, fast food)", "priority": 0.7},
        {"description": "15% cashback on travel bookings", "priority": 0.6},
    ],
    2: [
        {"description": "2% cashback on all transactions", "priority": 0.8},
        {"description": "15% cashback on entertainment (Disney+, Vodafone, etc.)", "priority": 0.6},
        {"description": "15% cashback on travel bookings", "priority": 0.5},
    ],
    3: [
        {"description": "15% cashback on travel bookings", "priority": 0.9},
        {"description": "5%/10% cashback on food (supermarkets, groceries, fast food)", "priority": 0.7},
        {"description": "15% cashback on entertainment (Disney+, Vodafone, etc.)", "priority": 0.6},
    ],
    4: [
        {"description": "5% cashback on retail (ZARA, H&M, etc.)", "priority": 0.9},
        {"description": "15% cashback on entertainment (Disney+, Vodafone, etc.)", "priority": 0.7},
        {"description": "5%/10% cashback on food (supermarkets, groceries, fast food)", "priority": 0.6},
    ],
}


def get_top_offers(segment: int, top_n: int = 3) -> list[str]:
    offers = SEGMENT_OFFERS.get(segment, [])
    sorted_offers = sorted(offers, key=lambda x: x["priority"], reverse=True)
    return [offer["description"] for offer in sorted_offers[:top_n]]


def attach_offer_recommendations(segmented_df: pd.DataFrame) -> pd.DataFrame:
    out = segmented_df.copy()
    out["Recommended Offers"] = out["Customer Segment"].apply(
        lambda seg: " | ".join(get_top_offers(seg))
    )
    return out


def save_offer_recommendations(df: pd.DataFrame, output_csv: str | Path) -> Path:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return output_csv


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    input_csv = project_root / "data" / "processed" / "customer_segments.csv"
    output_csv = project_root / "outputs" / "predictions" / "segmentation_offers.csv"

    segmented_df = pd.read_csv(input_csv)
    recommended_df = attach_offer_recommendations(segmented_df)
    save_offer_recommendations(recommended_df, output_csv)

    print(recommended_df[["Account No", "Customer Segment", "Recommended Offers"]].head())
    print(f"Saved to: {output_csv}")
