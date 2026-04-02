from __future__ import annotations

from pathlib import Path

import pandas as pd

from clis.data.preprocessing import preprocess_from_csv, filter_consumer_spending
from clis.categorization.category_mapping import add_category_column


def build_categorized_transactions(input_csv: str | Path) -> pd.DataFrame:
    cleaned = preprocess_from_csv(input_csv)
    spending = filter_consumer_spending(cleaned)
    categorized = add_category_column(
        spending,
        merchant_col="Third Party Name",
        output_col="Expenditure categories",
    )
    return categorized


def save_categorized_transactions(
    df: pd.DataFrame,
    output_csv: str | Path,
) -> Path:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return output_csv


def prepare_training_dataset(
    input_csv: str | Path,
    output_csv: str | Path,
) -> pd.DataFrame:
    categorized = build_categorized_transactions(input_csv)
    save_categorized_transactions(categorized, output_csv)
    return categorized


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    input_csv = project_root / "data" / "raw" / "simulated_fake_transactions_dataset_2.csv"
    output_csv = project_root / "data" / "processed" / "expenses_go_data_with_categories.csv"

    categorized_df = prepare_training_dataset(input_csv, output_csv)

    print(categorized_df.head())
    print(categorized_df["Expenditure categories"].value_counts())
    print(f"Saved to: {output_csv}")
