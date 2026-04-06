from __future__ import annotations

from pathlib import Path

import pandas as pd


NUMERIC_SEGMENT_COLUMNS = [
    "Transaction Frequency",
    "Average Transaction Amount",
    "Balance",
    "Peak Transaction Hour",
]


def load_categorized_transactions(input_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    return df


def preprocess_text_column(df: pd.DataFrame, column: str = "Third Party Name") -> pd.DataFrame:
    out = df.copy()
    out[column] = out[column].astype(str).str.strip().str.lower()
    return out


def engineer_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["Transaction Frequency"] = out.groupby("Account No")["Date"].transform("count")
    out["Average Transaction Amount"] = out.groupby("Account No")["Amount"].transform("mean").abs()
    out["Balance Change"] = out.groupby("Account No")["Balance"].diff().fillna(0)

    out["Transaction Hour"] = pd.to_datetime(out["Timestamp"], format="%H:%M", errors="coerce").dt.hour
    peak_hour = (
        out.groupby("Account No")["Transaction Hour"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else -1)
        .reset_index()
        .rename(columns={"Transaction Hour": "Peak Transaction Hour"})
    )
    out = out.merge(peak_hour, on="Account No", how="left")

    if "Expenditure categories" in out.columns:
        favorite_category = (
            out.groupby("Account No")["Expenditure categories"]
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "General")
            .reset_index()
            .rename(columns={"Expenditure categories": "Favorite Category"})
        )
        out = out.merge(favorite_category, on="Account No", how="left")
    else:
        out["Favorite Category"] = "General"

    return out


def build_customer_feature_table(input_csv: str | Path) -> pd.DataFrame:
    df = load_categorized_transactions(input_csv)
    df = preprocess_text_column(df, "Third Party Name")
    df = engineer_transaction_features(df)

    agg_data = (
        df.groupby("Account No")
        .agg(
            Transaction_Frequency=("Transaction Frequency", "mean"),
            Average_Transaction_Amount=("Average Transaction Amount", "mean"),
            Balance=("Balance", "last"),
            Peak_Transaction_Hour=("Peak Transaction Hour", "first"),
            Favorite_Category=("Favorite Category", "first"),
            Merchant_Text=("Third Party Name", lambda x: " ".join(x.dropna().astype(str))),
        )
        .reset_index()
    )

    agg_data = agg_data.rename(
        columns={
            "Transaction_Frequency": "Transaction Frequency",
            "Average_Transaction_Amount": "Average Transaction Amount",
            "Peak_Transaction_Hour": "Peak Transaction Hour",
            "Favorite_Category": "Favorite Category",
        }
    )

    return agg_data


def save_customer_feature_table(df: pd.DataFrame, output_csv: str | Path) -> Path:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return output_csv


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    input_csv = project_root / "data" / "processed" / "expenses_go_data_with_categories.csv"
    output_csv = project_root / "data" / "processed" / "segmentation_customer_features.csv"

    feature_df = build_customer_feature_table(input_csv)
    save_customer_feature_table(feature_df, output_csv)

    print(feature_df.head())
    print(f"Saved to: {output_csv}")
