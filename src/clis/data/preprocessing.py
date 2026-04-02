from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from typing import Iterable

import pandas as pd
import numpy as np

REQUIRED_COLUMNS = [
    "Date",
    "Timestamp",
    "Account No",
    "Balance",
    "Amount",
    "Third Party Account No",
    "Third Party Name",
]


def _validate_required_columns(
    df: pd.DataFrame,
    required: Iterable[str] = REQUIRED_COLUMNS,
) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def is_valid_date(value: object) -> bool:
    try:
        datetime.strptime(str(value), "%d/%m/%Y")
        return True
    except ValueError:
        return False


def is_valid_timestamp(value: object) -> bool:
    return bool(re.fullmatch(r"\d{2}:\d{2}", str(value)))


def is_valid_account(value: object) -> bool:
    if pd.isna(value):
        return True
    account = str(value).split(".")[0]
    return len(account) == 9 and account.isdigit()


def normalize_account(value: object) -> str | float:
    if pd.isna(value):
        return np.nan
    account = str(value).split(".")[0]
    return account.zfill(9)


def load_transactions(file_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    _validate_required_columns(df)
    return df


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    _validate_required_columns(df)

    # Drop rows missing essential fields
    df = df.dropna(subset=["Date", "Timestamp", "Account No", "Balance", "Amount"])

    # Remove exact duplicate rows
    df = df.drop_duplicates()

    # Keep only rows with valid account/date/time formats
    valid_mask = (
        df["Third Party Account No"].apply(is_valid_account)
        & df["Account No"].apply(is_valid_account)
        & df["Date"].apply(is_valid_date)
        & df["Timestamp"].apply(is_valid_timestamp)
    )
    df = df.loc[valid_mask].copy()

    # Normalize account identifiers
    df["Account No"] = df["Account No"].apply(normalize_account)
    df["Third Party Account No"] = df["Third Party Account No"].apply(normalize_account)

    # Construct unified datetime column
    df["Datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Timestamp"],
        format="%d/%m/%Y %H:%M",
    )

    # Sort chronologically for downstream rolling features
    df = df.sort_values(["Account No", "Datetime"]).reset_index(drop=True)

    return df


def filter_consumer_spending(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep consumer-initiated spending only:
    - negative amounts only
    - no explicit third-party account number
    - merchant name available

    Output Amount is converted to positive spending magnitude.
    """
    df = df.copy()

    spending = df.loc[df["Amount"] < 0].copy()
    spending = spending.loc[spending["Third Party Account No"].isna()].copy()
    spending = spending.dropna(subset=["Third Party Name"]).copy()

    spending["Amount"] = spending["Amount"].abs()
    spending = spending.sort_values(["Account No", "Datetime"]).reset_index(drop=True)

    return spending


def preprocess_from_csv(file_path: str | Path) -> pd.DataFrame:
    raw = load_transactions(file_path)
    cleaned = clean_transactions(raw)
    return cleaned


def preprocess_spending_from_csv(file_path: str | Path) -> pd.DataFrame:
    cleaned = preprocess_from_csv(file_path)
    spending = filter_consumer_spending(cleaned)
    return spending


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    input_csv = project_root / "data" / "raw" / "simulated_fake_transactions_dataset_2.csv"

    cleaned_df = preprocess_from_csv(input_csv)
    spending_df = preprocess_spending_from_csv(input_csv)

    print("Cleaned transactions:")
    print(cleaned_df.head())
    print(cleaned_df.shape)

    print("\nConsumer spending only:")
    print(spending_df.head())
    print(spending_df.shape)