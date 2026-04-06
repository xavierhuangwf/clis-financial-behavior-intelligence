from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from clis.data.preprocessing import preprocess_from_csv, filter_consumer_spending


WINDOW_SIZE = 3

ID_COLUMNS = ["Account No", "start_month", "end_month"]
FEATURE_COLUMNS = [
    "total_spending_3m",
    "avg_spending_per_m",
    "std_spending_per_m",
    "total_visits_3m",
    "avg_visits_per_m",
    "recency",
]
LABEL_COLUMN = "churn_label"


def load_consumer_spending(input_csv: str | Path) -> pd.DataFrame:
    cleaned = preprocess_from_csv(input_csv)
    spending = filter_consumer_spending(cleaned).copy()
    spending["YearMonth"] = spending["Datetime"].dt.to_period("M")
    return spending


def build_monthly_features(spending_df: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        spending_df.groupby(["Account No", "YearMonth"])
        .agg(
            total_amount=("Amount", "sum"),
            frequency=("Amount", "count"),
        )
        .reset_index()
    )
    return monthly


def fill_missing_months(monthly_features: pd.DataFrame) -> pd.DataFrame:
    if monthly_features.empty:
        return monthly_features.copy()

    all_months = pd.period_range(
        monthly_features["YearMonth"].min(),
        monthly_features["YearMonth"].max(),
        freq="M",
    )
    all_users = monthly_features["Account No"].unique()

    full_index = pd.MultiIndex.from_product(
        [all_users, all_months],
        names=["Account No", "YearMonth"],
    )

    filled = (
        monthly_features.set_index(["Account No", "YearMonth"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )
    return filled


def extract_rolling_features(
    monthly_data: pd.DataFrame,
    original_spending_df: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
) -> pd.DataFrame:
    rows = []

    original_spending_df = original_spending_df.copy()
    original_spending_df["YearMonth"] = original_spending_df["Datetime"].dt.to_period("M")

    for account_no, user_data in monthly_data.groupby("Account No"):
        user_data = user_data.sort_values("YearMonth").reset_index(drop=True)
        user_history = (
            original_spending_df.loc[original_spending_df["Account No"] == account_no]
            .sort_values("Datetime")
            .copy()
        )

        for i in range(len(user_data) - window_size + 1):
            window = user_data.iloc[i : i + window_size]
            target_month = window.iloc[-1]["YearMonth"]

            relevant_history = user_history.loc[user_history["YearMonth"] <= target_month]
            last_purchase_date = relevant_history["Datetime"].max()

            if pd.isna(last_purchase_date):
                recency = 999
            else:
                target_month_end = target_month.end_time.date()
                recency = (target_month_end - last_purchase_date.date()).days

            row = {
                "Account No": account_no,
                "start_month": str(window.iloc[0]["YearMonth"]),
                "end_month": str(window.iloc[-1]["YearMonth"]),
                "total_spending_3m": float(window["total_amount"].sum()),
                "avg_spending_per_m": float(window["total_amount"].mean()),
                "std_spending_per_m": float(window["total_amount"].std(ddof=0)),
                "total_visits_3m": float(window["frequency"].sum()),
                "avg_visits_per_m": float(window["frequency"].mean()),
                "recency": int(recency),
            }
            rows.append(row)

    features = pd.DataFrame(rows)
    if not features.empty:
        features["std_spending_per_m"] = features["std_spending_per_m"].fillna(0.0)

    return features


def filter_stable_users(features_df: pd.DataFrame) -> pd.DataFrame:
    stable = features_df.loc[
        (features_df["total_visits_3m"] >= 3) & (features_df["recency"] <= 60)
    ].copy()
    return stable.reset_index(drop=True)


def build_overall_window_baselines(features_df: pd.DataFrame) -> pd.DataFrame:
    baselines = (
        features_df.groupby(["start_month", "end_month"])
        .agg(
            overall_avg_total_amount=("total_spending_3m", "mean"),
            overall_avg_total_visits=("total_visits_3m", "mean"),
        )
        .reset_index()
    )
    return baselines


def set_churn_label(user_row: pd.Series, overall_row: pd.Series) -> int:
    low_engagement = (
        (user_row["total_spending_3m"] < overall_row["overall_avg_total_amount"] * 0.7)
        and (user_row["total_visits_3m"] < overall_row["overall_avg_total_visits"] * 0.7)
    )

    inactivity_with_decline = (
        (user_row["recency"] > 30)
        and (user_row["total_spending_3m"] < overall_row["overall_avg_total_amount"] * 0.8)
        and (user_row["total_visits_3m"] < overall_row["overall_avg_total_visits"] * 0.8)
    )

    return int(low_engagement or inactivity_with_decline)


def assign_churn_labels(features_df: pd.DataFrame) -> pd.DataFrame:
    baselines = build_overall_window_baselines(features_df)

    merged = features_df.merge(
        baselines,
        on=["start_month", "end_month"],
        how="left",
    ).copy()

    merged[LABEL_COLUMN] = merged.apply(
        lambda row: set_churn_label(row, row),
        axis=1,
    )

    return merged[ID_COLUMNS + FEATURE_COLUMNS + [LABEL_COLUMN]].copy()


def build_features_labels_table(input_csv: str | Path) -> pd.DataFrame:
    spending = load_consumer_spending(input_csv)
    monthly = build_monthly_features(spending)
    monthly_filled = fill_missing_months(monthly)
    rolling = extract_rolling_features(monthly_filled, spending, window_size=WINDOW_SIZE)
    stable = filter_stable_users(rolling)
    labeled = assign_churn_labels(stable)
    return labeled


def save_features_labels_table(df: pd.DataFrame, output_csv: str | Path) -> Path:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return output_csv


def _plot_distribution(
    df: pd.DataFrame,
    feature_name: str,
    title: str,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    churned = df.loc[df[LABEL_COLUMN] == 1, feature_name]
    non_churned = df.loc[df[LABEL_COLUMN] == 0, feature_name]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(churned, bins=30, alpha=0.6, label="Churn", color="red")
    ax.hist(non_churned, bins=30, alpha=0.6, label="Non-Churn", color="blue")
    ax.set_title(title)
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_diagnostic_figures(df: pd.DataFrame, figure_dir: str | Path) -> None:
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    _plot_distribution(
        df,
        "total_spending_3m",
        "Churn vs. Non-Churn Monetary Distribution",
        figure_dir / "churn_vs_nonchurn_monetary_distribution.png",
    )
    _plot_distribution(
        df,
        "recency",
        "Churn vs. Non-Churn Recency Distribution",
        figure_dir / "churn_vs_nonchurn_recency_distribution.png",
    )
    _plot_distribution(
        df,
        "total_visits_3m",
        "Churn vs. Non-Churn Frequency Distribution",
        figure_dir / "churn_vs_nonchurn_frequency_distribution.png",
    )


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    input_csv = project_root / "data" / "raw" / "simulated_fake_transactions_dataset_2.csv"
    output_csv = project_root / "data" / "processed" / "churn_features_labels.csv"
    figure_dir = project_root / "outputs" / "figures" / "churn"

    features_labels_df = build_features_labels_table(input_csv)
    save_features_labels_table(features_labels_df, output_csv)
    save_diagnostic_figures(features_labels_df, figure_dir)

    print(features_labels_df.head())
    print(features_labels_df[LABEL_COLUMN].value_counts(normalize=True))
    print(f"Saved features/labels to: {output_csv}")
