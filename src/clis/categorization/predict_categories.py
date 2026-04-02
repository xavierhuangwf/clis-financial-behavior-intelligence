from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from clis.data.preprocessing import preprocess_from_csv, filter_consumer_spending
from clis.categorization.train_random_forest import add_temporal_features, FEATURE_COLUMNS


def predict_transaction_categories(
    input_csv: str | Path,
    model_path: str | Path,
) -> pd.DataFrame:
    cleaned = preprocess_from_csv(input_csv)
    spending = filter_consumer_spending(cleaned)
    featured = add_temporal_features(spending)

    model = joblib.load(model_path)
    featured["Predicted category"] = model.predict(featured[FEATURE_COLUMNS])

    return featured


def save_predictions(df: pd.DataFrame, output_csv: str | Path) -> Path:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return output_csv


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]

    input_csv = project_root / "data" / "raw" / "simulated_fake_transactions_dataset_2.csv"
    model_path = project_root / "outputs" / "models" / "random_forest_categorizer.joblib"
    output_csv = project_root / "outputs" / "predictions" / "transactions_with_predicted_categories.csv"

    predicted_df = predict_transaction_categories(input_csv, model_path)
    save_predictions(predicted_df, output_csv)

    print(predicted_df.head())
    print(f"Saved predictions to: {output_csv}")
