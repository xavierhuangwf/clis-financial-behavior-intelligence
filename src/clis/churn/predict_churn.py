from __future__ import annotations

from pathlib import Path

import pandas as pd

from clis.churn.feature_label_builder import (
    FEATURE_COLUMNS,
    ID_COLUMNS,
    LABEL_COLUMN,
    build_features_labels_table,
)
from clis.churn.model_wrapper import XGBoostChurnModel


def predict_churn_from_csv(
    input_csv: str | Path,
    model_path: str | Path,
    threshold: float = 0.5,
) -> pd.DataFrame:
    features_labels = build_features_labels_table(input_csv)

    model = XGBoostChurnModel(model_path=model_path, threshold=threshold)
    prediction_df = model.predict_batch(features_labels[FEATURE_COLUMNS])

    result = features_labels[ID_COLUMNS].copy()
    result["probability"] = prediction_df["probability"]
    result["predicted_churn_label"] = prediction_df["churn_label"]

    if LABEL_COLUMN in features_labels.columns:
        result["actual_churn_label"] = features_labels[LABEL_COLUMN]

    return result


def save_predictions(df: pd.DataFrame, output_csv: str | Path) -> Path:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return output_csv


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]

    input_csv = project_root / "data" / "raw" / "simulated_fake_transactions_dataset_2.csv"
    model_path = project_root / "outputs" / "models" / "xgboost_churn_model.json"
    output_csv = project_root / "outputs" / "predictions" / "churn_prediction.csv"

    predicted_df = predict_churn_from_csv(
        input_csv=input_csv,
        model_path=model_path,
        threshold=0.5,
    )
    save_predictions(predicted_df, output_csv)

    print(predicted_df.head())
    print(f"Saved predictions to: {output_csv}")
