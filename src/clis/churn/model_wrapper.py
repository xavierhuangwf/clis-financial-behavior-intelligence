from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from clis.churn.feature_label_builder import FEATURE_COLUMNS


class XGBoostChurnModel:
    def __init__(self, model_path: str | Path, threshold: float = 0.5):
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        self.threshold = threshold

    @staticmethod
    def validate_features(data: pd.DataFrame) -> pd.DataFrame:
        missing = [col for col in FEATURE_COLUMNS if col not in data.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")
        return data[FEATURE_COLUMNS].copy()

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        X = self.validate_features(data)
        return self.model.predict_proba(X)[:, 1]

    def predict_label(self, data: pd.DataFrame) -> np.ndarray:
        probas = self.predict_proba(data)
        return (probas >= self.threshold).astype(int)

    def set_threshold(self, threshold: float) -> None:
        self.threshold = threshold

    def predict_batch(self, batch_data: pd.DataFrame) -> pd.DataFrame:
        probabilities = self.predict_proba(batch_data)
        labels = self.predict_label(batch_data)
        return pd.DataFrame(
            {
                "probability": probabilities,
                "churn_label": labels,
            }
        )
