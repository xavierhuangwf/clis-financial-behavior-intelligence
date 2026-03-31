import labelPreparation as lp
import dataPreparation as dp
import pandas as pd
import xgboost as xgb
import numpy as np


class XGBoostModel:
    def __init__(self, model_path, threshold=0.5):
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        self.model = model
        self.threshold = threshold

    @staticmethod
    def preprocess(data: pd.DataFrame) -> xgb.DMatrix:
        # assume that all the features are processed with columns
        # total_spending_3m, avg_spending_per_m, std_spending_per_m,
        # total_visits_3m, avg_visits_per_m, recency, balance
        return xgb.DMatrix(data)

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        dmatrix = self.preprocess(data)
        return self.model.predict_proba(dmatrix)[:, -1]

    def predict_label(self, data: pd.DataFrame) -> np.ndarray:
        probas = self.predict_proba(data)
        return (probas >= self.threshold).astype(int)

    def set_threshold(self, threshold: float):
        self.threshold = threshold

    def predict_batch(self, batch_data: pd.DataFrame) -> pd.DataFrame:
        probabilities = self.predict_proba(batch_data)
        labels = self.predict_label(batch_data)
        return pd.DataFrame({"probability": probabilities, "label": labels})
