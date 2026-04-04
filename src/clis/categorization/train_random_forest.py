from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from clis.categorization.prepare_training_data import build_categorized_transactions


TARGET_COLUMN = "Expenditure categories"
FEATURE_COLUMNS = [
    "Third Party Name",
    "Amount",
    "Balance",
    "Month",
    "DayOfWeek",
    "Hour",
]


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "Datetime" not in out.columns:
        raise ValueError("Expected column 'Datetime' to exist before feature engineering.")

    out["Month"] = out["Datetime"].dt.month
    out["DayOfWeek"] = out["Datetime"].dt.dayofweek
    out["Hour"] = out["Datetime"].dt.hour

    return out


def build_training_table(input_csv: str | Path) -> pd.DataFrame:
    categorized = build_categorized_transactions(input_csv)
    featured = add_temporal_features(categorized)
    return featured


def build_model() -> Pipeline:
    categorical_features = ["Third Party Name", "Month", "DayOfWeek", "Hour"]
    numeric_features = ["Amount", "Balance"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    return pipeline


def plot_confusion_matrix_figure(
    y_true: pd.Series,
    y_pred: np.ndarray,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=sorted(pd.unique(y_true)))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=sorted(pd.unique(y_true)),
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(cmap="Blues", xticks_rotation=45, ax=ax, colorbar=False)
    ax.set_title("Random Forest Transaction Categorization - Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_learning_curve_figure(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        cv=5,
        scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 5),
        n_jobs=-1,
        shuffle=True,
        random_state=42,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_sizes, train_mean, marker="o", color="red", label="Training score")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color="red")

    ax.plot(train_sizes, val_mean, marker="o", color="green", label="Cross-validation score")
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color="green")

    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    ax.set_title("Learning Curves (Random Forest)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_model_with_verification(model: Pipeline, model_output_path: str | Path) -> None:
    model_output_path = Path(model_output_path)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)

    # compress to reduce file size and write more robustly
    joblib.dump(model, model_output_path, compress=3)

    if not model_output_path.exists() or model_output_path.stat().st_size == 0:
        raise RuntimeError(f"Model file was not written correctly: {model_output_path}")

    # immediate reload check
    reloaded_model = joblib.load(model_output_path)
    if reloaded_model is None:
        raise RuntimeError("Model reload verification failed.")

    print(f"Model saved and verified: {model_output_path}")
    print(f"Model size: {model_output_path.stat().st_size / (1024 * 1024):.2f} MB")


def train_random_forest_pipeline(
    input_csv: str | Path,
    model_output_path: str | Path,
    figure_dir: str | Path,
    metrics_output_path: str | Path,
) -> Pipeline:
    df = build_training_table(input_csv)

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(report)

    save_model_with_verification(model, model_output_path)

    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix_figure(
        y_test,
        y_pred,
        figure_dir / "categorization_confusion_matrix.png",
    )

    plot_learning_curve_figure(
        model,
        X,
        y,
        figure_dir / "categorization_learning_curve.png",
    )

    metrics_output_path = Path(metrics_output_path)
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_output_path, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {accuracy:.6f}\n\n")
        f.write(report)

    return model


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]

    input_csv = project_root / "data" / "raw" / "simulated_fake_transactions_dataset_2.csv"
    model_output_path = project_root / "outputs" / "models" / "random_forest_categorizer.joblib"
    figure_dir = project_root / "outputs" / "figures" / "categorization"
    metrics_output_path = project_root / "outputs" / "metrics" / "categorization_report.txt"

    train_random_forest_pipeline(
        input_csv=input_csv,
        model_output_path=model_output_path,
        figure_dir=figure_dir,
        metrics_output_path=metrics_output_path,
    )