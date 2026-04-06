from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBClassifier

from clis.churn.feature_label_builder import (
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    build_features_labels_table,
)


def build_splits(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS].copy()
    y = df[LABEL_COLUMN].copy()

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_base_model(y_train: pd.Series) -> XGBClassifier:
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=pos_weight,
        random_state=42,
        n_jobs=-1,
    )
    return model


def tune_model(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    base_model = build_base_model(y_train)

    param_dist = {
        "max_depth": [3, 4, 5, 6],
        "n_estimators": [50, 100, 150, 200],
        "learning_rate": [0.03, 0.05, 0.07, 0.1],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    }

    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=12,
        scoring="f1",
        cv=5,
        verbose=1,
        n_jobs=-1,
        random_state=42,
    )
    random_search.fit(X_train, y_train)

    print("Best Parameters:", random_search.best_params_)
    print("Best CV F1 score:", random_search.best_score_)

    return random_search.best_estimator_


def fit_final_model(
    model: XGBClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> XGBClassifier:
    eval_set = [(X_train, y_train), (X_val, y_val)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    return model


def save_model_with_verification(model: XGBClassifier, model_output_path: str | Path) -> None:
    model_output_path = Path(model_output_path)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)

    model.save_model(model_output_path)

    reloaded = xgb.XGBClassifier()
    reloaded.load_model(model_output_path)

    print(f"Model saved and verified: {model_output_path}")
    print(f"Model size: {model_output_path.stat().st_size / (1024 * 1024):.2f} MB")


def plot_training_vs_validation_loss(model: XGBClassifier, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = model.evals_result()
    epochs = len(results["validation_0"]["logloss"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(epochs), results["validation_0"]["logloss"], label="Train Loss")
    ax.plot(range(epochs), results["validation_1"]["logloss"], label="Validation Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Log Loss")
    ax.set_title("Training vs Validation Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_probability_distribution(y_proba: np.ndarray, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(y_proba, bins=30, alpha=0.8, color="blue", edgecolor="black")
    ax.set_title("Churn Probability Distribution")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_confusion_matrix_figure(
    y_true: pd.Series,
    y_pred: np.ndarray,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("XGBoost Churn Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def evaluate_split(name: str, model: XGBClassifier, X: pd.DataFrame, y: pd.Series) -> dict:
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    print(f"{name} Accuracy: {accuracy:.4f}")
    print(f"{name} F1: {f1:.4f}")

    return {
        "name": name,
        "accuracy": accuracy,
        "f1": f1,
        "report": classification_report(y, y_pred),
        "pred": y_pred,
        "proba": y_proba,
    }


def train_xgboost_pipeline(
    input_csv: str | Path,
    model_output_path: str | Path,
    figure_dir: str | Path,
    metrics_output_path: str | Path,
) -> XGBClassifier:
    df = build_features_labels_table(input_csv)
    X_train, X_val, X_test, y_train, y_val, y_test = build_splits(df)

    best_model = tune_model(X_train, y_train)
    best_model = fit_final_model(best_model, X_train, y_train, X_val, y_val)

    train_metrics = evaluate_split("Train", best_model, X_train, y_train)
    val_metrics = evaluate_split("Validation", best_model, X_val, y_val)
    test_metrics = evaluate_split("Test", best_model, X_test, y_test)

    save_model_with_verification(best_model, model_output_path)

    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    plot_training_vs_validation_loss(
        best_model,
        figure_dir / "training_vs_validation_loss.png",
    )
    plot_probability_distribution(
        test_metrics["proba"],
        figure_dir / "churn_probability_distribution.png",
    )
    plot_confusion_matrix_figure(
        y_test,
        test_metrics["pred"],
        figure_dir / "churn_confusion_matrix.png",
    )

    metrics_output_path = Path(metrics_output_path)
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_output_path, "w", encoding="utf-8") as f:
        f.write("=== Train ===\n")
        f.write(f"Accuracy: {train_metrics['accuracy']:.6f}\n")
        f.write(f"F1: {train_metrics['f1']:.6f}\n")
        f.write(train_metrics["report"])
        f.write("\n\n=== Validation ===\n")
        f.write(f"Accuracy: {val_metrics['accuracy']:.6f}\n")
        f.write(f"F1: {val_metrics['f1']:.6f}\n")
        f.write(val_metrics["report"])
        f.write("\n\n=== Test ===\n")
        f.write(f"Accuracy: {test_metrics['accuracy']:.6f}\n")
        f.write(f"F1: {test_metrics['f1']:.6f}\n")
        f.write(test_metrics["report"])

    return best_model


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]

    input_csv = project_root / "data" / "raw" / "simulated_fake_transactions_dataset_2.csv"
    model_output_path = project_root / "outputs" / "models" / "xgboost_churn_model.json"
    figure_dir = project_root / "outputs" / "figures" / "churn"
    metrics_output_path = project_root / "outputs" / "metrics" / "churn_report.txt"

    train_xgboost_pipeline(
        input_csv=input_csv,
        model_output_path=model_output_path,
        figure_dir=figure_dir,
        metrics_output_path=metrics_output_path,
    )
