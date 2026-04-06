from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from clis.segmentation.feature_engineering import (
    NUMERIC_SEGMENT_COLUMNS,
    build_customer_feature_table,
)


def build_feature_matrix(customer_df: pd.DataFrame):
    tfidf = TfidfVectorizer(max_features=100)
    merchant_text_features = tfidf.fit_transform(customer_df["Merchant_Text"].fillna(""))

    numeric_features = customer_df[NUMERIC_SEGMENT_COLUMNS].copy()
    scaler = StandardScaler()
    scaled_numeric = scaler.fit_transform(numeric_features)

    combined_features = np.hstack((scaled_numeric, merchant_text_features.toarray()))

    artifacts = {
        "scaler": scaler,
        "tfidf": tfidf,
    }
    return combined_features, artifacts


def fit_customer_segments(feature_matrix, n_clusters: int = 5):
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(feature_matrix)
    return model, labels


def attach_segments(customer_df: pd.DataFrame, labels) -> pd.DataFrame:
    out = customer_df.copy()
    out["Customer Segment"] = labels
    return out


def save_segment_table(df: pd.DataFrame, output_csv: str | Path) -> Path:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return output_csv


def plot_customer_segments(feature_matrix, labels, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(feature_matrix)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=labels, alpha=0.7)
    ax.set_title("Customer Clusters (2D PCA)")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def run_segmentation_pipeline(
    input_csv: str | Path,
    output_csv: str | Path,
    figure_path: str | Path,
    n_clusters: int = 5,
) -> pd.DataFrame:
    customer_df = build_customer_feature_table(input_csv)
    feature_matrix, _ = build_feature_matrix(customer_df)
    model, labels = fit_customer_segments(feature_matrix, n_clusters=n_clusters)
    segmented_df = attach_segments(customer_df, labels)

    save_segment_table(segmented_df, output_csv)
    plot_customer_segments(feature_matrix, labels, figure_path)

    return segmented_df


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    input_csv = project_root / "data" / "processed" / "expenses_go_data_with_categories.csv"
    output_csv = project_root / "data" / "processed" / "customer_segments.csv"
    figure_path = project_root / "outputs" / "figures" / "segmentation" / "customer_clusters.png"

    segmented_df = run_segmentation_pipeline(
        input_csv=input_csv,
        output_csv=output_csv,
        figure_path=figure_path,
        n_clusters=5,
    )

    print(segmented_df.head())
    print(segmented_df["Customer Segment"].value_counts())
    print(f"Saved to: {output_csv}")
