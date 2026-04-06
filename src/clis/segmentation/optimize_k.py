from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from clis.segmentation.cluster_customers import build_feature_matrix
from clis.segmentation.feature_engineering import build_customer_feature_table


def evaluate_k_range(feature_matrix, k_range=range(2, 11)) -> pd.DataFrame:
    scores = defaultdict(list)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(feature_matrix)

        scores["k"].append(k)
        scores["inertia"].append(kmeans.inertia_)
        scores["silhouette"].append(silhouette_score(feature_matrix, labels))
        scores["calinski_harabasz"].append(calinski_harabasz_score(feature_matrix, labels))
        scores["davies_bouldin"].append(davies_bouldin_score(feature_matrix, labels))

    return pd.DataFrame(scores)


def plot_k_optimization(scores_df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(scores_df["k"], scores_df["inertia"], marker="o")
    axes[0].set_title("Elbow Method for Optimal k")
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia")

    axes[1].plot(scores_df["k"], scores_df["silhouette"], marker="o")
    axes[1].set_title("Silhouette Score for Optimal k")
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Silhouette Score")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    input_csv = project_root / "data" / "processed" / "expenses_go_data_with_categories.csv"
    figure_path = project_root / "outputs" / "figures" / "segmentation" / "elbow_silhouette_score.png"
    scores_csv = project_root / "outputs" / "metrics" / "segmentation_k_scores.csv"

    customer_df = build_customer_feature_table(input_csv)
    feature_matrix, _ = build_feature_matrix(customer_df)
    scores_df = evaluate_k_range(feature_matrix)

    scores_csv.parent.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(scores_csv, index=False)
    plot_k_optimization(scores_df, figure_path)

    print(scores_df)
    print(f"Saved scores to: {scores_csv}")
