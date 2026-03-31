import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from collections import defaultdict
import matplotlib.pyplot as plt


def preprocess_text(data, column):
    data[column] = data[column].astype(str).str.strip().str.lower()
    return data


def engineer_features(data):
    data['Transaction Frequency'] = data.groupby('Account No')['Date'].transform('count')
    data['Average Transaction Amount'] = data.groupby('Account No')['Amount'].transform('mean').abs()
    data['Balance Change'] = data['Balance'].diff().fillna(0)

    data['Transaction Hour'] = pd.to_datetime(data['Timestamp']).dt.hour
    transaction_hours = data.groupby('Account No')['Transaction Hour'].agg(
        lambda x: x.value_counts().index[0]).reset_index()
    transaction_hours.rename(columns={'Transaction Hour': 'Peak Transaction Hour'}, inplace=True)
    data = pd.merge(data, transaction_hours, on='Account No', how='left')

    if 'Expenditure categories' in data.columns:
        favorite_category = data.groupby('Account No')['Expenditure categories'].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else 'Other').reset_index()
        favorite_category.rename(columns={'Expenditure categories': 'Favorite Category'}, inplace=True)
        data = pd.merge(data, favorite_category, on='Account No', how='left')
    else:
        data['Favorite Category'] = 'Other'

    return data


def optimize_clustering(features):
    scores = defaultdict(list)
    inertia = []
    silhouette_scores = []
    K_range = range(2, 11)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(features)
        labels = kmeans.labels_

        scores['k'].append(k)
        scores['inertia'].append(kmeans.inertia_)
        scores['silhouette'].append(silhouette_score(features, labels))
        scores['calinski_harabasz'].append(calinski_harabasz_score(features, labels))
        scores['davies_bouldin'].append(davies_bouldin_score(features, labels))

        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(features, kmeans.labels_))
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(K_range, inertia, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')

    plt.subplot(1, 2, 2)
    plt.plot(K_range, silhouette_scores, marker='o')
    plt.title('Silhouette Score for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.show()

    return pd.DataFrame(scores)


def main():
    data = pd.read_csv('./dataset/expenses_go_data_with_categories.csv')

    data = preprocess_text(data, 'Third Party Name')

    data = engineer_features(data)

    data.to_csv("./dataset/Afterengineer_features.csv", index=False)

    agg_data = data.groupby('Account No').agg({
        'Transaction Frequency': 'mean',
        'Average Transaction Amount': 'mean',
        'Balance Change': 'mean',
        'Balance': 'last',
        'Third Party Name': lambda x: ' '.join(x.dropna().astype(str)),
        'Peak Transaction Hour': 'first',
        'Favorite Category': 'first'
    }).reset_index()

    tfidf = TfidfVectorizer(max_features=100)
    third_party_features = tfidf.fit_transform(agg_data['Third Party Name'])

    numerical_features = agg_data[
        ['Transaction Frequency', 'Average Transaction Amount', 'Balance', 'Peak Transaction Hour']]
    scaler = StandardScaler()
    scaled_numerical = scaler.fit_transform(numerical_features)

    combined_features = np.hstack((scaled_numerical, third_party_features.toarray()))

    clustering_scores = optimize_clustering(combined_features)
    print(clustering_scores)

    optimal_k = 5
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    agg_data['Customer Segment'] = kmeans.fit_predict(combined_features)

    offers = {
        0: [
            {"description": "5%/10% cashback on food(supermarkets, groceries, fast food)", "priority": 0.9},
            {"description": "15% cashback on entertainment(Disney+, Vodafone ..)", "priority": 0.7},
            {"description": "2% cashback on all transactions", "priority": 0.5}
        ],
        1: [
            {"description": "15% cashback on entertainment(Disney+, Vodafone ..)", "priority": 0.9},
            {"description": "5%/10% cashback on food(supermarkets, groceries, fast food)", "priority": 0.7},
            {"description": "15% cashback on travel bookings", "priority": 0.6}
        ],
        2: [
            {"description": "2% cashback on all transactions", "priority": 0.8},
            {"description": "15% cashback on entertainment(Disney+, Vodafone ..)", "priority": 0.6},
            {"description": "15% cashback on travel bookings", "priority": 0.5}
        ],
        3: [
            {"description": "15% cashback on travel bookings", "priority": 0.9},
            {"description": "5%/10% cashback on food(supermarkets, groceries, fast food)", "priority": 0.7},
            {"description": "15% cashback on entertainment(Disney+, Vodafone ..)", "priority": 0.6}
        ],
        4: [
            {"description": "5% cashback on retail(ZARA, H&M, ..)", "priority": 0.9},
            {"description": "15% cashback on entertainment(Disney+, Vodafone ..)", "priority": 0.7},
            {"description": "5%/10% cashback on food(supermarkets, groceries, fast food)", "priority": 0.6}
        ]
    }

    sample_account = agg_data['Account No'].iloc[100]
    customer_segment = agg_data.loc[agg_data['Account No'] == sample_account, 'Customer Segment'].values[0]
    segment_offers = offers.get(customer_segment, [])

    sorted_offers = sorted(segment_offers, key=lambda x: x['priority'], reverse=True)
    recommendations = [f"{i + 1}. {offer['description']}" for i, offer in enumerate(sorted_offers[:3])]

    print(f"Account: {sample_account}")
    print("Recommended offers:")
    print("\n".join(recommendations))


if __name__ == "__main__":
    main()
