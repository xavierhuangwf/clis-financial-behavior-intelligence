#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from collections import defaultdict
import matplotlib.pyplot as plt


def preprocess_text(data, column):
    data[column] = data[column].astype(str).str.strip().str.lower().str.replace(" ", "")
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


data = pd.read_csv('C:\\Users\\levit\\Downloads\expenses_go_data_with_categories.csv', parse_dates=["Date"])

# The start time of the data, only three months of data are used each time
start_month = 3

data_selected = data[(data['Date'].dt.month >= start_month) & (data['Date'].dt.month < start_month + 3)]

print(data_selected.head(5))

data = preprocess_text(data, 'Third Party Name')

data = engineer_features(data)

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

optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
agg_data['Customer Segment'] = kmeans.fit_predict(combined_features)




# In[30]:


cluster_summary = agg_data.groupby('Customer Segment').agg({
    'Transaction Frequency': 'mean',
    'Average Transaction Amount': 'mean',
    'Balance': 'mean',
    'Peak Transaction Hour': 'mean',
    'Favorite Category': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Other'
}).reset_index()
print(cluster_summary)



# In[31]:


import seaborn as sns

# Transaction frequency distribution
plt.figure(figsize=(10, 6))
sns.barplot(x='Customer Segment', y='Transaction Frequency', data=agg_data)
plt.title('Average Transaction Frequency by Cluster')
plt.show()

# Average transaction amount distribution
plt.figure(figsize=(10, 6))
sns.barplot(x='Customer Segment', y='Average Transaction Amount', data=agg_data)
plt.title('Average Transaction Amount by Cluster')
plt.show()



# In[32]:


# different segments' favorite categories
favorite_categories = agg_data.groupby('Customer Segment')['Favorite Category'].agg(lambda x: x.mode().iloc[0])
print(favorite_categories)


# In[33]:


# balance change
balance_change = agg_data.groupby('Customer Segment')['Balance Change'].agg(['mean', 'std'])
print(balance_change)


# In[34]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=100)
third_party_features = tfidf.fit_transform(agg_data['Third Party Name'])
feature_names = tfidf.get_feature_names_out()

# the most frequent third-party names for each cluster
# print the top 10 and bottom 10 third-party names for each cluster
for cluster in range(4):
    cluster_data = agg_data[agg_data['Customer Segment'] == cluster]
    tfidf_matrix = tfidf.transform(cluster_data['Third Party Name'])
    tfidf_sum = tfidf_matrix.sum(axis=0)
    tfidf_scores = [(feature_names[i], tfidf_sum[0, i]) for i in range(len(feature_names))]
    tfidf_scores.sort(key=lambda x: x[1], reverse=True)
    print(f"Cluster {cluster}:")
    print(tfidf_scores[:10])
    print(tfidf_scores[-10:])


# In[35]:


# Design cashback offers for each cluster based on the analysis
offers = {
    0: [
        {"description": "15% cashback on shopping categories (e.g., amazon, next, tkmaxx)", "priority": 0.9},
        {"description": "10% cashback on food delivery services (e.g., justeat, deliveroo)", "priority": 0.8},
        {"description": "7% cashback on groceries (e.g., sainsbury)", "priority": 0.7},
        {"description": "20% cashback on entertainment subscriptions (e.g., Disney+, Vodafone)", "priority": 0.6},
        {"description": "2.5% cashback on all transactions", "priority": 0.5}
    ],
    1: [
        {"description": "20% cashback on entertainment subscriptions (e.g., netflix, disney)", "priority": 0.9},
        {"description": "15% cashback on fitness memberships (e.g., puregym, grandunionbjj)", "priority": 0.8},
        {"description": "10% cashback on shopping categories (e.g., next, amazon)", "priority": 0.7},
        {"description": "20% cashback on travel bookings", "priority": 0.6},
        {"description": "2.5% cashback on all transactions", "priority": 0.5}
    ],
    2: [
        {"description": "6% cashback on banking services (e.g., halifax, lbg)", "priority": 0.8},
        {"description": "13% cashback on fitness memberships (e.g., puregym, grandunionbjj)", "priority": 0.7},
        {"description": "10% cashback on groceries (e.g., sainsbury)", "priority": 0.6},
        {"description": "20% cashback on entertainment subscriptions (e.g., netflix, disney)", "priority": 0.5},
        {"description": "2.5% cashback on all transactions", "priority": 0.4}
    ],
    3: [
        {"description": "15% cashback on shopping categories (e.g., next, topshop, hobbylobby)", "priority": 0.9},
        {"description": "10% cashback on groceries (e.g., sainsbury)", "priority": 0.8},
        {"description": "20% cashback on entertainment subscriptions (e.g., Disney+, Vodafone)", "priority": 0.7},
        {"description": "10% cashback on food delivery services (e.g., justeat, deliveroo)", "priority": 0.6},
        {"description": "2.5% cashback on all transactions", "priority": 0.5}
    ]
}


# In[36]:


# combine churn prediction
df = pd.read_csv("churn_prediction.csv", parse_dates=["start_month", "end_month"])
df_select = df[(df['start_month'].dt.month == start_month) & (df['churn_label'] == 1)]
print(df_select.head(5))


# In[37]:


# Recommend 4 offers to a customer in each cluster, it's just an example

sample_account = df_select['Account No'].iloc[3]

customer_segment = agg_data.loc[agg_data['Account No'] == sample_account, 'Customer Segment'].values[0]

segment_offers = offers.get(customer_segment, [])

sorted_offers = sorted(segment_offers, key=lambda x: x['priority'], reverse=True)
recommendations = [f"{i + 1}. {offer['description']}" for i, offer in enumerate(sorted_offers[:4])]

print(f"Account: {sample_account}")
print("Recommended offers:")
print("\n".join(recommendations))

