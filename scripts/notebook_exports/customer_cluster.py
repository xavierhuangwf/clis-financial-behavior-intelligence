#!/usr/bin/env python
# coding: utf-8

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# from sklearn.metrics import silhouette_score
# 
# data = pd.read_csv('C:\\Users\\levit\\Downloads\expenses_go_data_with_categories.csv')
# 
# 
# customer_data = data.groupby('Account No').agg({
#     'Amount': ['sum', 'mean', 'count'],
#     'Balance': 'mean',
#     'Expenditure categories': lambda x: x.mode()[0]
# }).reset_index()
# 
# 
# customer_data.columns = ['Account No', 'Total Amount', 'Average Amount', 'Transaction Count', 
#                          'Average Balance', 'Most Frequent Category']
# 
# 
# encoder = OneHotEncoder()
# encoded_categories = encoder.fit_transform(customer_data[['Most Frequent Category']]).toarray() 
# encoded_categories_df = pd.DataFrame(encoded_categories, columns=encoder.get_feature_names_out(['Most Frequent Category']))
# 
# customer_data = pd.concat([customer_data, encoded_categories_df], axis=1)
# 
# features = customer_data[['Total Amount', 'Average Amount', 'Transaction Count', 'Average Balance'] + 
#                          list(encoded_categories_df.columns)]
# 
# 
# scaler = StandardScaler()
# scaled_features = scaler.fit_transform(features)
# 
# # Determining the optimal number of clusters using the elbow method
# inertia = []
# silhouette_scores = []
# K_range = range(2, 11)
# 
# for k in K_range:
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     kmeans.fit(scaled_features)
#     inertia.append(kmeans.inertia_)
#     silhouette_scores.append(silhouette_score(scaled_features, kmeans.labels_))
# 
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.plot(K_range, inertia, marker='o')
# plt.title('Elbow Method for Optimal k')
# plt.xlabel('Number of Clusters (k)')
# plt.ylabel('Inertia')
# 
# plt.subplot(1, 2, 2)
# plt.plot(K_range, silhouette_scores, marker='o')
# plt.title('Silhouette Score for Optimal k')
# plt.xlabel('Number of Clusters (k)')
# plt.ylabel('Silhouette Score')
# plt.show()
# 
# 
# optimal_k = 8
# kmeans = KMeans(n_clusters=optimal_k, random_state=42)
# customer_data['Cluster'] = kmeans.fit_predict(scaled_features)
# 
# 
# pca = PCA(n_components=2)
# pca_result = pca.fit_transform(scaled_features)
# customer_data['PCA1'] = pca_result[:, 0]
# customer_data['PCA2'] = pca_result[:, 1]
# 
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=customer_data, palette='viridis', s=100)
# plt.title('Customer Clusters (2D PCA)')
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.show()
# 
# cluster_summary = customer_data.groupby('Cluster').agg({
#     'Total Amount': 'mean',
#     'Average Amount': 'mean',
#     'Transaction Count': 'mean',
#     'Average Balance': 'mean',
#     'Most Frequent Category': lambda x: x.mode()[0]
# }).reset_index()
# 
# print("Cluster Summary:")
# print(cluster_summary)

# In[ ]:




