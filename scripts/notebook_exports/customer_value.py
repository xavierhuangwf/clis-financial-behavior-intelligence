#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:\\Users\\levit\\Downloads\expenses_go_data_with_categories.csv')

df['Date'] = pd.to_datetime(df['Date'])
current_date = df['Date'].max()

rfm = df.groupby('Account No').agg(
    Recency=('Date', lambda x: (current_date - x.max()).days),
    Frequency=('Amount', 'count'),
    Monetary=('Amount', 'sum')
).reset_index()

balance_stability = df.groupby('Account No').agg(
    Balance_STD=('Balance', np.std),
    Balance_CV=('Balance', lambda x: (x.std() / x.mean()) if x.mean() != 0 else 0)
).reset_index()

rfm = pd.merge(rfm, balance_stability, on='Account No', how='left')


def calculate_network_diversity(subset):
    partner_count = subset['Third Party Name'].nunique()
    if partner_count <= 1:
        return 0.0
    p = subset['Third Party Name'].value_counts(normalize=True)
    return -np.sum(p * np.log(p))


network_features = []
for acc in df['Account No'].unique():
    subset = df[df['Account No'] == acc]
    network_features.append({
        'Account No': acc,
        'Partner_Count': subset['Third Party Name'].nunique(),
        'Network_Diversity': calculate_network_diversity(subset)
    })

network_df = pd.DataFrame(network_features)
rfm = pd.merge(rfm, network_df, on='Account No', how='left')

category_dummies = df['Expenditure categories'].str.get_dummies(sep=';')
category_features = pd.concat([df['Account No'], category_dummies], axis=1)
category_agg = category_features.groupby('Account No').sum().reset_index()


def calculate_category_diversity(row):
    total = row.sum()
    if total == 0:
        return 0.0
    p = row / total
    return min(-np.sum(p * np.log(p)), 1.0)


category_agg['Category_Diversity'] = category_agg.iloc[:, 1:].apply(calculate_category_diversity, axis=1)
rfm = pd.merge(rfm, category_agg, on='Account No', how='left')

scaler = MinMaxScaler()

weights = {
    'Recency': 0.15,
    'Frequency': 0.15,
    'Monetary': 0.20,
    'Balance_CV': 0.05,
    'Partner_Count': 0.15,
    'Network_Diversity': 0.15,
    'Category_Diversity': 0.15
}

score_direction = {
    'Recency': 1,
    'Frequency': 1,
    'Monetary': 1,
    'Balance_CV': -1,
    'Partner_Count': 1,
    'Network_Diversity': 1,
    'Category_Diversity': 1
}

features = list(weights.keys())
rfm_features = rfm[features].copy()
rfm_features[features] = scaler.fit_transform(rfm_features[features])

for col, direction in score_direction.items():
    rfm_features[col] = rfm_features[col] * direction

rfm['Composite_Score'] = rfm_features.multiply(pd.Series(weights)).sum(axis=1)


def value_segment(score):
    if score >= 0.6:
        return 'VIP'
    elif score >= 0.55:
        return 'High Value'
    elif score >= 0.48:
        return 'Medium Value'
    else:
        return 'Low Value'


rfm['Segment'] = rfm['Composite_Score'].apply(value_segment)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.countplot(x='Segment', data=rfm,
              order=['VIP', 'High Value', 'Medium Value', 'Low Value'],
              palette=['red', 'orange', 'blue', 'green'])
plt.title('Customer Value Distribution (Optimized)')

plt.subplot(1, 2, 2)
corr_matrix = rfm[features + ['Composite_Score']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
plt.title('Feature Correlation Matrix')

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
sns.histplot(rfm['Composite_Score'], bins=20, kde=True, color='purple')
plt.title('Optimized Composite Score Distribution')
plt.xlabel('Composite Score')
plt.ylabel('Frequency')
plt.show()

print("Customer value distribution statistics：")
print(rfm['Segment'].value_counts(normalize=True).mul(100).round(2).astype(str) + '%')

print("\nVIP Customer Example：")
print(rfm[rfm['Segment'] == 'VIP'].head())

print("\nHigh Value Customer Exampl：")
print(rfm[rfm['Segment'] == 'High Value'].head())

