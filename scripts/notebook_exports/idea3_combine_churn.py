#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import LabelEncoder


# In[2]:


# read churn data
churn_df = pd.read_csv("/Users/zhuangyujie/Desktop/dsmp-2024-groupm15/churn_prediction.csv")
churn_df['end_month'] = pd.to_datetime(churn_df['end_month'], format="%Y-%m")
churn_df['Month'] = churn_df['end_month'].dt.month
churn_df['Year'] = churn_df['end_month'].dt.year

# read transaction data
tran_df = pd.read_csv("/Users/zhuangyujie/Downloads/simulated_fake_transactions_dataset_2.csv")
columns_to_check = ["Date", "Timestamp", "Account No", "Balance", "Amount", "Third Party Name"]
tran_df.dropna(subset=columns_to_check, inplace=True)

# Third party Label Encoding(turn categorial variable into numerical variable)
merchant_encoder = LabelEncoder()
tran_df["MerchantID"] = merchant_encoder.fit_transform(tran_df["Third Party Name"])

tran_df['Date'] = pd.to_datetime(tran_df['Date'], dayfirst=True)
tran_df['Date'] = pd.to_datetime(tran_df['Date'], format="%d/%m/%Y")
tran_df['Month'] = tran_df['Date'].dt.month
tran_df['Year'] = tran_df['Date'].dt.year


# In[3]:


# create merchant column，use Third Party Name if it exist，else use Third Party Account No
tran_df['merchant'] = tran_df['Third Party Name'].combine_first(tran_df['Third Party Account No'])

tran_df = tran_df[['Year', 'Month', 'Account No', 'MerchantID']]
churn_df = churn_df[['Year', 'Month', 'Account No', 'churn_label']]


# In[4]:


tran_df


# In[5]:


churn_df


# In[6]:


#  filter churn = 0
churn = churn_df[churn_df['churn_label'] == 0].copy()

# merge tran_df and churn
tran_churn = tran_df.merge(
    churn,
    on=['Account No', 'Year', 'Month']
)


# In[7]:


tran_churn


# In[8]:


#filter top 1 customer visit store
top1_tx = (
    tran_churn.groupby(['Account No', 'Year', 'Month', 'MerchantID'])
    .size().reset_index(name='count')
    .sort_values(['Account No', 'Year', 'Month', 'count'], ascending=[True, True, True, False])
    .groupby(['Account No', 'Year', 'Month'])
    .head(1)
    .groupby(['Account No', 'Year', 'Month'])['MerchantID']
    .apply(list)
    .reset_index()
)


# In[9]:


top1_tx


# In[10]:


import json

with open("rules_dict.json", "r") as f:
    rules_dict = json.load(f)


# In[11]:


#decode merchants
def decode_merchants(frozenset_merchants):
    return frozenset(merchant_encoder.inverse_transform(list(frozenset_merchants)))


top1_tx["MerchantID"] = top1_tx["MerchantID"].apply(decode_merchants)


# In[12]:


top1_tx


# In[ ]:


#apply association rule mining to top1_tx MerchantID
def get_recommendations(merchants):
    recommended = set()
    for m in merchants:
        if m in rules_dict:
            recommended.update(rules_dict[m])
    return list(recommended) if recommended else []


top1_tx['RecommendedStores'] = top1_tx['MerchantID'].apply(get_recommendations)


# In[16]:


for _, row in top1_tx[:10].iterrows():
    print(f"{row['Account No']} | {row['Month']} | {', '.join(row['RecommendedStores'])}")


# In[15]:


#save as .txt file
with open("accounts_recommendations.txt", "w", encoding="utf-8") as f:
    for _, row in top1_tx.iterrows():
        line = f"{row['Account No']} | {row['Year']} | {row['Month']} | {', '.join(map(str, row['RecommendedStores']))}\n"
        f.write(line)

