#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


# % of NA data of Data Timestamp account No balance amount is less than 1%, so delete the NA row.
# 
# 

# In[11]:


file_path = "/Users/zhuangyujie/Downloads/simulated_fake_transactions_dataset_2.csv"

try:
    df = pd.read_csv(file_path)
    print(df.head())
except FileNotFoundError:
    print(f"File not found: {file_path}")


# In[12]:


# % of NA data
print(df.isnull().mean())


# % of NA data of Data Timestamp account No balance amount is less than 1%, so delete the NA row.
# 
# 

# In[13]:


columns_to_check = ["Date", "Timestamp", "Account No", "Balance", "Amount", "Third Party Name"]
df.dropna(subset=columns_to_check, inplace=True)

# % of NA data
print(df.isnull().mean())


# In[14]:


#Creating a time feature
df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y")
df["Timestamp"] = pd.to_datetime(df['Timestamp'], format='%H:%M')
df["Hour"] = df["Timestamp"].dt.hour
df["DayOfWeek"] = df["Date"].dt.dayofweek
df["Month"] = df["Date"].dt.month


# In[15]:


# Third party Label Encoding(turn categorial variable into numerical variable)
merchant_encoder = LabelEncoder()
df["MerchantID"] = merchant_encoder.fit_transform(df["Third Party Name"])


# In[16]:


# filter merchant less than 3 times
#merchant_counts = df["MerchantID"].value_counts()
#df_filtered = df[df["MerchantID"].isin(merchant_counts[merchant_counts > 3].index)]


# In[17]:


from mlxtend.frequent_patterns import apriori, association_rules


# In[18]:


#Group by Account No and MerchantID, and see whether the customer has shopped at this store
basket = df.groupby(['Account No', 'MerchantID']).size().unstack().fillna(0)
basket = basket.applymap(lambda x: 1 if x > 0 else 0)


# In[19]:


basket = basket.astype(bool)


# In[20]:


basket


# In[21]:


# Apriori find Frequent itemsets
min_support2 = 0.5  # min support threshold
#Support: The frequency of a condition or result occurring in all transactions.
frequent_itemsets = apriori(basket, min_support=min_support2, use_colnames=True)
print("frequent itemsets:")
frequent_itemsets = frequent_itemsets.sort_values(by="support", ascending=False)  #sort by support
print(frequent_itemsets)


# In[22]:


# Association Rules
#Confidence: The probability that the consequent occurs given that the antecedent has occurred.
min_confidence = 0.8  #set min cconfidence as 0.85
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)


# In[23]:


#decode merchants
def decode_merchants(frozenset_merchants):
    return frozenset(merchant_encoder.inverse_transform(list(frozenset_merchants)))


rules["antecedents"] = rules["antecedents"].apply(decode_merchants)
rules["consequents"] = rules["consequents"].apply(decode_merchants)


# In[24]:


#Lift: The strength of the association between the antecedent and consequent, indicating how much more likely they are to occur together than independently.
#if lift>1 means positive correlated, lift=1 means no correlation, lift<1 means negative correlation
#I want antecedents and consequents are highly correlated, so set lift between 1.2 and 1.5
rules = rules[(rules['lift'] < 1.5) & (rules['lift'] > 1.2)]


# In[25]:


print(rules[['antecedents', 'consequents', 'confidence']]
      .to_string(index=False))


# Arrange the correspondence between antecedents and consequents so that each antecedent appears only once, and list all its corresponding consequents.

# In[26]:


import pandas as pd

# rules is a copy
rules = rules.copy()

# antecedents and consequents turn into list（avoid frozen set）
rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x))
rules['consequents'] = rules['consequents'].apply(lambda x: list(x))

exploded_rules = rules.explode('antecedents')

# group by antecedents and combine consequents and delete duplicate values
antecedent_table = (
    exploded_rules
    .groupby('antecedents')['consequents']
    .apply(lambda x: sorted(set(sum(x, []))))
    .reset_index()
)

antecedent_table['consequents'] = antecedent_table['consequents'].apply(lambda x: ', '.join(x))

print(antecedent_table.to_string(index=False))


# In[27]:


rules_dict = dict(zip(
    antecedent_table['antecedents'],
    antecedent_table['consequents'].apply(lambda x: x.split(', '))
))


# In[28]:


import json

with open("rules_dict.json", "w") as f:
    json.dump(rules_dict, f)

