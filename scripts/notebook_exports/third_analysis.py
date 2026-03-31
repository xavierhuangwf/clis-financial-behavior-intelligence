#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('C:\\Users\\levit\\Downloads\sorted_data.csv')
data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Timestamp'])

# Transaction frequency analysis

transaction_freq = data.groupby(['Account No', 'Third Party Account No']).size().reset_index(name='Transaction Count')

plt.figure(figsize=(12, 6))
sns.histplot(transaction_freq['Transaction Count'], bins=50, kde=True)
plt.title('Transaction Frequency Distribution')
plt.xlabel('Transaction Count')
plt.ylabel('Frequency')
plt.show()

# Transaction Amount Analysis
plt.figure(figsize=(12, 6))
sns.histplot(data['Amount'], bins=50, kde=True)
plt.title('Transaction Amount Distribution')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.show()

# Transaction time analysis
data['Hour'] = data['Datetime'].dt.hour

plt.figure(figsize=(12, 6))
sns.histplot(data['Hour'], bins=24, kde=True)
plt.title('Transaction Time Distribution')
plt.xlabel('Hour of the Day')
plt.ylabel('Frequency')
plt.show()

# Third-party account analysis: Calculate the number of transactions for each third-party account
third_party_freq = data.groupby('Third Party Account No').size().reset_index(name='Transaction Count')

plt.figure(figsize=(12, 6))
sns.histplot(third_party_freq['Transaction Count'], bins=50, kde=True)
plt.title('Third Party Transaction Frequency Distribution')
plt.xlabel('Transaction Count')
plt.ylabel('Frequency')
plt.show()

# Find third-party accounts that have frequent transactions with multiple customers
third_party_customers = data.groupby(['Third Party Account No', 'Account No']).size().reset_index(
    name='Transaction Count')
third_party_customers = third_party_customers.groupby('Third Party Account No').size().reset_index(
    name='Customer Count')

plt.figure(figsize=(12, 6))
sns.histplot(third_party_customers['Customer Count'], bins=50, kde=True)
plt.title('Third Party Accounts with Multiple Customers')
plt.xlabel('Number of Customers')
plt.ylabel('Frequency')
plt.show()

# Find the third-party accounts with the most transactions
top_third_parties = third_party_freq.sort_values(by='Transaction Count', ascending=False).head(10)
print("Top 10 Third Party Accounts by Transaction Count:")
print(top_third_parties)

# Find the third-party accounts that have transactions with the most customers
top_third_parties_customers = third_party_customers.sort_values(by='Customer Count', ascending=False).head(10)
print("Top 10 Third Party Accounts by Number of Customers:")
print(top_third_parties_customers)


# In[ ]:




