#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:\\Users\\levit\\Downloads\expenses_go_data_with_categories.csv')


# In[2]:


# basic analysis
print(df['Amount'].describe())
print(df['Expenditure categories'].value_counts())


# In[13]:


# Visualization of expenditure distribution
plt.figure(figsize=(10, 6))
sns.histplot(-df['Amount'], bins=50, kde=True)
plt.title('Distribution of Expenditure Amount')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.show()


# In[4]:


# Visualization of expenditure category share
category_summary = df['Expenditure categories'].value_counts()
plt.figure(figsize=(10, 6))
category_summary.plot(kind='pie', autopct='%1.1f%%')
plt.title('Expenditure Categories Distribution')
plt.show()


# In[56]:


df['Absolute_Amount'] = df['Amount'].abs()

category_amounts = df.groupby('Expenditure categories')['Absolute_Amount'].sum().reset_index()

total_amount = category_amounts['Absolute_Amount'].sum()

category_amounts['Percentage'] = (category_amounts['Absolute_Amount'] / total_amount) * 100

print("Percentage of spending in each category:")
print(category_amounts[['Expenditure categories', 'Percentage']])

plt.figure(figsize=(10, 6))
plt.pie(
    category_amounts['Absolute_Amount'],
    labels=category_amounts['Expenditure categories'],
    autopct='%1.1f%%',
    # startangle=140,
    wedgeprops=dict(width=0.3),
    pctdistance=0.85,
    labeldistance=1.1,
    textprops={'fontsize': 9}
)
plt.title('Percentage of spending in each category')

plt.show()


# In[6]:


# Simple time series analysis
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)


# In[7]:


monthly_spending = -df.resample('ME')['Amount'].sum()
plt.figure(figsize=(10, 6))
monthly_spending.plot()
plt.title('Monthly Expenditure Trend')
plt.xlabel('Date')
plt.ylabel('Total Amount')
plt.show()


# Why is there so little spending in December? Because there are only nine days of data in December.

# In[8]:


df = pd.read_csv('C:\\Users\\levit\\Downloads\expenses_go_data_with_categories.csv')

df['YearMonth'] = pd.to_datetime(df['Date']).dt.to_period('M')

monthly_spending = -df.groupby('YearMonth')['Amount'].sum()

df['Date'] = pd.to_datetime(df['Date'])
days_in_month = df.groupby('YearMonth')['Date'].nunique()

daily_avg_spending = monthly_spending / days_in_month

plt.figure(figsize=(12, 6))
daily_avg_spending.plot(kind='line', marker='o')
plt.title('Daily Average Spending by Month')
plt.xlabel('Month')
plt.ylabel('Average Daily Spending')
plt.grid()
plt.show()


# In[9]:


df['Hour'] = pd.to_datetime(df['Timestamp'], format='%H:%M:%S').dt.hour

# Hourly statistics of spending amount and number of transactions
hourly_spending = -df.groupby('Hour')['Amount'].sum()
hourly_transactions = df.groupby('Hour').size()

print(hourly_transactions)


# In[10]:


plt.figure(figsize=(12, 6))
plt.plot(hourly_spending.index, hourly_spending.values, marker='o', label='Total Spending')
plt.title('Hourly Spending and Transaction Trends')
plt.xlabel('Hour of the Day')
plt.ylabel('Amount')
plt.legend()
plt.grid()
plt.show()


# In[11]:


plt.figure(figsize=(12, 6))
plt.plot(hourly_transactions.index, hourly_transactions.values, marker='o', label='Transaction Count')
plt.title('Hourly Spending and Transaction Trends')
plt.xlabel('Hour of the Day')
plt.ylabel('Count')
plt.legend()
plt.grid()
plt.show()


# In[12]:


df = pd.read_csv('C:\\Users\\levit\\Downloads\expenses_go_data_with_categories.csv')
df['Hour'] = pd.to_datetime(df['Timestamp'], format='%H:%M:%S').dt.hour

hourly_category_spending = -df.groupby(['Hour', 'Expenditure categories'])['Amount'].sum().unstack()
hourly_category_spending.plot(kind='bar', stacked=True, figsize=(14, 8))
plt.title('Hourly Spending by Category')
plt.xlabel('Hour of the Day')
plt.ylabel('Total Amount')
plt.show()


# In[ ]:




