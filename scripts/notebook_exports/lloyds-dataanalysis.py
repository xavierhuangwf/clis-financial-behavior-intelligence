#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns

file_path = "/Users/zhuangyujie/Downloads/simulated_fake_transactions_dataset_2.csv"

try:
    df = pd.read_csv(file_path)
    print(df.head())
except FileNotFoundError:
    print(f"File not found: {file_path}")


# In[2]:


print(df.shape)
print(df.info())


# In[3]:


# % of NA data
print(df.isnull().mean())


# % of NA data of Data Timestamp account No balance amount is less than 1%, so delete the NA row.
# 
# 

# In[4]:


columns_to_check = ["Date", "Timestamp", "Account No", "Balance", "Amount"]
df.dropna(subset=columns_to_check, inplace=True)


# In[5]:


# % of NA data
print(df.isnull().mean())


# In[6]:


df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')


# 

# ## 1. Analyze daily spending in the time series.
# 

# ### amount>0

# In[7]:


import matplotlib.pyplot as plt
import pandas as pd

# group by date, calculate total amount
daily_total_amount = df[df['Amount'] > 0].groupby('Date')['Amount'].sum()

plt.figure(figsize=(20, 6))
plt.plot(daily_total_amount.index, daily_total_amount.values, marker='o', linestyle='-', color='b')

plt.title('Total Transaction Amount Per Day')
plt.xlabel('Date')
plt.ylabel('Total Transaction Amount')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ### amount<0

# In[8]:


# group by date, calculate total amount
daily_total_amount = df[df['Amount'] < 0].groupby('Date')['Amount'].sum()

plt.figure(figsize=(20, 6))
plt.plot(daily_total_amount.index, daily_total_amount.values, marker='o', linestyle='-', color='b')

plt.title('Total Transaction Amount Per Day')
plt.xlabel('Date')
plt.ylabel('Total Transaction Amount')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ## Insight1:
# ### On the first day of each month, both positive and negative amounts are particularly high, which could be due to salary payments or bill payments like credit card dues.

# ## 2. Observe the distribution of amounts across different types of stores and customer groups.

# ## Check corr between amount and balance

# In[9]:


# only use positive Amount
df_positive_amount = df[df['Amount'] > 0]

# group by account no, calcualte total amount and avg of balance
account_summary = df_positive_amount.groupby('Account No').agg(
    total_amount=('Amount', 'sum'),
    average_balance=('Balance', 'mean')
)

plt.figure(figsize=(10, 6))
plt.scatter(account_summary['total_amount'], account_summary['average_balance'], color='b', alpha=0.6)

plt.title('Total Amount vs Average Balance by Account(amount>0)')
plt.xlabel('Total Transaction Amount(cost)')
plt.ylabel('Average Balance')
plt.tight_layout()
plt.show()


# In[10]:


correlation = account_summary['total_amount'].corr(account_summary['average_balance'])
print(f"Pearson Correlation Coefficient between Total Amount and Average Balance: {correlation}")


# → amount and balance is highly correlated

# ## Use k-means to classify the scatter plot into three clusters

# In[11]:


from sklearn.cluster import KMeans

# only choose amount>0
df_positive_amount = df[df['Amount'] > 0]

# group by accounts calculate total spending amount and average balance
account_summary = df_positive_amount.groupby('Account No').agg(
    total_amount=('Amount', 'sum'),
    average_balance=('Balance', 'mean')
)

# K-Means, 3 groups
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
account_summary['Cluster'] = kmeans.fit_predict(account_summary)

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=account_summary['total_amount'],
    y=account_summary['average_balance'],
    hue=account_summary['Cluster'],
    palette="viridis",
    alpha=0.6
)
plt.title('Customer Clustering Based on Total Amount and Average Balance (Amount > 0)')
plt.xlabel('Total Transaction Amount (Cost)')
plt.ylabel('Average Balance')
plt.legend(title='Cluster')
plt.tight_layout()
plt.show()


# ## Classify stores and plot the spending patterns of each customer group across different store categories

# In[12]:


# define catergories
def classify_spending(name):
    if name in ['Coop Local', 'Sainsbury']:
        return 'Groceries'

    elif name in ['Zara', 'Next', 'Topshop', 'Dorothy Perkins', 'Matalan', 'River Island',
                  'New Look', 'Millets', 'North Face', 'Umbro', 'TK Maxx', 'Barbiee Boutique']:
        return 'Clothing'

    elif name in ['Amazon', 'Etsy']:
        return 'e-commerce'

    elif name in ['Hobby Lobby']:
        return 'retailer'

    elif name in ['Deliveroo', 'JustEat', 'Harvester', 'Chiquito', 'ASK Italian', "Bill's"]:
        return 'Food'

    elif name in ['Coffee Republic', 'Full of Beans', 'AMT Coffee', 'Coffee #1', 'Costa Coffee', 'Starbucks',
                  'The Royal Oak', 'Red Lion', 'Kings Arms', 'Rose & Crown', 'The Crown', 'White Hart']:
        return 'Coffee shops and bars'

    elif name in ['Netflix', 'Blizzard', 'Xbox', 'Mojang Studios', 'SquareOnix', 'Disney', 'Game', 'Gamestation',
                  'CeX', 'Amazon', 'Green Park', 'Victoria Park', 'A Life on Canvas', 'Five Senses Art', 'Craftastic',
                  'Brilliant Brushes', 'Cass Art', 'Stitch By Stitch',
                  'A Yarn Story', 'Hobbycraft']:
        return 'Entertainment'

    elif name in ['PureGym', 'Grand Union BJJ', 'FootballPitch', 'Mountain Warehouse', 'Adidas', 'Nike', 'Reebok',
                  'JD Sports']:
        return 'Fitness or Sporting Goods'

    elif name in ['Halifax', 'LBG', 'Premier Finance', 'Howlader and Co Chartered Accountants', 'CPA']:
        return 'Finances'

    elif name in ['Boots', 'Lloyds Pharmacy', 'Remedy plus care', 'Westport Care Home', 'Kew House', 'Specsavers',
                  'University College Hospital']:
        return 'Healthcare/ Personal care'

    elif name in ["Blackwell's", 'Waterstones', 'Foyles', 'Daunt Books', 'The Works']:
        return 'Bookstore'

    elif name in ['Town High', 'Lavender Primary', 'Green Park Academy']:
        return 'Education'

    elif name in ['Pets at Home', 'Jollyes', 'Pets Corner', 'Happy Days Home']:
        return 'Pet / Home'

    return 'Others'


# In[13]:


# Step 1: apply() classify_spending to df
df["Store Category"] = df["Third Party Name"].apply(classify_spending)

# Step 2: add clustering results in df
account_summary = account_summary.rename(columns={"Cluster": "Cluster_summary"})
df = df.merge(account_summary[["Cluster_summary"]], on="Account No", how="left")

# Step 3: filter positive and negative amount
df_positive = df[df["Amount"] > 0]
df_negative = df[df["Amount"] < 0]

# Step 4: calculate different groups in different categories of total amount
category_spending_positive = df_positive.groupby(["Cluster_summary", "Store Category"])["Amount"].sum().reset_index()
category_spending_negative = df_negative.groupby(["Cluster_summary", "Store Category"])["Amount"].sum().reset_index()

# Step 5: plot（Amount > 0）
plt.figure(figsize=(12, 6))
sns.barplot(
    data=category_spending_positive,
    x="Store Category",
    y="Amount",
    hue="Cluster_summary",
    palette="viridis"
)
plt.xticks(rotation=45, ha="right")
plt.title("Spending Patterns for Positive Amounts by Customer Clusters & Merchant Categories")
plt.ylabel("Total Spending Amount")
plt.xlabel("Store Category")
plt.legend(title="Customer Cluster")
plt.tight_layout()
plt.show()

# Step 6: plot（Amount < 0）
plt.figure(figsize=(12, 6))
sns.barplot(
    data=category_spending_negative,
    x="Store Category",
    y="Amount",
    hue="Cluster_summary",
    palette="viridis"
)
plt.xticks(rotation=45, ha="right")
plt.title("Spending Patterns for Negative Amounts by Customer Clusters & Merchant Categories")
plt.ylabel("Total Spending Amount")
plt.xlabel("Store Category")
plt.legend(title="Customer Cluster")
plt.tight_layout()
plt.show()


# ## Insight 2
# 
# ### 1. Customer Groups:
# - **Cluster 0 (Dark Purple):** High spending and savings, likely wealthy customers.
# - **Cluster 1 (Blue-Green):** Low spending and savings, likely just starting to save or spend less.
# - **Cluster 2 (Yellow):** High spending, average savings, likely middle-range spenders.
# 
# ### 2. Spending in Specific Categories:
# - Cluster 2 is the biggest negative amount in **"Finance,"** suggesting they may have more investments.
# - Cluster 2 also spends a lot in **"Clothing"** and **"Coffee shops and bars,"** showing they enjoy entertainment and leisure.
# - Cluster 1 has moderate spending, suggesting more conservative spending habits.
# - In the **"Clothing"** and **"Finance"** categories, there are unusually high incomes, which could indicate returns or other unaccounted revenue sources.

# ## 3. Attempt to define potential financially vulnerable customer types, and visualize the proportion of each type of vulnerable customer and their main spending merchants.

# ### Potentially Vulnerable Customer Types:
# 
# 1. **Low Average Balance**  
#    The average account balance is consistently below a certain threshold (e.g., the lower 25th percentile of overall customer balances).  
#    Frequent occurrences of balances close to zero or negative suggest potential financial strain.
# 
# 2. **Frequent Negative Transactions**  
#    A high proportion of transactions with negative amounts may indicate:  
#    - Frequent overdrafts or negative spending (e.g., credit card repayments, fines, interest payments).  
#    - Over-reliance on borrowing (e.g., credit card advances or short-term loans).
# 
# 3. **High Balance Volatility**  
#    Significant monthly fluctuations in balance may suggest unstable financial management (e.g., a large paycheck coming in but quickly spent).  
#    This may indicate the customer lacks stable savings and is vulnerable to sudden economic changes.
# 
# 4. **Spending Exceeds Balance**  
#    If total spending exceeds twice the average balance (e.g., spending more than double the average balance),  
#    it suggests that spending habits may be exceeding financial capabilities, potentially relying on external funds (credit cards, loans).
# 

# In[14]:


# Define thresholds
low_balance_threshold = account_summary['average_balance'].quantile(0.25)
high_negative_ratio = 0.4
high_volatility_threshold = df.groupby('Account No')['Balance'].std().quantile(0.75)

# Calculate negative transaction ratio
negative_amount_ratio = df[df['Amount'] < 0].groupby('Account No')['Amount'].count() / df.groupby('Account No')[
    'Amount'].count()

# Calculate spending exceeding balance threshold
spending_exceeds_threshold = account_summary['total_amount'] > (account_summary['average_balance'] * 2)

# Add new features to account_summary
account_summary['negative_amount_ratio'] = negative_amount_ratio
account_summary['balance_volatility'] = df.groupby('Account No')['Balance'].std()

# Flag vulnerable customers
account_summary['vulnerable_customer'] = (
        (account_summary['average_balance'] < low_balance_threshold) |
        (account_summary['negative_amount_ratio'] > high_negative_ratio) |
        (account_summary['balance_volatility'] > high_volatility_threshold) |
        spending_exceeds_threshold
)

# Filter vulnerable customers
vulnerable_customers = account_summary[account_summary['vulnerable_customer']]
print(vulnerable_customers)


# In[15]:


# ratio of each catergories of vulnerable customer
vulnerability_factors = {
    "Low Balance": account_summary['average_balance'] < low_balance_threshold,
    "High Negative Transactions": account_summary['negative_amount_ratio'] > high_negative_ratio,
    "High Balance Volatility": account_summary['balance_volatility'] > high_volatility_threshold,
    "Spending Exceeds Balance": spending_exceeds_threshold
}
vulnerability_counts = {k: v.sum() for k, v in vulnerability_factors.items()}


# In[16]:


# plot
plt.figure(figsize=(8, 6))
sns.barplot(x=list(vulnerability_counts.keys()), y=list(vulnerability_counts.values()), palette="coolwarm")
plt.xticks(rotation=20)
plt.title("Distribution of Financially Vulnerable Customers")
plt.xlabel("Vulnerability Type")
plt.ylabel("Number of Customers")
plt.show()


# In[17]:


 #main stores of vulnerable customers
vulnerable_transactions = df[df['Account No'].isin(vulnerable_customers.index)]
top_merchant_counts = vulnerable_transactions['Third Party Name'].value_counts().head(10)

plt.figure(figsize=(8, 6))
sns.barplot(y=top_merchant_counts.index, x=top_merchant_counts.values, palette="coolwarm")
plt.title("Top 10 Merchants for Vulnerable Customers")
plt.xlabel("Transaction Count")
plt.ylabel("Merchant")
plt.show()


# ## Insight3
# ### First Chart:Main Vulnerable Types:
# - **High Negative Transactions** and **Spending Exceeds Balance** are the most common types, with over 1,000 customers in each category.
# 
# 
# ### Second Chart: Top 10 Merchants for Vulnerable Customers
# 
# - **Coop Local** stands out significantly compared to other merchants, indicating that vulnerable customers tend to shop at local supermarkets, likely for convenience or lower prices.
# - **Next** and **Umbro** also rank highly. These brands likely offer mid- to low-priced clothing and sports goods, meeting the needs of vulnerable customers.
# 
