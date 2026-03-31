import pandas as pd

# read CSV file
file_path = './dataset/simulated_fake_transactions_dataset_2.csv'
df = pd.read_csv(file_path)

# print rows and columns
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

# Remove null values in 'Account No', 'Balance', 'Amount' columns
df_cleaned = df.dropna(subset=['Account No', 'Balance', 'Amount'])

Num_1stCleaned = df.shape[0] - df_cleaned.shape[0]
print("1st filter out of %s rows" % (Num_1stCleaned))

df_cleaned = df.dropna(subset=['Date', 'Timestamp', 'Account No', 'Balance', 'Amount'])

Num_2rdCleaned = df.shape[0] - df_cleaned.shape[0] - Num_1stCleaned
print("2nd Filter out of %s rows " % (Num_2rdCleaned))

# convert 'Date' column to datetime type
df_cleaned['Date'] = pd.to_datetime(df_cleaned['Date'], format='%d/%m/%Y')

# convert 'Timestamp' column to datetime type
df_cleaned['Timestamp'] = pd.to_datetime(df_cleaned['Timestamp'], format='%H:%M').dt.time

# sort by 'Date' and 'Timestamp' columns
df_sorted = df_cleaned.sort_values(by=['Date', 'Timestamp'])

print("Number of rows:", df_sorted.shape[0])

df_sorted.to_csv("./dataset/sorted_data.csv", index=False)

expenses_data = df_sorted[df_sorted['Amount'] < 0]
expenses_data.to_csv("./dataset/expenses_data.csv", index=False)

expenses_go_data = expenses_data.dropna(subset=['Third Party Name'])
expenses_go_data.to_csv("./dataset/expenses_go_data.csv", index=False)

sorted_thirdparty_count = expenses_go_data['Third Party Name'].value_counts().sort_values(ascending=False)

top_10 = sorted_thirdparty_count.head(10)

import matplotlib.pyplot as plt

top_10.plot(kind='bar')
plt.title('Expense Frequency Top 10')
plt.xlabel('Third Party Name')
plt.ylabel('Frequency')
plt.show()


def classify_spending(name):
    if name in ['Coop Local', 'Sainsbury']:
        return 'Groceries'
    elif name in ['Deliveroo', 'JustEat', 'Harvester', 'Chiquito', 'ASK Italian', "Bill's"]:
        return 'Eating'
    elif name in ['Next', 'Topshop', 'Dorothy Perkins', 'Matalan', 'Reebok', 'River Island', 'Nike', 'Adidas',
                  'New Look', 'JD Sports', 'Millets', 'North Face', 'Umbro', 'TK Maxx', 'Hobby Lobby', 'Hobbycraft',
                  'Etsy', 'A Life on Canvas', 'Five Senses Art', 'Craftastic', 'Brilliant Brushes', 'Cass Art',
                  'Stitch By Stitch', 'A Yarn Story', 'Sports Direct', 'Amazon', 'AMAZON', 'Mountain Warehouse']:
        return 'Shopping'
    elif name in ['Netflix', 'Blizzard', 'Xbox', 'Mojang Studios', 'SquareOnix', 'Disney', 'Game', 'Gamestation',
                  'CeX']:
        return 'Entertainment'
    elif name in ['PureGym', 'Grand Union BJJ', 'FootballPitch']:
        return 'Fitness'
    elif name in ['Halifax', 'LBG', 'Premier Finance']:
        return 'Finances'
    elif name in ['Boots', 'Lloyds Pharmacy', 'Remedy plus care']:
        return 'Personal care'
    elif name in ['Coffee Republic', 'Full of Beans', 'AMT Coffee', 'Coffee #1', 'Costa Coffee', 'Starbucks',
                  'The Royal Oak', 'Red Lion', 'Kings Arms', 'Rose & Crown', 'The Crown', 'White Hart']:
        return 'Coffee shops and bars'
    elif name in ["Blackwell's", 'Waterstones', 'Foyles', 'Daunt Books', 'The Works']:
        return 'Bookstore'
    return 'General'


expenses_go_data['Expenditure categories'] = expenses_go_data['Third Party Name'].apply(classify_spending)

print(expenses_go_data)

expenses_go_data.to_csv("./dataset/expenses_go_data_with_categories.csv", index=False)
