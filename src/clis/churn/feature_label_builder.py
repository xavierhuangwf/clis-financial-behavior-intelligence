from datetime import datetime

import dataPreparation as dp
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# define a threshold, return the churn label
def set_thresholds(user_3m_row, overall_3m_row):
    threshold = ((user_3m_row['total_spending_3m'] < overall_3m_row['overall_avg_total_amount'] * 0.7) &
                 (user_3m_row['total_visits_3m'] < overall_3m_row['overall_avg_total_visits'] * 0.7)) | \
                (user_3m_row['recency'] > 30 &
                 (user_3m_row['total_spending_3m'] < overall_3m_row['overall_avg_total_amount'] * 0.8) &
                 (user_3m_row['total_visits_3m'] < overall_3m_row['overall_avg_total_visits'] * 0.8))
    return threshold.iloc[0]


# fill in 0 for months with no consumptions for each user
def fill_missing_months(monthly_features):
    start_month = '2025-1'
    end_month = '2025-12'
    time_format = "%Y-%m"
    all_months = pd.period_range(datetime.strptime(start_month, time_format),
                                 datetime.strptime(end_month, time_format), freq='M')

    all_users = monthly_features['Account No'].unique()

    full_index = pd.MultiIndex.from_product([all_users, all_months],
                                            names=['Account No', 'YearMonth'])

    monthly_features = monthly_features.set_index(['Account No', 'YearMonth'])
    monthly_features = monthly_features.reindex(full_index, fill_value=0).reset_index()

    return monthly_features


def extract_3m_data(monthly_data, window_size, original_df):
    train_data = []

    for account_no, user_data in monthly_data.groupby('Account No'):
        user_data = user_data.sort_values(by='YearMonth')
        full_user_data = original_df[original_df['Account No'] == account_no].sort_values(by='Datetime')

        for i in range(12 - window_size + 1):
            feature_window = user_data.iloc[i:i + window_size]
            target_month = feature_window.iloc[-1]['YearMonth']
            # recency
            last_purchase_date = full_user_data[full_user_data['Datetime'].dt.to_period('M') <= target_month][
                'Datetime'].max()

            if pd.isna(last_purchase_date):  # no consumption history
                recency = 999
            else:
                recency = (pd.Period(target_month, freq='M').end_time.date() - last_purchase_date.date()).days

            features = {
                'Account No': account_no,
                'start_month': feature_window.iloc[0]['YearMonth'],
                'end_month': feature_window.iloc[-1]['YearMonth'],
                'total_spending_3m': feature_window['total_amount'].sum(),
                'avg_spending_per_m': feature_window['total_amount'].mean(),
                'std_spending_per_m': feature_window['total_amount'].std(),
                'total_visits_3m': feature_window['frequency'].sum(),
                'avg_visits_per_m': feature_window['frequency'].mean(),
                'recency': recency
            }
            train_data.append(features)

    return pd.DataFrame(train_data)


def filter_stable_users(tmp_features):
    stable_features = tmp_features.drop(tmp_features[(tmp_features['total_visits_3m'] < 3) |
                                                     (tmp_features['recency'] > 60)].index)
    return stable_features


def get_every3m_overall_data(all_features):
    overall_features = pd.DataFrame(columns=['start_month', 'end_month',
                                             'overall_avg_total_amount', 'overall_avg_total_visits'])
    for i, group_3m in all_features.groupby(['start_month', 'end_month']):
        curr_row = [group_3m['start_month'].iloc[0], group_3m['end_month'].iloc[0],
                    group_3m['total_spending_3m'].mean(), group_3m['total_visits_3m'].mean()]
        overall_features.loc[len(overall_features)] = curr_row
    return pd.DataFrame(overall_features)


def get_features_labels(file_path):
    # data cleaning (drop null, duplicated and abnormal)
    df = pd.read_csv(file_path)
    df = dp.preprocess(df)
    # print(df.info())

    # filter consumption records with third party names
    df_consume = df.drop(df[(df['Amount'] > 0) | (pd.isna(df['Third Party Name']))].index)
    df_consume = df_consume.drop(columns='Third Party Account No')
    df_consume["Amount"] = -df_consume["Amount"]

    # find consumption features for each third party
    df_consume['YearMonth'] = df_consume['Datetime'].dt.to_period('M')

    user_monthly_features = df_consume.groupby(['Account No', 'YearMonth']).agg(
        total_amount=('Amount', 'sum'),  # total spending per month
        frequency=('Amount', 'count'),  # total visits per month
    ).reset_index()
    user_monthly_features = fill_missing_months(user_monthly_features)
    user_monthly_features.fillna(0, inplace=True)

    three_months_features = extract_3m_data(user_monthly_features, 3, df_consume)
    three_months_features = filter_stable_users(three_months_features)
    three_months_overall_features = get_every3m_overall_data(three_months_features)

    three_months_features['churn_label'] = 0
    for index, row in three_months_features.iterrows():
        overall_row = three_months_overall_features[(three_months_overall_features['start_month'] == row['start_month'])
                                                    & (three_months_overall_features['end_month'] == row['end_month'])]
        churn = set_thresholds(row, overall_row)
        three_months_features.at[index, 'churn_label'] = int(churn)

    churn_per = three_months_features['churn_label'].mean()
    print("churn percentage: ", churn_per)
    print("three_months_features: ", three_months_features)
    return three_months_features


path = "../../simulated_fake_transactions_dataset_2.csv"
three_months_features = get_features_labels(path)
sns.histplot(three_months_features[three_months_features['churn_label'] == 1]['total_spending_3m'], kde=True,
             color='red', label='Churn')
sns.histplot(three_months_features[three_months_features['churn_label'] == 0]['total_spending_3m'], kde=True,
             color='blue', label='Non-Churn')
plt.legend()
plt.title('Churn vs. Non-Churn Monetary Distribution')
plt.show()

sns.histplot(three_months_features[three_months_features['churn_label'] == 1]['recency'], kde=True,
             color='red', label='Churn')
sns.histplot(three_months_features[three_months_features['churn_label'] == 0]['recency'], kde=True,
             color='blue', label='Non-Churn')
plt.legend()
plt.title('Churn vs. Non-Churn Recency Distribution')
plt.show()

sns.histplot(three_months_features[three_months_features['churn_label'] == 1]['total_visits_3m'], kde=True,
             color='red', label='Churn')
sns.histplot(three_months_features[three_months_features['churn_label'] == 0]['total_visits_3m'], kde=True,
             color='blue', label='Non-Churn')
plt.legend()
plt.title('Churn vs. Non-Churn Frequency Distribution')
plt.show()
