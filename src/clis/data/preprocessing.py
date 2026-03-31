import numpy as np
import pandas as pd
from datetime import datetime
import re
import matplotlib.pyplot as plt


def is_valid_date(date):
    date_format = '%d/%m/%Y'
    try:
        datetime.strptime(str(date), date_format)
        return True
    except ValueError:
        print(str(date))
        return False


def is_valid_timestamp(timestamp):
    timestamp = str(timestamp)
    pattern = re.compile(r'^\d{2}:\d{2}$')
    if not bool(pattern.match(timestamp)):
        print(timestamp)
    return bool(pattern.match(timestamp))


def is_valid_account(account_no):
    if np.isnan(account_no):
        return True
    account_no = str(account_no).split('.')[0]
    if len(account_no) != 9:
        return False
    if not account_no.isdigit():
        return False
    return True


def preprocess(df):
    # 3. handle the null values
    # drop null in columns Date, Timestamp, Account No, Balance and Amount
    df = df.dropna(subset=['Date', 'Timestamp', 'Account No', 'Balance', 'Amount'])

    # 4. handle the duplicated values
    # delete the duplicated rows
    df = df.drop_duplicates()

    # 5. handle the abnormal values
    # check the length and format of Account no and Third Party Account No
    df = df[(df["Third Party Account No"].apply(is_valid_account)) & (df["Account No"].apply(is_valid_account))
            & (df["Date"].apply(is_valid_date)) & (df["Timestamp"].apply(is_valid_timestamp))]

    # 6. change dtype
    # Account No/Third Party Account No: float=>string
    df['Account No'] = df['Account No'].apply(lambda x: str(int(x)))
    df['Third Party Account No'] = df['Third Party Account No'].apply(lambda x: str(int(x)) if not np.isnan(x) else x)
    # Date+Timestamp: string=>datetime
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Timestamp'], format="%d/%m/%Y %H:%M")
    df = df.drop(columns=['Date', 'Timestamp'])

    return df
