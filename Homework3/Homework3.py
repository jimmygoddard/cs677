"""
Jimmy Goddard
2/18/19
CS 677 Data Analytics with Python
Homework 3
"""
import datetime
import os
import platform
import statistics

import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as web

HEADER_LONG = 'Long_MA'
HEADER_SHORT = 'Short_MA'
HEADER_WEEK = 'Week'
HEADER_RETURN = 'Return'
HEADER_DATE = 'Date'
HEADER_AVG = 'Rolling AVG'
HEADER_SD = 'Rolling SD'
HEADER_PRICE = 'Adj Close'


def get_stock(ticker, start_date, end_date, s_window, l_window):
    try:
        df = web.get_data_yahoo(ticker, start=start_date, end=end_date)
        df[HEADER_RETURN] = df[HEADER_PRICE].pct_change()
        df[HEADER_RETURN].fillna(0, inplace=True)
        df[HEADER_DATE] = df.index
        df[HEADER_DATE] = pd.to_datetime(df[HEADER_DATE])
        df['Month'] = df[HEADER_DATE].dt.month
        df['Year'] = df[HEADER_DATE].dt.year
        df['Day'] = df[HEADER_DATE].dt.day
        for col in ['Open', 'High', 'Low', 'Close', HEADER_PRICE]:
            df[col] = df[col].round(2)
        df['Weekday'] = df[HEADER_DATE].dt.weekday_name
        df[HEADER_SHORT] = df[HEADER_PRICE].rolling(window=s_window, min_periods=1).mean()
        df[HEADER_LONG] = df[HEADER_PRICE].rolling(window=l_window, min_periods=1).mean()
        col_list = [HEADER_DATE, 'Year', 'Month', 'Day', 'Weekday', 'Open',
                    'High', 'Low', 'Close', 'Volume', HEADER_PRICE,
                    HEADER_RETURN, HEADER_SHORT, HEADER_LONG]
        df = df[col_list]
        return df
    except Exception as error:
        print(error)
        return None


def get_last_digit(y):
        x = str(round(float(y), 2))
        x_list = x.split('.')
        fraction_str = x_list[1]
        if len(fraction_str) == 1:
            return 0
        else:
            return int(fraction_str[1])


def get_data_table():
    """
    Retrieves stock data, writes it to a CSV file and returns it as a matrix.  Provided to us as is by the course
    Professor

    :return: data table matrix
    """
    ticker = 'GS'  # Goldman Sachs Group Inc
    # ticker = 'GDDY'  # GoDaddy
    # ticker = 'GM'  # General Motors
    # ticker = 'GRUB'  # GrubHub
    start_date = '2014-01-01'
    end_date = '2018-12-31'
    s_window = 14
    l_window = 50

    if platform.system() == 'Windows':
        home_dir = os.path.join('C:', os.path.sep, 'Users', 'jimmy_000')  # MS Windows home directory
    else:  # Assumes Linux
        home_dir = os.path.join(os.path.sep + 'home', 'jgoddard')  # Linux home directory
    input_dir = os.path.join(home_dir, 'src', 'git', 'CS677', 'datasets')
    output_file = os.path.join(input_dir, ticker + '.csv')

    if not os.path.isfile(output_file):
        df = get_stock(ticker, start_date, end_date, s_window, l_window)
        df.to_csv(output_file, index=False)
    else:
        df = pd.read_csv(output_file)
    return df


def get_week(local_date):
    date_format = '%Y-%M-%d'
    dt = datetime.datetime.strptime(local_date, date_format)
    start = dt - datetime.timedelta(days=dt.weekday())
    return start.strftime(date_format)


def label_good_weeks(good_weeks):
    def get_label(local_date):
        if local_date in good_weeks:
            return 1
        else:
            return 0
    return get_label


def get_rolling_std_dev(df, w):
    s = df[HEADER_PRICE].rolling(window=w, min_periods=1).std()
    s.name = HEADER_SD
    return s


def get_rolling_avg(df, w):
    s = df[HEADER_PRICE].rolling(window=w, min_periods=1).mean()
    s.name = HEADER_AVG
    return s


def trade_bollinger(df, w, k):
    new_df = pd.concat([df[HEADER_PRICE], get_rolling_avg(df, w), get_rolling_std_dev(df, w)], axis=1)
    current_balance = 100
    stock_buys = []
    pnl = []
    for index, row in new_df.iterrows():
        close_price = row[HEADER_PRICE]
        ma = row[HEADER_AVG]
        sd = row[HEADER_SD]
        if close_price < ma - (k * sd):
            # Buying stocks
            num_stocks = current_balance / close_price
            stock_buys.append((close_price, num_stocks))
        elif close_price > ma + (k * sd):
            while len(stock_buys) > 0:
                # Selling stocks
                buy_price, num_stocks = stock_buys.pop()
                pnl.append((close_price - buy_price) / buy_price)
    return pnl


def construct_bollinger_data_points(df):
    k_values = [value * 0.1 for value in range(5, 31, 5)]
    w_values = list(range(10, 110, 10))
    data = []
    for k in k_values:
        for w in w_values:
            pnl = trade_bollinger(df, w, k)
            if len(pnl) > 0:
                data.append({'k': k, 'w': w, 'value': statistics.mean(pnl)})
    return data


def get_bollinger_plotting_data(data):
    x_values = list(map(lambda datum: datum['w'], data))
    y_values = list(map(lambda datum: datum['k'], data))
    data_values = list(map(lambda datum: datum['value'], data))
    color_values = list(map(lambda datum: 'g' if datum['value'] > 0 else 'r', data))
    return x_values, y_values, data_values, color_values


def construct_second_strategy_data(df):
    w_values = list(range(10, 110, 10))
    data = []
    for short_w in w_values:
        for long_w in w_values:
            if short_w > long_w:
                continue
            pnl = trade_strategy_two(df, short_w, long_w)
            if len(pnl) > 0:
                data.append({'short_w': short_w, 'long_w': long_w, 'value': statistics.mean(pnl)})
    return data


def get_short_ma(df, w):
    s = df[HEADER_PRICE].rolling(window=w, min_periods=1).mean()
    s.name = HEADER_SHORT
    return s


def get_long_ma(df, w):
    s = df[HEADER_PRICE].rolling(window=w, min_periods=1).mean()
    s.name = HEADER_LONG
    return s


def trade_strategy_two(df, short_w, long_w):
    new_df = pd.concat([df[HEADER_PRICE], get_short_ma(df, short_w), get_long_ma(df, long_w)], axis=1)
    current_balance = 100
    stock_buys = []
    pnl = []
    for index, row in new_df.iterrows():
        close_price = row[HEADER_PRICE]
        short_ma = row[HEADER_SHORT]
        long_ma = row[HEADER_LONG]
        if short_ma > long_ma:
            # Buying stocks
            num_stocks = current_balance / close_price
            stock_buys.append((close_price, num_stocks))
        elif short_ma < long_ma:
            while len(stock_buys) > 0:
                # Selling stocks
                buy_price, num_stocks = stock_buys.pop()
                pnl.append((close_price - buy_price) / buy_price)
    return pnl


def get_second_strategy_plotting_data(data):
    x_values = list(map(lambda datum: datum['long_w'], data))
    y_values = list(map(lambda datum: datum['short_w'], data))
    data_values = list(map(lambda datum: datum['value'], data))
    color_values = list(map(lambda datum: 'g' if datum['value'] > 0 else 'r', data))
    return x_values, y_values, data_values, color_values


# Assignment 1
gs_df = get_data_table()
gs_df[HEADER_WEEK] = gs_df[HEADER_DATE].apply(get_week)
# criteria for a good week is that the sum of the returns for each day of that week were positive:
is_good_return_by_week = gs_df[HEADER_RETURN].groupby(gs_df[HEADER_WEEK]).sum() > 0
only_good = is_good_return_by_week[is_good_return_by_week == True].index
positive_week_dates = list(only_good)
gs_df['Good_Week'] = gs_df[HEADER_WEEK].apply(label_good_weeks(positive_week_dates))

# Assignment 2
bollinger_data = construct_bollinger_data_points(gs_df)
x, y, values, colors = get_bollinger_plotting_data(bollinger_data)

s = [abs(value * 5000) for value in values]
plt.scatter(x=x, y=y, s=s, c=colors)
plt.show()

# Assignment 3
second_strategy_data = construct_second_strategy_data(gs_df)
x, y, values, colors = get_second_strategy_plotting_data(second_strategy_data)
s = [abs(value * 5000) for value in values]
plt.scatter(x=x, y=y, s=s, c=colors)
plt.show()




