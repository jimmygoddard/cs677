"""
Jimmy Goddard
2/24/19
CS 677 Assignment 4
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
HEADER_YEAR = 'Year'
HEADER_LABEL = 'Week Label'


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
    date_format = '%Y-%m-%d'
    dt = datetime.datetime.strptime(local_date, date_format)
    return dt.isocalendar()[1]  # get number of week in year


def label_good_weeks(good_weeks):
    def get_label(local_date):
        if local_date in good_weeks:
            return 'green'
        else:
            return 'red'
    return get_label


def naive_strategy(df):
    current_balance = 100
    stock_buys = []
    pnl = []
    is_holding_stock = False
    for index, row in df.iterrows():
        close_price = row[HEADER_PRICE]
        return_amount = row[HEADER_RETURN]
        if return_amount >= 0 and not is_holding_stock:
            # Buying stocks
            num_stocks = current_balance / close_price
            stock_buys.append((close_price, num_stocks))
            is_holding_stock = True
        elif return_amount < 0:
            if len(stock_buys) > 0:
                is_holding_stock = False
            while len(stock_buys) > 0:
                # Selling stocks
                buy_price, num_stocks = stock_buys.pop()
                pnl.append((close_price - buy_price) / buy_price)

    return pnl


def get_num_trades(trades):
    """
    Helper function to get the number of trades

    :param trades: list of trades
    :return: number of trades
    """
    return len(trades)


def get_num_profitable_trades(trades):
    """
    Helper function to count the number of profitable trades

    :param trades: list of trades
    :return: number of profitable trades
    """
    return len([trade for trade in trades if trade >= 0])


def get_num_losing_trades(trades):
    """
    Helper function to count the number of losing trades

    :param trades: list of trades
    :return: number of losing trades
    """
    return len([trade for trade in trades if trade < 0])


def get_profit_per_profitable_trade(trades):
    """
    Helper function to calculate the average profit per profitable trade

    :param trades: list of trades
    :return: mean of profitable trades
    """
    return statistics.mean([trade for trade in trades if trade >= 0])


def get_loss_per_losing_trade(trades):
    """
    Helper function to calculate the average loss per losing trade

    :param trades: list of trades
    :return: mean of losing trades
    """
    return statistics.mean([trade for trade in trades if trade < 0])


def print_trade_analysis(trades):
    """
    Pretty print the trades

    :param trades: list of trades
    :return: void
    """
    print()
    print('Naive Strategy Analysis')
    print('W\tTrades\t# Profitable Trades\tProfit per Profitable Trade\t# Losing Trades\tLoss per Losing Trade')
    num_trades = get_num_trades(trades)
    num_prof_trades = get_num_profitable_trades(trades)
    prof_per_prof_trade = get_profit_per_profitable_trade(trades)
    num_losing_trades = get_num_losing_trades(trades)
    loss_per_losing_trade = get_loss_per_losing_trade(trades)
    print('{:6}\t{:19}\t{:27.2f}\t{:15}\t{:21.2f}'.format(
        num_trades, num_prof_trades, prof_per_prof_trade, num_losing_trades, loss_per_losing_trade))


gs_df = get_data_table()
gs_df[HEADER_WEEK] = gs_df[HEADER_DATE].apply(get_week)
# criteria for a good week is that the sum of the returns for each day of that week were positive:
is_good_return_by_week = gs_df[HEADER_RETURN].groupby(gs_df[HEADER_WEEK]).sum() > 0
only_good = is_good_return_by_week[is_good_return_by_week == True].index
positive_week_dates = list(only_good)
gs_df[HEADER_LABEL] = gs_df[HEADER_WEEK].apply(label_good_weeks(positive_week_dates))

# Question 1
naive_data = naive_strategy(gs_df)
print_trade_analysis(naive_data)

# Simple Naive Strategy Analysis (buy every time return is positive)
# W	Trades	# Profitable Trades	Profit per Profitable Trade	# Losing Trades	Loss per Losing Trade
#    643	                241	                       0.02	            402	                -0.01

# Buy-Once Naive Strategy Analysis (buy once you get one positive return, then hold those stocks through more positive
# return days and finally sell on the first negative return day)
# W	Trades	# Profitable Trades	Profit per Profitable Trade	# Losing Trades	Loss per Losing Trade
#    316	                108	                       0.02	            208	                -0.01

year_2018_df = gs_df[gs_df[HEADER_YEAR] == 2018]
week_returns = year_2018_df[HEADER_RETURN].groupby(year_2018_df[HEADER_WEEK])
x = week_returns.mean().values
y = week_returns.std().values
s = week_returns.sum().values
returns = [group.values for name, group in week_returns]
s2 = [abs(week[-1] - week[0]) for week in returns]
# scale values
s = [abs(value * 5000) for value in s]
s2 = [abs(value * 5000) for value in s2]
colors = [group.values[0] for name, group in year_2018_df[HEADER_LABEL].groupby(year_2018_df[HEADER_WEEK])]

plt.title('Sum of each return in a week')
plt.xlabel('Weekly average return')
plt.ylabel('Weekly return standard deviation')
plt.scatter(x=x, y=y, s=s, c=colors)
plt.show()

plt.title('End of week return - Beginning of week return')
plt.xlabel('Weekly average return')
plt.ylabel('Weekly return standard deviation')
plt.scatter(x=x, y=y, s=s2, c=colors)
plt.show()
