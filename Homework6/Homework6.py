"""
Jimmy Goddard
3/9/19
CS 677 Assignment 6
"""
import datetime
import os
import platform
import statistics

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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
HEADER_OPEN = 'Open'
HEADER_OVERNIGHT = 'Overnight Gain'


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
        for col in [HEADER_OPEN, 'High', 'Low', 'Close', HEADER_PRICE]:
            df[col] = df[col].round(2)
        df['Weekday'] = df[HEADER_DATE].dt.weekday_name
        df[HEADER_SHORT] = df[HEADER_PRICE].rolling(window=s_window, min_periods=1).mean()
        df[HEADER_LONG] = df[HEADER_PRICE].rolling(window=l_window, min_periods=1).mean()
        col_list = [HEADER_DATE, 'Year', 'Month', 'Day', 'Weekday', HEADER_OPEN,
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


def get_data_table(ticker='GS', start_date='2014-01-01', end_date='2018-12-31'):
    """
    Retrieves stock data, writes it to a CSV file and returns it as a matrix.  Provided to us as is by the course
    Professor

    :return: data table matrix
    """
    # ticker = 'GS'  # Goldman Sachs Group Inc
    # ticker = 'GDDY'  # GoDaddy
    # ticker = 'GM'  # General Motors
    # ticker = 'GRUB'  # GrubHub
    # start_date = '2014-01-01'
    # end_date = '2018-12-31'
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


def inertia_strategy(df, r=-100):
    current_balance = 100
    pnl = []
    for index, row in df.iterrows():
        overnight_gain = row[HEADER_OVERNIGHT]
        open_price = row[HEADER_OPEN]
        close_price = row[HEADER_PRICE]
        if overnight_gain * 100 > r:  # willing to trade
            if overnight_gain == 0:  # do nothing
                continue
            elif overnight_gain < 0:  # sell short
                num_stocks = current_balance / open_price
                pnl.append((open_price - close_price) * num_stocks)
            else:  # long position
                num_stocks = current_balance / open_price
                pnl.append((close_price - open_price) * num_stocks)
    return pnl


def reverse_inertia_strategy(df, r=-100):
    current_balance = 100
    pnl = []
    for index, row in df.iterrows():
        overnight_gain = row[HEADER_OVERNIGHT]
        open_price = row[HEADER_OPEN]
        close_price = row[HEADER_PRICE]
        if overnight_gain * 100 > r:  # willing to trade
            if overnight_gain == 0:  # do nothing
                continue
            elif overnight_gain > 0:  # sell short
                num_stocks = current_balance / open_price
                pnl.append((open_price - close_price) * num_stocks)
            else:  # long position
                num_stocks = current_balance / open_price
                pnl.append((close_price - open_price) * num_stocks)
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
    print('Inertia Trade Strategy Analysis')
    print('Trades\t# Profitable Trades\tProfit per Profitable Trade\t# Losing Trades\tLoss per Losing Trade')
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

# Task 1
overnight_gains = [0]
gs_df.loc[0, HEADER_OVERNIGHT] = 0
for i in range(1, len(gs_df)):
    # (open - previous close) / previous close
    open_price = gs_df.loc[i, HEADER_OPEN]
    previous_close = gs_df.loc[i - 1, HEADER_PRICE]
    gs_df.loc[i, HEADER_OVERNIGHT] = (open_price - previous_close) / previous_close
    overnight_gains.append((open_price - previous_close) / previous_close)

gs_df[HEADER_OVERNIGHT] = overnight_gains

# Task 1 part a
gs_2018 = gs_df[gs_df['Year'] == 2018]
pnl = inertia_strategy(gs_2018)
print_trade_analysis(pnl)
# Inertia Trade Strategy Analysis
# Trades	# Profitable Trades	Profit per Profitable Trade	# Losing Trades	Loss per Losing Trade
#    251	                 44	                       1.06	            207	                -1.74

# reverse strategy
reverse_pnl = reverse_inertia_strategy(gs_2018)
print_trade_analysis(reverse_pnl)


# Task 1 part b
plt.scatter(x=gs_2018[HEADER_OVERNIGHT].values, y=gs_2018[HEADER_RETURN].values)
plt.xlabel('Overnight Return')
plt.ylabel('Daily Return')
plt.title('2018 Daily Returns vs Overnight Returns')
plt.show()

print(gs_2018[[HEADER_OVERNIGHT, HEADER_RETURN]].corr())
#                 Overnight Gain    Return
# Overnight Gain        1.000000  0.390044
# Return                0.390044  1.000000

# Based on both the scatter plot as well as the Pearson Correlation Coefficient, there appears to be very little
# positive linear relationship between overnight returns and daily returns

# Task 1 part c
x = range(-10, 11, 1)
y = []
for r in x:
    pnl = inertia_strategy(gs_2018, r)
    if len(pnl) > 0:
        y.append(statistics.mean(pnl))
    else:
        y.append(None)

x = range(-10, 11, 1)
reverse_y = []
for r in x:
    reverse_pnl = reverse_inertia_strategy(gs_2018, r)
    if len(reverse_pnl) > 0:
        reverse_y.append(statistics.mean(reverse_pnl))
    else:
        reverse_y.append(None)

plt.scatter(x=x, y=reverse_y)
plt.title('Average gain per R value')
plt.xlabel('R Value')
plt.ylabel('Average gain')
plt.show()

# Task 2
tips = sns.load_dataset('tips')
tips['percentage'] = ((tips['tip'] / tips['total_bill']) * 100).round(2)
# question 1
print(tips.groupby(['time'])['tip'].mean())
# time
# Lunch     2.728088
# Dinner    3.102670
# Name: tip, dtype: float64

# question 2
print(tips.groupby(['day', 'time'])['tip'].mean())
# day   time
# Thur  Lunch     2.767705
#       Dinner    3.000000
# Fri   Lunch     2.382857
#       Dinner    2.940000
# Sat   Dinner    2.993103
# Sun   Dinner    3.255132
# Name: tip, dtype: float64
print(tips.groupby(['day', 'time'])['percentage'].mean())
# day   time
# Thur  Lunch     16.129016
#       Dinner    15.970000
# Fri   Lunch     18.875714
#       Dinner    15.892500
# Sat   Dinner    15.314598
# Sun   Dinner    16.689605
# Name: percentage, dtype: float64

# question 3
sns.regplot(x="total_bill", y="tip", data=tips)
plt.title('Total bill vs Tip')
plt.xlabel('Total Bill')
plt.ylabel('Tip')
plt.show()

print(tips[['total_bill', 'percentage']].corr())
#             total_bill       tip
# total_bill    1.000000  0.675734
# tip           0.675734  1.000000

print(tips[['tip', 'percentage']].corr())
#                  tip  percentage
# tip         1.000000    0.342361
# percentage  0.342361    1.000000

sns.regplot(x="tip", y="percentage", data=tips)
plt.title('Tip Percentage vs Tip')
plt.xlabel('Tip')
plt.ylabel('Tip percentage')
plt.show()

sns.regplot(x="total_bill", y="percentage", data=tips)
plt.title('Total bill vs Tip Percentage')
plt.xlabel('Total Bill')
plt.ylabel('Tip Percentage')
plt.show()

print(tips[['total_bill', 'tip']].corr())
# question 4
sns.regplot(x='total_bill', y='percentage', data=tips)
plt.title('Total bill vs Tip percentage')
plt.xlabel('Total Bill')
plt.ylabel('Tip percentage')
plt.show()

print(tips[['total_bill', 'percentage']].corr())
#             total_bill  percentage
# total_bill    1.000000   -0.338629
# percentage   -0.338629    1.000000

# question 5
print(len(tips[tips['smoker'] == 'Yes']) / len(tips) * 100)
# 38.114754098360656

# question 6
sns.regplot(x=tips.index.values, y=tips['tip'])
plt.title('Tips over time')
plt.xlabel('Row index')
plt.ylabel('Tip')
plt.show()

# question 7
sns.catplot(x='sex', col='time', kind='count', data=tips)
plt.show()

# question 8
sns.boxplot(x='smoker', y='tip', data=tips)
plt.show()

sns.boxplot(x='smoker', y='percentage', data=tips)
plt.show()

# question 9
print(tips.groupby(['day'])['tip'].mean())
# day
# Thur    2.771452
# Fri     2.734737
# Sat     2.993103
# Sun     3.255132
# Name: tip, dtype: float64
print(tips.groupby(['day'])['percentage'].mean())
# day
# Thur    16.126452
# Fri     16.991579
# Sat     15.314598
# Sun     16.689605
# Name: percentage, dtype: float64


# question 10
sns.catplot(x='smoker', col='sex', kind='count', data=tips)
plt.show()

males = tips[tips['sex'] == 'Male']
females = tips[tips['sex'] != 'Male']
male_smokers = males[males['smoker'] == 'Yes']
female_smokers = females[females['smoker'] == 'Yes']
print(len(male_smokers) / len(males))
# 0.3821656050955414
print(len(female_smokers) / len(females))
# 0.3793103448275862
