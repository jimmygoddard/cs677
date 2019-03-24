"""
Jimmy Goddard
3/23/19
CS 677 Assignment 7
"""
import datetime
import operator
import os
import platform
import statistics

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas_datareader import data as web
from sklearn.preprocessing import StandardScaler

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


def long_position_strategy(df, r):
    current_balance = 100
    pnl = []
    for index, row in df.iterrows():
        overnight_gain = row[HEADER_OVERNIGHT]
        open_price = row[HEADER_OPEN]
        close_price = row[HEADER_PRICE]
        if overnight_gain * 100 > r:  # willing to trade
            if overnight_gain == 0:  # do nothing
                continue
            # long position
            num_stocks = current_balance / open_price
            pnl.append((close_price - open_price) * num_stocks)
    return pnl


def short_position_strategy(df, r):
    current_balance = 100
    pnl = []
    for index, row in df.iterrows():
        overnight_gain = row[HEADER_OVERNIGHT]
        open_price = row[HEADER_OPEN]
        close_price = row[HEADER_PRICE]
        if overnight_gain * 100 < r:  # willing to trade
            if overnight_gain == 0:  # do nothing
                continue
            # long position
            num_stocks = current_balance / open_price
            pnl.append((open_price - close_price) * num_stocks)
    return pnl


def add_overnight_gains(gs_df):
    overnight_gains = [0]
    gs_df.loc[0, HEADER_OVERNIGHT] = 0
    for i in range(1, len(gs_df)):
        # (open - previous close) / previous close
        open_price = gs_df.loc[i, HEADER_OPEN]
        previous_close = gs_df.loc[i - 1, HEADER_PRICE]
        gs_df.loc[i, HEADER_OVERNIGHT] = (open_price - previous_close) / previous_close
        overnight_gains.append((open_price - previous_close) / previous_close)

    gs_df[HEADER_OVERNIGHT] = overnight_gains


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
    profitable_trades = [trade for trade in trades if trade >= 0]
    if len(profitable_trades) > 0:
        return statistics.mean(profitable_trades)
    else:
        return 0


def get_loss_per_losing_trade(trades):
    """
    Helper function to calculate the average loss per losing trade

    :param trades: list of trades
    :return: mean of losing trades
    """
    losing_trades = [trade for trade in trades if trade < 0]
    if len(losing_trades) > 0:
        return statistics.mean(losing_trades)
    else:
        return 0


def print_trade_analysis(trades, r, descriptor):
    """
    Pretty print the trades

    :param trades: list of trades
    :return: void
    """
    print()
    print(f'{descriptor} Trade Strategy Analysis R = {r}')
    print('Trades\t# Profitable Trades\tProfit per Profitable Trade\t# Losing Trades\tLoss per Losing Trade')
    num_trades = get_num_trades(trades)
    num_prof_trades = get_num_profitable_trades(trades)
    prof_per_prof_trade = get_profit_per_profitable_trade(trades)
    num_losing_trades = get_num_losing_trades(trades)
    loss_per_losing_trade = get_loss_per_losing_trade(trades)
    print(f'{num_trades:6}\t{num_prof_trades:19}\t{prof_per_prof_trade:27.2f}\t{num_losing_trades:15}\t{loss_per_losing_trade:21.2f}')
    # print('{:6}\t{:19}\t{:27.2f}\t{:15}\t{:21.2f}'.format(
    #     num_trades, num_prof_trades, prof_per_prof_trade, num_losing_trades, loss_per_losing_trade))


gs_df = get_data_table(end_date='2019-01-08')
add_overnight_gains(gs_df)

long_r = [1, 2, 3, 4, 5]
short_r = [-1, -2, -3, -4, -5]
gs_2018 = gs_df[gs_df['Year'] == 2018]
full_r = []
profitable_trades_percent = []
avg_pnl = []
for r in long_r:
    full_r.append(r)
    pnl = long_position_strategy(gs_2018, r)
    if len(pnl) > 0:
        percent = 100 * get_num_profitable_trades(pnl) / len(pnl)
        profitable_trades_percent.append(percent)
        avg_pnl.append(statistics.mean(pnl))
        print_trade_analysis(pnl, r, 'Long')
    else:
        profitable_trades_percent.append(None)
        avg_pnl.append(None)
        print(f'No trades qualified for R = {r}')

for r in short_r:
    full_r.append(r)
    pnl = short_position_strategy(gs_df, r)
    if len(pnl) > 0:
        percent = 100 * get_num_profitable_trades(pnl) / len(pnl)
        profitable_trades_percent.append(percent)
        avg_pnl.append(statistics.mean(pnl))
        print_trade_analysis(pnl, r, 'Short')
    else:
        profitable_trades_percent.append(None)
        avg_pnl.append(None)
        print(f'No trades qualified for R = {r}')


# Long Trade Strategy Analysis R = 0
# Trades	# Profitable Trades	Profit per Profitable Trade	# Losing Trades	Loss per Losing Trade
#    229	                 27	                       0.81	            202	                -1.74
# Long Trade Strategy Analysis R = 1
# Trades	# Profitable Trades	Profit per Profitable Trade	# Losing Trades	Loss per Losing Trade
#    167	                 19	                       0.65	            148	                -1.72
# Long Trade Strategy Analysis R = 2
# Trades	# Profitable Trades	Profit per Profitable Trade	# Losing Trades	Loss per Losing Trade
#     47	                  4	                       0.33	             43	                -1.91
# Long Trade Strategy Analysis R = 3
# Trades	# Profitable Trades	Profit per Profitable Trade	# Losing Trades	Loss per Losing Trade
#      4	                  1	                       0.18	              3	                -1.65
# No trades qualified for R = 4
# No trades qualified for R = 5
# Short Trade Strategy Analysis R = 0
# Trades	# Profitable Trades	Profit per Profitable Trade	# Losing Trades	Loss per Losing Trade
#     26	                 20	                       1.73	              6	                -2.29
# Short Trade Strategy Analysis R = -1
# Trades	# Profitable Trades	Profit per Profitable Trade	# Losing Trades	Loss per Losing Trade
#      7	                  3	                       2.00	              4	                -2.09
# Short Trade Strategy Analysis R = -2
# Trades	# Profitable Trades	Profit per Profitable Trade	# Losing Trades	Loss per Losing Trade
#      1	                  1	                       4.44	              0	                 0.00
# No trades qualified for R = -3
# No trades qualified for R = -4
# No trades qualified for R = -5

plt.scatter(x=full_r, y=avg_pnl)
plt.xlabel('R')
plt.ylabel('Average PNL')
plt.title('Average PNL vs R')
plt.show()

plt.scatter(x=full_r, y=profitable_trades_percent)
plt.xlabel('R')
plt.ylabel('Profitable Trades Percent')
plt.title('Percent of Profitable Trades vs R')
plt.show()


gs_df[HEADER_WEEK] = gs_df[HEADER_DATE].apply(get_week)
# criteria for a good week is that the sum of the returns for each day of that week were positive:
is_good_return_by_week = gs_df[HEADER_RETURN].groupby(gs_df[HEADER_WEEK]).sum() > 0
only_good = is_good_return_by_week[is_good_return_by_week == True].index
positive_week_dates = list(only_good)
gs_df[HEADER_LABEL] = gs_df[HEADER_WEEK].apply(label_good_weeks(positive_week_dates))

training_df = gs_df[gs_df[HEADER_YEAR] == 2017]
testing_df = gs_df[gs_df[HEADER_YEAR] == 2018]

training_returns = training_df[HEADER_RETURN].groupby(training_df[HEADER_WEEK])
training_means = training_returns.mean().values
training_std = training_returns.std().values
training_labels = [group.values[0] for name, group in training_df[HEADER_LABEL].groupby(training_df[HEADER_WEEK])]

testing_returns = testing_df[HEADER_RETURN].groupby(testing_df[HEADER_WEEK])
testing_means = testing_returns.mean().values
testing_std = testing_returns.std().values
testing_labels = [group.values[0] for name, group in testing_df[HEADER_LABEL].groupby(testing_df[HEADER_WEEK])]

training = pd.DataFrame({'mean': training_means, 'std': training_std, 'label': training_labels})
testing = pd.DataFrame({'mean': testing_means, 'std': testing_std, 'label': testing_labels})

X_train = training[['mean', 'std']].values
Y_train = training[['label']].values.ravel()
X_test = testing[['mean', 'std']].values
Y_test = testing[['label']].values.ravel()

k_values = [1, 3, 5, 7, 9, 11]
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# error_rate = []
# for k in k_values:
#     knn_classifier = KNeighborsClassifier(n_neighbors=k)
#     knn_classifier.fit(X_train, Y_train)
#     pred_k = knn_classifier.predict(X_test)
#     error_rate.append(np.mean(pred_k != Y_test))
#
# x = k_values
# y = error_rate
# plt.title('Error Rate vs k')
# plt.xlabel('number of neighbors: k')
# plt.ylabel('Error Rate')
# plt.plot(x, y, '-bo')
# plt.show()


def euclidean_distance(V):
    return np.linalg.norm(V, ord=2)


def manhattan_distance(V):
    return np.linalg.norm(V, ord=1)


def minkowski_distance(V):
    return np.linalg.norm(V, ord=1.5)


def get_neighbors(training_set, test_instance, k, distance):
    distances = []
    for datum in training_set:
        dist = distance(test_instance - datum[0])
        distances.append((datum, dist))
    return sorted(distances, key=operator.itemgetter(1))[:k]


def get_response(neighbors):
    labels = [neighbor[0][1] for neighbor in neighbors]
    red_count = labels.count('red')
    green_count = labels.count('green')
    if red_count > green_count:
        return 'red'
    return 'green'


train_data = list(zip(X_train, Y_train))
distances = [
    (euclidean_distance, 'euclidean'),
    (manhattan_distance, 'manhattan'),
    (minkowski_distance, 'minkowski')
]
for distance_func, label in distances:
    error_rates = []
    for k in k_values:
        predictions = []
        for test_instance in zip(X_test, Y_test):
            neighbors = get_neighbors(train_data, test_instance[0], k, distance_func)
            response = get_response(neighbors)
            predictions.append(response)
        error_rate = np.mean(np.array(predictions) != Y_test)
        error_rates.append(error_rate)

    x = k_values
    y = error_rates
    plt.title(f'Error Rate vs k for {label}')
    plt.xlabel('number of neighbors: k')
    plt.ylabel('Error Rate')
    plt.plot(x, y, '-bo')
    plt.show()
