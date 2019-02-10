"""
Jimmy Goddard
CS 677 Homework 1
2/2/19
"""
import calendar
import itertools
import operator
import os
import statistics

from stock_data import get_stock


def get_data_table():
    """
    Retrieves stock data, writes it to a CSV file and returns it as a matrix.  Provided to us as is by the course
    Professor

    :return: data table matrix
    """
    # ticker = 'GS'  # Goldman Sachs Group Inc
    # ticker = 'GDDY'  # GoDaddy
    # ticker = 'GM'  # General Motors
    ticker = 'GRUB'  # GrubHub
    start_date = '2014-01-01'
    end_date = '2018-12-31'
    s_window = 14
    l_window = 50
    home_dir = os.path.join(os.path.sep + 'home', 'jgoddard')  # Linux home directory
    # home_dir = os.path.join('C:', os.path.sep, 'Users', 'jimmy_000')  # MS Windows home directory
    input_dir = os.path.join(home_dir, 'src', 'git', 'CS677', 'datasets')
    output_file = os.path.join(input_dir, ticker + '.csv')

    if not os.path.isfile(output_file):
        df = get_stock(ticker, start_date, end_date, s_window, l_window)
        df.to_csv(output_file, index=False)

    with open(output_file) as in_file:
        # split csv file into a list of lists.  first item in the list are the headers
        lines = [line.split(',') for line in in_file.read().splitlines()]

    return lines


def construct_returns_dict(data_table, by_column, key_formatter=lambda x: x):
    """
    Construct a dictionary from stock data matrix grouped by the by_column with a list of returns as the values

    :param data_table: matrix of stock data with the first row being the headers
    :param by_column: column to group by
    :param key_formatter: optional argument which will be used to format the by_column before it becomes a key in the
    returns dictionary
    :return: a dictionary containing the groupings from by_column of returns values
    """
    returns_dict = {}
    by_column_idx = None
    return_idx = None
    for i, line in enumerate(data_table):
        if i == 0:
            by_column_idx = line.index(by_column)
            return_idx = line.index('Return')
            continue
        key = key_formatter(line[by_column_idx])
        return_value = float(line[return_idx])
        if key in returns_dict:
            returns_dict[key].append(return_value)
        else:
            returns_dict[key] = [return_value]
    return returns_dict


def construct_values_dict(returns_dict):
    """
    Construct a dictionary of returns statistics for each grouping created by construct_returns_dict

    :param returns_dict: the output of the construct_returns_dict function
    :return: a dictionary containing statistics for each returns list in the returns_dict
    """
    values_dict = {}
    for key in returns_dict:
        returns = returns_dict[key]
        values_dict[key] = {
            'min': min(returns),
            'max': max(returns),
            'average': statistics.mean(returns),
            'median': statistics.median(returns)
        }
    return values_dict


def get_max_average(values_dict):
    """
    Contruct a dictionary containing the max value and which grouping it was for
    :param values_dict: output of construct_values_dict function
    :return: a dictionary with a "max" key and a "key" key which identifies which grouping in values_dict had the max
    average
    """
    average_dict = {'key': None, 'max': None}
    for key in values_dict:
        if average_dict['max'] is None:
            average_dict['max'] = values_dict[key]['average']
        elif values_dict[key]['average'] > average_dict['max']:
            average_dict['max'] = values_dict[key]['average']
            average_dict['key'] = key
    return average_dict


def print_weekday_analysis(weekday_values):
    """
    Helper function to print out the weekdays returned by the construct_values_dict function

    :param weekday_values: dictionary returned by the construct_values_dict when using by_column of 'Weekday'
    :return: void
    """
    print()
    print('{:9}\t{:6}\t{:6}\t{:6}\t{:6}'.format('Weekday', 'min', 'max', 'average', 'median'))
    for day in weekday_values:
        print('{weekday:9}\t{min:6.2f}\t{max:6.2f}\t{average:6}\t{median:6.2f}'.format(
            weekday=day, **weekday_values[day]))

    max_weekday_avg = get_max_average(weekday_values)
    print('{key} has the highest average of {max}'.format(**max_weekday_avg))


def print_month_analysis(month_values):
    """
    Helper function to print out the months returned by the construct_values_dict function

    :param month_values: dictionary returned by the construct_values_dict when using by_column of 'Month'
    :return: void
    """
    print()
    print('{:9}\t{:6}\t{:6}\t{:6}\t{:6}'.format('Month', 'min', 'max', 'average', 'median'))
    for month in month_values:
        print('{month:9}\t{min:6.2f}\t{max:6.2f}\t{average:6.2f}\t{median:6.2f}'.format(
            month=month, **month_values[month]))

    max_month_avg = get_max_average(month_values)
    print('{key} has the highest average of {max}'.format(**max_month_avg))


def get_runs(data_table):
    """
    Return a list of lists containing the indexes of all negative Return runs in the data table

    :param data_table: a matrix of stock values with the first row being the headers
    :return: a list of lists containing the row indexes for the data table of all negative Return runs in the data table
    """
    return_idx = data_table[0].index('Return')
    neg_indexes = [i + 1 for i, item in enumerate(data_table[1:]) if float(item[return_idx]) < 0]

    # From https://stackoverflow.com/questions/2361945/detecting-consecutive-integers-in-a-list
    # Example of detecting consecutive integers: https://docs.python.org/2.6/library/itertools.html#examples
    # Comments on SO post describe adjusting the 2.6 example for Python 3
    runs = []
    for k, g in itertools.groupby(enumerate(neg_indexes), lambda ix: ix[0] - ix[1]):
        runs.append(list(map(operator.itemgetter(1), g)))
    return runs


def get_run_end_indexes(runs, w):
    """
    Given a list of lists of runs and a window size, return all indexes that mark the of a run of that window size

    :param runs: list of lists of runs row index numbers
    :param w: size of the runs window
    :return: flat list of all row indexes marking the end of a run of length w
    """
    greater_than_w = [run for run in runs if len(run) >= w]
    run_end_indexes = [run[w - 1::w] for run in greater_than_w]
    return [item for sublist in run_end_indexes for item in sublist]


def get_trades(runs, data_table):
    """
    Construct a list of lists.  Each index in the outer list corresponds to a list of trades where the index equals w -1
    and w is the window size.  This function will construct this matrix for w equal to 1, 2, 3, 4, and 5

    :param runs: list of lists of indexes corresponding to negative returns runs
    :param data_table: matrix of stock data with the first row being the headers
    :return: list of lists of trades constructed from the runs passed in. each list in the list contains data for
    w = i + 1 where i is the index in the outer list
    """
    current_balance = 100  # dollars
    adj_stock_price_idx = data_table[0].index('Adj Close')
    trades = []
    for w in range(1, 6):
        run_end_indexes = get_run_end_indexes(runs, w)
        trades.append([])
        for idx in run_end_indexes:
            buy_price = float(data_table[idx][adj_stock_price_idx])
            num_stocks = current_balance / buy_price
            sell_price = float(data_table[idx + 1][adj_stock_price_idx])
            trade = current_balance - (num_stocks * sell_price)
            trades[w - 1].append(trade)
    return trades


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
    print('Trade Strategy One Analysis')
    print('W\tTrades\t# Profitable Trades\tProfit per Profitable Trade\t# Losing Trades\tLoss per Losing Trade')
    for i, trade in enumerate(trades):
        w = i + 1
        num_trades = get_num_trades(trade)
        num_prof_trades = get_num_profitable_trades(trade)
        prof_per_prof_trade = get_profit_per_profitable_trade(trade)
        num_losing_trades = get_num_losing_trades(trade)
        loss_per_losing_trade = get_loss_per_losing_trade(trade)
        print('{}\t{:6}\t{:19}\t{:27.2f}\t{:15}\t{:21.2f}'.format(
            w, num_trades, num_prof_trades, prof_per_prof_trade, num_losing_trades, loss_per_losing_trade))


def get_trades_strategy_two(data_table):
    """
    Execute the second trading strategy

    :param data_table: matrix of stock data with the first row being the headers
    :return: list of trades
    """
    total_buys = 0
    total_num_stocks = 0
    current_balance = 100  # dollars
    adj_stock_price_idx = data_table[0].index('Adj Close')
    short_ma_idx = data_table[0].index('Short_MA')
    trades = []
    for row in data_table[1:]:
        adj_close = float(row[adj_stock_price_idx])
        s_ma = float(row[short_ma_idx])
        if adj_close > s_ma:
            num_stocks = current_balance / adj_close
            total_num_stocks += num_stocks
            total_buys += 1
        else:
            if total_num_stocks > 0:
                trade = (current_balance * total_buys) - (total_num_stocks * adj_close)
                trades.append(trade)
                total_buys = 0
                total_num_stocks = 0
    return trades


def print_strategy_two_trades(trades):
    """
    Pretty print the trades returned by get_trades_strategy_two

    :param trades: list of trades
    :return: void
    """
    print()
    print('Trade Strategy Two Analysis')
    print('Trades\t# Profitable Trades\tProfit per Profitable Trade\t# Losing Trades\tLoss per Losing Trade')
    num_trades = get_num_trades(trades)
    num_prof_trades = get_num_profitable_trades(trades)
    prof_per_prof_trade = get_profit_per_profitable_trade(trades)
    num_losing_trades = get_num_losing_trades(trades)
    loss_per_losing_trade = get_loss_per_losing_trade(trades)
    print('{:6}\t{:19}\t{:27.2f}\t{:15}\t{:21.2f}'.format(
        num_trades, num_prof_trades, prof_per_prof_trade, num_losing_trades, loss_per_losing_trade))


def main():
    """
    Main procedure

    :return: void
    """
    data_table = get_data_table()

    # Homework Question 1 - data broken out by weekday
    weekday_dict = construct_returns_dict(data_table, 'Weekday')
    weekday_values = construct_values_dict(weekday_dict)
    print_weekday_analysis(weekday_values)

    # Homework Question 2 - data broken out by month
    month_dict = construct_returns_dict(data_table, 'Month', lambda idx: calendar.month_name[int(idx)])
    month_values = construct_values_dict(month_dict)
    print_month_analysis(month_values)

    # Homework Question 3, Trading Strategy 1
    runs = get_runs(data_table)
    trades = get_trades(runs, data_table)
    print_trade_analysis(trades)

    # Homework Question 3, Trading Strategy 2
    trades = get_trades_strategy_two(data_table)
    print_strategy_two_trades(trades)


if __name__ == '__main__':
    main()
