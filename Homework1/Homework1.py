import calendar
import itertools
import operator
import os
import statistics

from orig_stock_data import get_stock


def get_data_table():
    # ticker = 'GS'  # Goldman Sachs Group Inc
    # ticker = 'GDDY'  # GoDaddy
    # ticker = 'GM'  # General Motors
    ticker = 'GRUB'  # GrubHub
    start_date = '2014-01-01'
    end_date = '2018-12-31'
    s_window = 14
    l_window = 50
    input_dir = r'C:\Users\jimmy_000\src\git\cs677\datasets'
    output_file = os.path.join(input_dir, ticker + '.csv')

    if not os.path.isfile(output_file):
        df = get_stock(ticker, start_date, end_date, s_window, l_window)
        df.to_csv(output_file, index=False)

    with open(output_file) as in_file:
        # split csv file into a list of lists.  first item in the list are the headers
        lines = [line.split(',') for line in in_file.read().splitlines()]

    return lines


def construct_returns_dict(data_table, by_column, key_formatter=lambda x: x):
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
    average_dict = {'key': None, 'max': None}
    for key in values_dict:
        if average_dict['max'] is None:
            average_dict['max'] = values_dict[key]['average']
        elif values_dict[key]['average'] > average_dict['max']:
            average_dict['max'] = values_dict[key]['average']
            average_dict['key'] = key
    return average_dict


def print_weekday_analysis(weekday_values):
    print()
    print('{:9}\t{:6}\t{:6}\t{:6}\t{:6}'.format('Weekday', 'min', 'max', 'average', 'median'))
    for day in weekday_values:
        print('{weekday:9}\t{min:6.2f}\t{max:6.2f}\t{average:6.2f}\t{median:6.2f}'.format(
            weekday=day, **weekday_values[day]))

    max_weekday_avg = get_max_average(weekday_values)
    print('{key} has the highest average of {max}'.format(**max_weekday_avg))


def print_month_analysis(month_values):
    print()
    print('{:9}\t{:6}\t{:6}\t{:6}\t{:6}'.format('Month', 'min', 'max', 'average', 'median'))
    for month in month_values:
        print('{month:9}\t{min:6.2f}\t{max:6.2f}\t{average:6.2f}\t{median:6.2f}'.format(
            month=month, **month_values[month]))

    max_month_avg = get_max_average(month_values)
    print('{key} has the highest average of {max}'.format(**max_month_avg))


def get_runs(data_table):
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
    greater_than_w = [run for run in runs if len(run) >= w]
    run_end_indexes = [run[w - 1::w] for run in greater_than_w]
    return [item for sublist in run_end_indexes for item in sublist]


def get_trades(runs, data_table):
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
    return len(trades)


def get_num_profitable_trades(trades):
    return len([trade for trade in trades if trade >= 0])


def get_num_losing_trades(trades):
    return len([trade for trade in trades if trade < 0])


def get_profit_per_profitable_trade(trades):
    return statistics.mean([trade for trade in trades if trade >= 0])


def get_loss_per_losing_trade(trades):
    return statistics.mean([trade for trade in trades if trade < 0])


def print_trade_analysis(trades):
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
    data_table = get_data_table()

    # data broken out by weekday
    weekday_dict = construct_returns_dict(data_table, 'Weekday')
    weekday_values = construct_values_dict(weekday_dict)
    print_weekday_analysis(weekday_values)

    # data broken out by month
    month_dict = construct_returns_dict(data_table, 'Month', lambda idx: calendar.month_name[int(idx)])
    month_values = construct_values_dict(month_dict)
    print_month_analysis(month_values)

    # data broken out by trading strategy one
    runs = get_runs(data_table)
    trades = get_trades(runs, data_table)
    print_trade_analysis(trades)

    # data broken out by trading strategy two
    trades = get_trades_strategy_two(data_table)
    print_strategy_two_trades(trades)


if __name__ == '__main__':
    main()
