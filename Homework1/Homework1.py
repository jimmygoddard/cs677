import operator
import os
import statistics
import calendar
import itertools

from orig_stock_data import get_stock

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

df = get_stock(ticker, start_date, end_date, s_window, l_window)
df.to_csv(output_file, index=False)


with open(output_file) as in_file:
    # split csv file into a list of lists.  first item in the list are the headers
    lines = [line.split(',') for line in in_file.read().splitlines()]


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


weekday_dict = construct_returns_dict(lines, 'Weekday')
weekday_values = construct_values_dict(weekday_dict)

print('{:9}\t{:6}\t{:6}\t{:6}\t{:6}'.format('Weekday', 'min', 'max', 'average', 'median'))
for day in weekday_values:
    print('{weekday:9}\t{min:.2f}\t{max:.2f}\t{average:.2f}\t{median:.2f}'.format(weekday=day, **weekday_values[day]))


max_weekday_avg = get_max_average(weekday_values)
print('{key} has the highest average of {max}'.format(**max_weekday_avg))

month_dict = construct_returns_dict(lines, 'Month', lambda idx: calendar.month_name[int(idx)])
month_values = construct_values_dict(month_dict)

print()
print('{:9}\t{:6}\t{:6}\t{:6}\t{:6}'.format('Month', 'min', 'max', 'average', 'median'))
for month in month_values:
    print('{month:9}\t{min:.2f}\t{max:.2f}\t{average:.2f}\t{median:.2f}'.format(month=month, **month_values[month]))

max_month_avg = get_max_average(month_values)
print('{key} has the highest average of {max}'.format(**max_month_avg))


return_idx = lines[0].index('Return')
neg_indexes = [i + 1 for i, item in enumerate(lines[1:]) if float(item[return_idx]) < 0]

# From https://stackoverflow.com/questions/2361945/detecting-consecutive-integers-in-a-list
# Example of detecting consecutive integers: https://docs.python.org/2.6/library/itertools.html#examples
# comments on SO post describe adjusting the 2.6 example for Python 3
runs = []
for k, g in itertools.groupby(enumerate(neg_indexes), lambda ix: ix[0] - ix[1]):
     runs.append(list(map(operator.itemgetter(1), g)))


greater_than_three = [run for run in runs if len(run) >= 3]
print(greater_than_three[0])
# Out[48]: [10, 11, 12, 13, 14, 15]
print(greater_than_three[0][2::3])
# Out[55]: [12, 15]

# generalized that will be: greater_than_w[i][w -1::w]
