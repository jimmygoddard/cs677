import os
import statistics
import calendar

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

# construct a dictionary of weekday name to list of return values
weekday_dict = {}
weekday_idx = None
return_idx = None
for i, line in enumerate(lines):
    if i == 0:
        weekday_idx = line.index('Weekday')
        return_idx = line.index('Return')
        continue
    weekday = line[weekday_idx]
    return_value = float(line[return_idx])
    if weekday in weekday_dict:
        weekday_dict[weekday].append(return_value)
    else:
        weekday_dict[weekday] = [return_value]

# construct a dictionary of weekday name to dictionary of min, max, average, and median
weekday_values = {}
for day in weekday_dict:
    returns = weekday_dict[day]
    weekday_values[day] = {
        'min': min(returns),
        'max': max(returns),
        'average': statistics.mean(returns),
        'median': statistics.median(returns)
    }

print('{:9}\t{:6}\t{:6}\t{:6}\t{:6}'.format('Weekday', 'min', 'max', 'average', 'median'))
for day in weekday_values:
    print('{weekday:9}\t{min:.2f}\t{max:.2f}\t{average:.2f}\t{median:.2f}'.format(weekday=day, **weekday_values[day]))

max_average = {'weekday': None, 'max': -10e25}
for day in weekday_values:
    if weekday_values[day]['average'] > max_average['max']:
        max_average['max'] = weekday_values[day]['average']
        max_average['weekday'] = day

print('{weekday} has the highest average of {max}'.format(**max_average))


# construct a dictionary of month name to list of return values
month_dict = {}
month_idx = None
return_idx = None
for i, line in enumerate(lines):
    if i == 0:
        month_idx = line.index('Month')
        return_idx = line.index('Return')
        continue
    month = calendar.month_name[int(line[month_idx])]
    return_value = float(line[return_idx])
    if month in month_dict:
        month_dict[month].append(return_value)
    else:
        month_dict[month] = [return_value]

