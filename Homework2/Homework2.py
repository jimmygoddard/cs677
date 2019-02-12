import math
import os
import random
import statistics

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stock_data import get_stock

# home_dir = os.path.join('C:', os.path.sep, 'Users', 'jimmy_000')  # MS Windows home directory
home_dir = os.path.join(os.path.sep + 'home', 'jgoddard')  # Linux home directory
input_dir = os.path.join(home_dir, 'src', 'git', 'CS677', 'datasets')


# From the transactions_from_bakery.py provided by Professor Pinsky
def compute_period(hour):
    if hour in range(0, 6):
        return 'night'
    elif hour in range(6, 12):
        return 'morning'
    elif hour in range(12, 18):
        return 'afternoon'
    elif hour in range(18, 24):
        return 'evening'
    else:
        return 'unknown'


# From the transactions_from_bakery.py provided by Professor Pinsky
def get_bakery_df():
    input_file = os.path.join(input_dir, 'BreadBasket_DMS.csv')
    output_file = os.path.join(input_dir, 'BreadBasket_DMS_output.csv')

    df = pd.read_csv(input_file)

    items = set(df['Item'])
    price_dict = dict()
    price_list = list(np.linspace(0.99, 10.99, 100))
    for x in items:
        price_dict[x] = random.choice(price_list)

    df['Item_Price'] = df['Item'].map(price_dict)
    df['Item_Price'] = df['Item_Price'].round(2)

    df['Date'] = pd.to_datetime(df['Date'])
    df['Time'] = pd.to_datetime(df['Time'])
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Day'] = df['Date'].dt.day
    df['Weekday'] = df['Date'].dt.weekday_name

    df['Hour'] = df['Time'].dt.hour
    df['Min'] = df['Time'].dt.minute
    df['Sec'] = df['Time'].dt.second

    df['Period'] = df['Hour'].apply(compute_period)

    df.reset_index(level=0)
    col_list = ['Year', 'Month', 'Day', 'Weekday', 'Period', 'Hour', 'Min', 'Sec', 'Transaction', 'Item', 'Item_Price']
    df = df[col_list]

    df.to_csv(output_file, index=False)
    return df


def get_stocks_df():
    # ticker = 'GS'  # Goldman Sachs Group Inc
    # ticker = 'GDDY'  # GoDaddy
    # ticker = 'GM'  # General Motors
    ticker = 'GRUB'  # GrubHub
    start_date = '2014-01-01'
    end_date = '2018-12-31'
    s_window = 14
    l_window = 50
    output_file = os.path.join(input_dir, ticker + '.csv')

    if os.path.isfile(output_file):
        return pd.read_csv(output_file)

    df = get_stock(ticker, start_date, end_date, s_window, l_window)
    df.to_csv(output_file, index=False)
    return df


# Question 1
stocks_df = get_stocks_df()
df_pos = stocks_df[stocks_df['Return'] > 0]
avg = stocks_df['Return'].mean()
sd = stocks_df['Return'].std()

lower_bound, upper_bound = (avg - 2 * sd, avg + 2 * sd)
df_left_tails = stocks_df[stocks_df['Return'] < lower_bound]
print(len(df_left_tails))
# 22
df_right_tails = stocks_df[stocks_df['Return'] > upper_bound]
print(len(df_right_tails))
# 24
expected_tail_size = (len(stocks_df) * .05) / 2
print(expected_tail_size)

n, bins, patches = plt.hist(stocks_df['Return'], 100, density=True, facecolor='g', alpha=0.75)
plt.show()

# Answer to Question 1
# The expected size of each tail is 30 observations based on a mean of 0.001129786860956086 and standard deviation
# of 0.03017623614190026.  The actual number of observations for the tails was 22 on the left side and 24 on the right
# side.  This is just a bit lower than what you would expect from a normal distribution.  The histogram does show a
# distribution that does look very close to normal.  That each actual tail contained nearly the same amount of
# observations and they were both in the ballpark of the expected number of observations makes me conclude that this
# distribution of Returns is quite close to a normal distribution.  Also, the mean and median pretty close together
# too: mean of 0.001129786860956086 and a median of 0.0015156072544540988

# Question 2
stocks_df['last_digit'] = stocks_df['Open'].apply(lambda x: str(round(x, 2))[-1])

counts_df = pd.DataFrame(stocks_df.groupby(['last_digit']).count()['Date'])
counts_df = counts_df.rename_axis({'Date': 'Count'}, axis='columns')
print(counts_df)
#             Count
# last_digit
# 0              65
# 1             117
# 2             109
# 3             109
# 4             103
# 5             168
# 6             124
# 7             128
# 8             138
# 9             133

counts_df['Freq'] = counts_df['Count'] / counts_df['Count'].sum()
A = counts_df['Freq'].to_numpy()
P = np.array([0.1] * 10)


def calculate_abs_error(actual, predicted):
    return abs(actual - predicted)


def calculate_rmse(actual, predicted):
    return math.sqrt(((actual - predicted)**2).mean())


def calculate_rel_error(actual, predicted):
    return abs((actual - predicted) / actual)


# Answers for Question 2
abs_error = calculate_abs_error(A, P)
print('Max absolute error: {}'.format(max(abs_error)))
print('Median absolute error: {}'.format(statistics.median(abs_error)))
print('Mean absolute error: {}'.format(statistics.mean(abs_error)))
print('Root mean squared error: {}'.format(calculate_rmse(A, P)))

# Question 3
bakery_df = get_bakery_df()
transactions_by_hour = bakery_df['Transaction'].groupby(bakery_df['Hour'])
print(transactions_by_hour.count())
plt.plot(transactions_by_hour.count())
plt.show()
# The busiest hour of the day is hour 12, noon, with 3021 unique transactions occurring at noon


transactions_by_weekday = bakery_df['Transaction'].groupby(bakery_df['Weekday'])
print(transactions_by_weekday.count())
plt.plot(transactions_by_weekday.count())
plt.show()
# The busiest day of the week is Saturday with 4803 transactions

transactions_by_period = bakery_df['Transaction'].groupby(bakery_df['Period'])
print(transactions_by_period.count())
plt.plot(transactions_by_period.count())
plt.show()
# The busiest period of the day is afternoon with 12408 unique transactions

# The most profitable hour of the day is 11am with $12,517.81 in sales
print(bakery_df['Item_Price'].groupby(bakery_df['Hour']).sum())
# Hour
# 1         1.29
# 7        93.81
# 8      2390.47
# 9      7933.33
# 10    10709.55
# 11    12517.81
# 12    12227.23
# 13    11773.70
# 14    11630.25
# 15     8604.45
# 16     5418.95
# 17     1770.56
# 18      416.21
# 19      247.28
# 20       99.36
# 21       11.35
# 22       62.19
# 23       11.77

# The most profitable Weekday is Saturday with $20,352.97 in sales
print(bakery_df['Item_Price'].groupby(bakery_df['Weekday']).sum())
# Weekday
# Friday       13253.25
# Monday        9417.48
# Saturday     20352.52
# Sunday       13089.97
# Thursday     10679.90
# Tuesday       9712.18
# Wednesday     9414.26

# The most profitable period of the day is afternoon with $51,425.15 in sales
print(bakery_df['Item_Price'].groupby(bakery_df['Period']).sum())
# Period
# afternoon    51425.14
# evening        848.16
# morning      33644.97
# night            1.29

transactions_by_item = bakery_df['Transaction'].groupby(bakery_df['Item'])
print(transactions_by_item.count().sort_values(ascending=False).head(1))
# The most popular item is Coffee which was involved in 5471 transactions

s = transactions_by_item.count().sort_values()
print(s[s == 1])
# The least popular items were:
# Item
# Adjustment        1
# Bacon             1
# Chicken sand      1
# Gift voucher      1
items_per_transaction = bakery_df['Item'].groupby(bakery_df['Transaction'])
two_item_transactions = items_per_transaction.count()[items_per_transaction.count() == 2].index.values
grouping = bakery_df[bakery_df['Transaction'].isin(two_item_transactions)].groupby(['Transaction'])
two_item_groups = [', '.join(list(group['Item'])) for name, group in grouping]
two_item_groups_df = pd.DataFrame({'Items': two_item_groups})
unique, counts = np.unique(two_item_groups_df['Items'], return_counts=True)
two_items = pd.Series(counts, index=unique)

# Coffee with Coffee is the most popular two item transaction
print(two_items.sort_values(ascending=False).head())
# Coffee, Coffee    163
# Coffee, Bread     129
# Coffee, Pastry     85
# Coffee, Cake       84

# There are many two item transactions that were the least popular (with 1 occurrence)
print(two_items.sort_values().head())
# Afternoon with the baker, Afternoon with the baker    1
# Medialuna, Dulce de Leche                             1
# Medialuna, Crepes                                     1
# Medialuna, Cake                                       1
print(len(two_items[two_items == 1]))
# 286

# We will need the following numbers of baristas on each day of the week assuming each barista can handle 60
# transactions per hour
print(np.ceil(bakery_df['Transaction'].groupby(bakery_df['Hour']).count() / 60))
# Hour
# 1      1.0
# 7      1.0
# 8     12.0
# 9     35.0
# 10    47.0
# 11    54.0
# 12    51.0
# 13    47.0
# 14    46.0
# 15    36.0
# 16    23.0
# 17     7.0
# 18     2.0
# 19     1.0
# 20     1.0
# 21     1.0
# 22     1.0
# 23     1.0

drinks_df = bakery_df[bakery_df['Type'] == 'Drink']
print(np.ceil(drinks_df['Type'].groupby(drinks_df['Hour']).count() / 60))
# Hour
# 7      1.0
# 8      4.0
# 9     14.0
# 10    19.0
# 11    22.0
# 12    19.0

# I used the following loop to interactively construct a mapping of food to type
# item_to_type = {}
# types = ['Food', 'Drink', 'Unknown']
# for item in bakery_df['Item'].unique():
#     for i, item_type in enumerate(types):
#         print('{}. {}'.format(i + 1, item_type))
#     index = input('What type of item is: {}'.format(item))
#     item_to_type[item] = types[int(index) - 1]

item_to_type = {
    'Bread': 'Food',
    'Scandinavian': 'Unknown',
    'Hot chocolate': 'Drink',
    'Jam': 'Food',
    'Cookies': 'Food',
    'Muffin': 'Food',
    'Coffee': 'Drink',
    'Pastry': 'Food',
    'Medialuna': 'Unknown',
    'Tea': 'Drink',
    'NONE': 'Unknown',
    'Tartine': 'Unknown',
    'Basket': 'Unknown',
    'Mineral water': 'Drink',
    'Farm House': 'Food',
    'Fudge': 'Food',
    'Juice': 'Drink',
    "Ella's Kitchen Pouches": 'Unknown',
    'Victorian Sponge': 'Food',
    'Frittata': 'Food',
    'Hearty & Seasonal': 'Food',
    'Soup': 'Food',
    'Pick and Mix Bowls': 'Food',
    'Smoothies': 'Drink',
    'Cake': 'Food',
    'Mighty Protein': 'Food',
    'Chicken sand': 'Food',
    'Coke': 'Drink',
    'My-5 Fruit Shoot': 'Unknown',
    'Focaccia': 'Food',
    'Sandwich': 'Food',
    'Alfajores': 'Unknown',
    'Eggs': 'Food',
    'Brownie': 'Food',
    'Dulce de Leche': 'Unknown',
    'Honey': 'Food',
    'The BART': 'Unknown',
    'Granola': 'Food',
    'Fairy Doors': 'Unknown',
    'Empanadas': 'Food',
    'Keeping It Local': 'Unknown',
    'Art Tray': 'Unknown',
    'Bowl Nic Pitt': 'Food',
    'Bread Pudding': 'Food',
    'Adjustment': 'Unknown',
    'Truffles': 'Food',
    'Chimichurri Oil': 'Food',
    'Bacon': 'Food',
    'Spread': 'Food',
    'Kids biscuit': 'Food',
    'Siblings': 'Unknown',
    'Caramel bites': 'Food',
    'Jammie Dodgers': 'Unknown',
    'Tiffin': 'Unknown',
    'Olum & polenta': 'Food',
    'Polenta': 'Food',
    'The Nomad': 'Unknown',
    'Hack the stack': 'Unknown',
    'Bakewell': 'Unknown',
    'Lemon and coconut': 'Food',
    'Toast': 'Food',
    'Scone': 'Food',
    'Crepes': 'Food',
    'Vegan mincepie': 'Food',
    'Bare Popcorn': 'Food',
    'Muesli': 'Food',
    'Crisps': 'Food',
    'Pintxos': 'Unknown',
    'Gingerbread syrup': 'Food',
    'Panatone': 'Unknown',
    'Brioche and salami': 'Food',
    'Afternoon with the baker': 'Unknown',
    'Salad': 'Food',
    'Chicken Stew': 'Food',
    'Spanish Brunch': 'Food',
    'Raspberry shortbread sandwich': 'Food',
    'Extra Salami or Feta': 'Food',
    'Duck egg': 'Food',
    'Baguette': 'Food',
    "Valentine's card": 'Unknown',
    'Tshirt': 'Unknown',
    'Vegan Feast': 'Food',
    'Postcard': 'Unknown',
    'Nomad bag': 'Unknown',
    'Chocolates': 'Food',
    'Coffee granules ': 'Drink',
    'Drinking chocolate spoons ': 'Unknown',
    'Christmas common': 'Unknown',
    'Argentina Night': 'Unknown',
    'Half slice Monster ': 'Food',
    'Gift voucher': 'Unknown',
    'Cherry me Dried fruit': 'Food',
    'Mortimer': 'Unknown',
    'Raw bars': 'Food',
    'Tacos/Fajita': 'Food'
}

bakery_df['Type'] = bakery_df['Item'].map(item_to_type)

bakery_df['Item_Price'].groupby(bakery_df['Type']).mean()
# Type
# Drink      2.871306
# Food       4.253100
# Unknown    6.699293
# The average price of a drink is $2.87 and the average price of food is $4.25

bakery_df['Item_Price'].groupby(bakery_df['Type']).sum()
# Type
# Drink      23745.70
# Food       43590.02
# Unknown    18583.84
# Food items are the most profitable for the bakery

# Top 5 most popular items in general:
print(bakery_df.groupby(bakery_df['Item']).size().sort_values(ascending=False).head(5))
# Item
# Coffee    5471
# Bread     3325
# Tea       1435
# Cake      1025
# Pastry     856

# Least popular 5 items in general:
print(bakery_df.groupby(bakery_df['Item']).size().sort_values().head(5))
# Item
# Adjustment        1
# Chicken sand      1
# Olum & polenta    1
# Polenta           1
# Bacon             1

# top 5 most popular items per day of week
weekdays = {}
for weekday in bakery_df['Weekday'].unique():
    weekdays[weekday] = bakery_df[bakery_df['Weekday'] == weekday]

# most popular by day:
for weekday in weekdays:
    print(weekday)
    print(weekdays[weekday].groupby('Item').size().sort_values(ascending=False).head())

# Sunday
# Item
# Coffee    825
# Bread     473
# Tea       171
# Cake      167
# NONE      138
#
# Monday
# Item
# Coffee      681
# Bread       360
# Tea         193
# Pastry      105
# Sandwich    101
#
# Tuesday
# Item
# Coffee    710
# Bread     350
# Tea       194
# Cake      139
# Pastry    119
#
# Wednesday
# Item
# Coffee    628
# Bread     405
# Tea       188
# Cake      123
# NONE      108
#
# Thursday
# Item
# Coffee    670
# Bread     450
# Tea       183
# Cake      141
# Pastry    121
#
# Friday
# Item
# Coffee      854
# Bread       527
# Tea         218
# Sandwich    134
# Cake        120
#
# Saturday
# Item
# Coffee    1103
# Bread      760
# Tea        288
# Cake       246
# NONE       198

# least popular by day
for weekday in weekdays:
    print(weekday)
    print(weekdays[weekday].groupby('Item').size().sort_values().head())

# Sunday
# Item
# Coffee granules     1
# Christmas common    1
# Argentina Night     1
# Chocolates          1
# Bacon               1
#
# Monday
# Item
# Dulce de Leche                1
# Crisps                        1
# Pick and Mix Bowls            1
# Mighty Protein                1
# Drinking chocolate spoons     1
#
# Tuesday
# Item
# Honey                  1
# Nomad bag              1
# Kids biscuit           1
# Half slice Monster     1
# Granola                1
#
# Wednesday
# Item
# Adjustment        1
# Vegan Feast       1
# Raw bars          1
# Polenta           1
# Olum & polenta    1
#
# Thursday
# Item
# Lemon and coconut             1
# Argentina Night               1
# Duck egg                      1
# Spread                        1
# Drinking chocolate spoons     1
#
# Friday
# Item
# Honey               1
# Mighty Protein      1
# Panatone            1
# Coffee granules     1
# Fairy Doors         1
#
# Saturday
# Item
# Victorian Sponge          1
# Lemon and coconut         1
# Mortimer                  1
# Fairy Doors               1
# Ella's Kitchen Pouches    1

hours = {}
for hour in bakery_df['Hour'].unique():
    hours[hour] = bakery_df[bakery_df['Hour'] == hour]

# most popular item by hour of day
for hour in hours:
    print(hour)
    print(hours[hour].groupby('Item').size().sort_values(ascending=False).head())

# 9
# Item
# Coffee       583
# Bread        400
# Pastry       191
# Medialuna    120
# Tea          103
#
# 10
# Item
# Coffee       820
# Bread        508
# Pastry       203
# Tea          156
# Medialuna    125
#
# 11
# Item
# Coffee    946
# Bread     528
# Tea       176
# Pastry    151
# Cake      133
#
# 12
# Item
# Coffee      740
# Bread       474
# Tea         183
# NONE        167
# Sandwich    162
#
# 13
# Item
# Coffee      607
# Bread       340
# Sandwich    234
# Tea         181
# NONE        159
#
# 14
# Item
# Coffee      636
# Bread       341
# Tea         233
# Cake        182
# Sandwich    171
#
# 8
# Item
# Coffee       199
# Bread        171
# Pastry        57
# Medialuna     43
# NONE          24
#
# 15
# Item
# Coffee           519
# Bread            310
# Tea              207
# Cake             174
# Hot chocolate     89
#
# 17
# Item
# Coffee     69
# Bread      46
# Tea        41
# Cake       30
# Cookies    20
#
# 18
# Item
# Afternoon with the baker    14
# Coffee                      11
# Bread                        6
# Tea                          5
# Tshirt                       5
#
# 7
# Item
# Coffee       13
# Medialuna     6
# Pastry        2
# Bread         2
# Toast         1
#
# 16
# Item
# Coffee           321
# Bread            196
# Tea              126
# Cake             124
# Hot chocolate     71
#
# 19
# Item
# Tshirt              11
# Coke                 6
# Coffee               6
# Valentine's card     4
# Tea                  3
#
# 20
# Item
# Postcard          7
# Tshirt            5
# Pintxos           4
# Vegan mincepie    1
# Nomad bag         1
#
# 21
# Item
# Hot chocolate    2
# Vegan Feast      1
#
# 1
# Item
# Bread    1
#
# 23
# Item
# Valentine's card    2
# Vegan Feast         1
#
# 22
# Item
# Vegan Feast      5
# Scandinavian     1
# Mineral water    1
# Juice            1

# least popular items by hour of day
for hour in hours:
    print(hour)
    print(hours[hour].groupby('Item').size().sort_values().head())
# 9
# Item
# Victorian Sponge                 1
# Art Tray                         1
# Salad                            1
# Raspberry shortbread sandwich    1
# My-5 Fruit Shoot                 1
#
# 10
# Item
# Half slice Monster            1
# Duck egg                      1
# Drinking chocolate spoons     1
# Lemon and coconut             1
# Gingerbread syrup             1
#
# 11
# Item
# Afternoon with the baker    1
# Vegan Feast                 1
# The BART                    1
# Tacos/Fajita                1
# Panatone                    1
#
# 12
# Item
# Chimichurri Oil               1
# Drinking chocolate spoons     1
# Crepes                        1
# Ella's Kitchen Pouches        1
# Empanadas                     1
#
# 13
# Item
# Victorian Sponge              1
# Chicken sand                  1
# Drinking chocolate spoons     1
# Empanadas                     1
# Bread Pudding                 1
#
# 14
# Item
# Victorian Sponge    1
# Mortimer            1
# My-5 Fruit Shoot    1
# Polenta             1
# Christmas common    1
#
# 8
# Item
# Vegan mincepie       1
# Granola              1
# Gingerbread syrup    1
# Eggs                 1
# Dulce de Leche       1
#
# 15
# Item
# Victorian Sponge        1
# Focaccia                1
# Hearty & Seasonal       1
# Extra Salami or Feta    1
# Eggs                    1
#
# 17
# Item
# Afternoon with the baker    1
# Duck egg                    1
# Hack the stack              1
# Coffee granules             1
# Kids biscuit                1
#
# 18
# Item
# Vegan Feast    1
# Sandwich       1
# Cookies        1
# Empanadas      1
# Focaccia       1
#
# 7
# Item
# NONE         1
# Toast        1
# Bread        2
# Pastry       2
# Medialuna    6
#
# 16
# Item
# Afternoon with the baker         1
# Ella's Kitchen Pouches           1
# Empanadas                        1
# Raspberry shortbread sandwich    1
# Chocolates                       1
#
# 19
# Item
# Adjustment        1
# Truffles          1
# Medialuna         1
# Jammie Dodgers    1
# Fudge             1
#
# 20
# Item
# Coffee            1
# Coke              1
# Dulce de Leche    1
# Fudge             1
# Nomad bag         1
#
# 21
# Item
# Vegan Feast      1
# Hot chocolate    2
#
# 1
# Item
# Bread    1
#
# 23
# Item
# Vegan Feast         1
# Valentine's card    2
#
# 22
# Item
# Juice            1
# Mineral water    1
# Scandinavian     1
# Vegan Feast      5

periods = {}
for period in bakery_df['Period'].unique():
    periods[period] = bakery_df[bakery_df['Period'] == period]

# Top 5 most popular item by period of day
for period in periods:
    print(period)
    print(periods[period].groupby('Item').size().sort_values(ascending=False).head())

# morning
# Item
# Coffee       2561
# Bread        1609
# Pastry        604
# Tea           456
# Medialuna     402
#
# afternoon
# Item
# Coffee      2892
# Bread       1707
# Tea          971
# Cake         761
# Sandwich     675
#
# evening
# Item
# Tshirt                      21
# Coffee                      18
# Afternoon with the baker    14
# Postcard                    10
# Hot chocolate                9
#
# night
# Item
# Bread    1

# 5 least popular items per period of day
for period in periods:
    print(period)
    print(periods[period].groupby('Item').size().sort_values().head())

# morning
# Item
# Half slice Monster     1
# Gift voucher           1
# Lemon and coconut      1
# The BART               1
# Mighty Protein         1
#
# afternoon
# Item
# Polenta                  1
# Cherry me Dried fruit    1
# Brioche and salami       1
# Chicken sand             1
# Raw bars                 1
#
# evening
# Item
# Adjustment    1
# Sandwich      1
# Nomad bag     1
# Brownie       1
# Cookies       1
#
# night
# Item
# Bread    1

drinks_per_transaction = bakery_df[bakery_df['Type'] == 'Drink'].groupby('Transaction').size()
# average group size per transaction
total_drinks_purchased_count = len(bakery_df[bakery_df['Type'] == 'Drink'])
total_transaction_count = len(bakery_df['Transaction'].unique())
avg_group_size = total_drinks_purchased_count / total_transaction_count
print(avg_group_size)
# 0.8676948903577799
