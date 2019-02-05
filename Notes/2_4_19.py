import os
import pandas as pd
from orig_stock_data import get_stock


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

df = get_stock(ticker, start_date, end_date, s_window, l_window)

df_pos = df[df['Return'] > 0]
avg = df['Return'].mean()
std_dev = df['Return'].std()

lower_bound, upper_bound = (avg - 2 * std_dev, avg + 2 * std_dev)

# test for normal dist
# count number of days fall outside of 2*std_dev (on both sides of the dist)

# task 1 show taht your stock follows or does not follow normal distribution
# compute mean and sd and compute number of days outside +/- 2 * sd

df_left_tails = df[df['Return'] < lower_bound]
# 22
df_right_tails = df[df['Return'] > upper_bound]
# 24

# compare this with the expected 5% predicted by normal distribution


# Task 2: analyze the distribution of the last digit (cent position) in the opening price
# we are assuming that all digits are equally likely to appear in the last position (cent position)
# compute distribution of frequency of digits and compare it against the 10% rule

# generate a vector with frequencies of the last digit of the opening price for your stock
# this will be vector A
# A = (a0, ..., a9) where a0 is the frequency of days that end in zero
# P = (.1, .1, ...., .1) 10% expected for totally random digit occurrence
# calculate four error metrics for the above data
# 1. max absolute error
# 2. median absolute error
# 3. mean absolute error
# 4. root mean squared error



# Homework assignment
# you can use whatever you want (numpy, pandas, etc)
# first: empirically show whether return dist is normal or not normal. compute mean, sd, and # of days in the tails
# which then get compared against the expected number of observations in the tails
#
# second: do the error analysis for the occurrence of last digit of opening price
#
# third assignment: transactions from a bakery kaggle dataset
# consider the transactions from a bakery dataset
# let us add one more time description to the dataset:
# 12am to 6am -> night
# 6am to 12pm -> morning
# 12pm to 6pm -> afternoon
# 6pm to 12am -> evening

# what is the busiest (based on number of transactions):
# 1. time of day (hour)
# 2. day of week
# 3. period of the day (morning, evening, whatever)

# what is the most profitable time:
# 1. hour
# 2. day of week
# 3. period of day

# what is the most popular item?
# what is the least popular item?
# what combination of 2 items are most popular (assuming the items are purchased in the same transaction?
# least popular

# assume that one barista can handle 60 transactions per day
# how may barista professionals do we need to hire for each day of the week

# divide all items into three groups
# drinks, food, and unknown

# what is the average price of a drink
# what is the average price of a food item

# does this coffee shop make most money from drinks or food?

# take the top 5 items by popularity
# when (day of week, hour, time of day)

# take the bottom 5 items by popularity
# when (day of week, hour, time of day)

# estimate the group size.  assume that everyone in a group will
# purchase a drink in a single transaction.  take the average number
# of drinks per transaction.  round the resulting number

# to get cent digit:

open_price = 34.349998
last_digit = str(open_price).split('.')[1][1]
# or
last_digit = str(round(open_price, 2))[-1]
df['last_digit'] = df['Open'].apply(lambda x: str(round(x, 2))[-1])

df_1 = df.groupby(['last_digit'])['count'].sum()

# how do you compute the error of your data
P1 = (30, 35, 38, 28, 18)  # one prediction
A = (55, 40, 31, 27, 16)  # actual
P2 = (16, 27, 31, 2, 10)  # another prediction


# two techniques
# absolute error = |A - P| = 10
# relative error = |(A - P) / A|

# compute average error between p1 and a1
# (|a1 - p1| + |a2 - p2\ + |a3 - p3| ...) / len(P1)

# ex
A = (100, 120)
P1 = (90, 150)
P2 = (95, 105)

P1_error = (10, 30)
P1_error_avg = 20

P2_error = (5, 15)
P2_error_avg = 10

# other way to do this?
# can do median error (calculate errors and just find the median)

# you can also use correlation as a calculation of distance between two data sets and compare absolute correlations
# to compare to sets of predictions

# average relative error

# Rel_err(A, P1) = (10%, 25%) -> 17.5%
# Rel_err(A, P2) = (5%, 12.5%) -> 8.75%

# maybe take the maximum absolute error
# or take maximum relative error

# What is the most common error metric that we use?
# root mean squared error

# sqrt(((a1 - p1)**2 + ... + (an - pn)**2) / n)

# read bakery set:
input_file = '/home/jgoddard/src/git/CS677/datasets/BreadBasket_DMS_output.csv'
df = pd.read_csv(input_file)

