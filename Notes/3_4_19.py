"""
Day trading strategies

Two different graphing packages: matplotlib and seaborn

Day trading strategy:
% range column: overnight percent change in the stock price
Thursday night (close) to open on Friday
Is there a strong positive correlation between your range and return (?)

if the overnight is up, i will buy the stock because i think that it will continue to go up

then we will graph it

overnight change on the x axis
daily percent change on the y axis

we will trade "in both directions".  sell short means we think that the stock price will do down.

We will take "long positions" where we hope the price goes up
and "short positions" where we think the price goes down

Strategy:
for every day, you have:
Open, High, Low, Adj Close

You have daily returns
We have to do:
1) Compute an overnight return:  (open - previous close) / previous close
2) We have daily returns

Trading strategy:
Assumption: Overnight return has "inertia" for the whole day

Example:
         open_price       close_price    overnight_gain    decision                                     PnL
Mon      100                 100
Tue      110                  95          10%              BUY (believe there's daily inertia)          -$15 (made a mistake)
Wed       92                  90          -3%              SELL SHORT (assume price keeps doing down)    $2
Thur      89                  85          -x%              SELL SHORT                                    $4
Fri       90                  95          +x%              BUY                                           $5


Profit calculation:
Long position: buy at open, sell at close
  Profit: (close - open) * num_shares

Short position: sell short ("short" because you don't own the stock) at open, buy at close
  Profit: (open - close) * num_shares
  num_shares on wed: 100/92 = 1.08696 shares
  profit on wed: (92 - 90) * num_shares = 2.17391

If open price == close of previous day, you do nothing

Parameters:
For each trade you are given $100
num_shares = 100/open_price

What to do with strategy (only for 2018):
a. implement without any preliminary analysis (might generate a lot of losses, hard to say)
b. examine your data and modify your algorithm -- maybe only buy if the gain is larger/smaller than some threshold (like
it gained or lost more than 5% of its value [see below], or maybe some multiple of standard deviation or rolling
standard deviation)
c. plot the following result:

suppose you are willing to trade only if overnight change is at least some value, r-min.  suppose r-min is 1%...

for r-min = [-10, -9, ..., -2, -1, 0, 1, ..., 10] compute your average profit and plot it (zero should contain no data
points, probably)

Analyze the results and write up your conclusions

################################################################
Examine Tips dataset

How can we answer questions about whether or not smokers and non smokers tip more?

Ten questions:
1) Are tips higher as a percentage of price for lunch or dinner?
2) When are tips higher?  Which day and time?
  a) Not only choose day and time, but also want to choose gender of payer and whether or not they smoke
3) is there any relationship between price and tip percentage?
4) Any relationship between tip percentage and size of the group?
5) What percentage of people are smoking?
6) Assume that rows are arranged in time, do tips increase with time (you can assign an index to each row in a certain
day and plot them on the x axis as time data.  try to account for that days repeat)
7) Any correlation between gender and time (assume that each meal is split into two equal times)
8) Correlation between tip amounts from smokers and non smokers
9) Average tip for each day of the week
10) Which gender smokes more?

Assignment 7 (after our break)
Do knn ourselves (rather than using a library)
Calculate a distance (three distances: euclidean, manhattan, minkowski)
Euclid p = 2
Man p = 1
Mink p = 1.5

Use 2017 to train
Use 2018 for test
"""

import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

tips = sns.load_dataset('tips')

set(tips['day'])  # unique values in day column
tips['total_bill'].max()  # or min

x = tips.copy()
x['percent_tip'] = 100 * x['tip'] / x['total_bill']
x['percent_tip'] = x['percent_tip'].round(2)

# do smokers or non smokers tip more?
y = x.groupby(['smoker'])['percent_tip'].mean()
z = y.reset_index()

# do women tip more than men?
y_gender_tip = x.groupby(['sex'])['percent_tip'].mean()
z_gender_tip = y_gender_tip.reset_index()

y_gender_smoke_tip = x.groupby(['sex', 'smoker'])['percent_tip'].mean()
z_gender_smoke_tip = y_gender_smoke_tip.reset_index()

y_day_time_tip = x.groupby(['day', 'time'])['percent_tip'].mean()
z_day_time_tip = y_day_time_tip.reset_index()

y_day_time_gender_tip = x.groupby(['day', 'time', 'sex']).mean()

# distribution of total bill prices
fig = plt.figure()
axes1 = fig.add_subplot(1, 1, 1)
axes1.hist(tips['total_bill'], bins=30, color='red')
axes1.set_title('Histogram of bill')
axes1.set_xlabel('Frequency')
axes1.set_ylabel('Total Bill')
fig.show()


hist, ax = plt.subplots()
ax = sns.distplot(tips['total_bill'])
ax.set_title('Bill with Density')
plt.show()

# without density line
hist, ax = plt.subplots()
ax = sns.distplot(tips['total_bill'], kde=False, color='magenta')
ax.set_title('Bill with Density')
plt.show()

# without histogram
hist, ax = plt.subplots()
ax = sns.distplot(tips['total_bill'], hist=False, color='black')
ax.set_title('Bill with Density')
plt.show()


# pairwise relationships
fig = sns.pairplot(tips)
plt.show()

