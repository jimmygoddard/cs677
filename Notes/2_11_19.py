from random import random

import numpy as np

"""
homework due next tuesday rather than monday

online quiz this weekend.  10 Q's, multiple choice, 30 minutes enforced


back to the stock market.  this is with regards to our CSV file for our stock ticker


*****Assignment 1: label each week as follows:
+1 means that it was a good week to be in the market to be invested in your stock
0 means that it is better to be in cash

look at dataset (manually), split it into weeks.  look at adjust cl price from one friday and then adj_close on the 
friday after that.  come up with some rules and describe the rules.  "if the adj_close on the end is above one 
percentage point positive then it's a positive week"

maybe look at the return as well as the variance on that week and use variance to influence your decision. high variance
is not a good week.

similarly have a rule to determine what a bad week means

Just do this for 2018 (manually).  we'll use this dataset to train on some machine learning
Or do it for the entire 5 years of the dataset

No need to submit this, but we will be using that labeling in subsequent weeks

******Assignment 2:
50-day moving average
consecutive W down days

in both strategies we had a parameter w.  in one case w = 50 in another case w was 1, 2, 3, 4, 5
this is called a "hyperparameter"
we will need to investigate the best hyperparameters to use

Implement 2 strategies:
Bollinger Band trading strategy
calculate both the moving average as well as the standard deviation over the last w (50) days (moving average)
then, you do +/- 2 * sd on the moving average for every day which creates a band
if the price is more than moving average + 2 * sd then you buy
if the price is less than the moving average - 2 * sd then you sell

suppose w = 20
for each day, compute 20-day MA and 20-day sd
buy if adj_close (aka "price") < ma - 2*sd
sell if: 
  (a) you own stock, and
  (b) adj_close > ma + 2*sd
  
do this year by year, not all together
  
there's another strategy with the bands which reverses the above

hyperparameters here are: w, and multiple of sd (should it be 1, 1.5, 2, etc)
ma(w) +/- k * sd(w)

which combination of w and k give us the best result.  "best result" is ...

assumptions:
each time you trade, MET college gives us $100

compute average gain/loss per trade (average percentage loss or gain per trade)

write a function "trade_bollinger" which takes three parameters: year, w, and k:

trade_bollinger(year, window_size, k) => average gain/loss per trade

look at w from 10 to 100 in increments of 10 (10 points there)
look at k as 0.5, 1, 1.5, 2, 2.5, 3 (6 points)
for every one of the 60 points, draw a circle. red is loss, green is gain.  size of point is related to magnitude of 
loss or gain

take w to be range(10, 110, 10): [10, 20, 30, ..., 100]
or range(10, 101, 10)

take k to be [x * 0.1 for x in range(5, 31, 5)]: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

*******Assignment 3:
take two windows, w_short and w_long, compute moving averages on both

if ma(short window) > ma(long window) then buy
if ma(short window) < ma(long window) and you own stock then sell
short has to be smaller than long
do same as assignment 2 with graphing the short and long values on y and x axis and the data points are the
loss/gain with size of dot related to the magnitude of the loss/gain
if done right, all points should be below the diagonal

take each w to be range(10, 101, 10)
do this one for each year just like assignment 2

write func:  trade_strategy_two(year, small_window_size, large_window_size) => avg gain/loss per trade
"""

n = 1000
x_list = [random() for i in range(n)]
y_list = [random() for i in range(n)]
z_list = [x_list[i] + y_list[i] for i in range(n)]

x_vector = np.array(x_list)
y_vector = np.array(y_list)
z_vector = x_vector + y_vector  # much faster


x = np.array(range(10))
y = x[2:7]  # this is a "view", not a copy
y[2] = 44  # this also affects x
print(x)
print(y)
# [ 0  1  2  3 44  5  6  7  8  9]
# [ 2  3 44  5  6]

y = np.copy(x)  # this gives you a copy
y[2] = 44  # this also affects x
print(x)
print(y)
# [ 0  1  2  3 44  5  6  7  8  9]
# [ 0  1 44  3 44  5  6  7  8  9]

# trimmed mean:  remove max, remove min and compute mean over the remaining elements
x = np.array([2, 3, -10, 10, 7])
x.mean()
# 2.4
y = np.array([2, 3, 7])
y.mean()  # this would be the trimmed mean of x
# 4.0

np.delete(x, [x.argmin(), x.argmax()]).mean()  # another trimmed mean
# you can extend trimmed mean to percentiles -- so remove the top 5% and the bottom 5%

x_points = np.arange(-2, 3, 1)
y_points = np.arange(-1, 2, 1)

xs, ys = np.meshgrid(x_points, y_points)
z = np.round(np.sqrt(xs**2, ys**2), 2)  # this is in example, but doesn't work

y = np.arange(1, 37).reshape(6, 6)
# array([[ 1,  2,  3,  4,  5,  6],
#        [ 7,  8,  9, 10, 11, 12],
#        [13, 14, 15, 16, 17, 18],
#        [19, 20, 21, 22, 23, 24],
#        [25, 26, 27, 28, 29, 30],
#        [31, 32, 33, 34, 35, 36]])

print(y[1, 2])
print(y[1][2])
print(y[2, :])  # third row
print(y[2, ])   # same thing
print(y[:, 2])  # third column

print(y[[1, 2, 5], ])   # prints second, third, and 6th row
print(np.sum(y[[1, 2, 5], ], axis=1))  # add each of the three rows separately
# out: array([ 57,  93, 201])

print(y[1::2, ])  # print every other row starting with the second row
print(y[::2, ::3])  # every second row, every third column
print(y[::-1, ::-1])  # everything in reverse
print(y[:, 4])
# out: [ 5 11 17 23 29 35]
print(y[:, [4]])
# [[ 5]
#  [11]
#  [17]
#  [23]
#  [29]
#  [35]]

# add the 1x1 and then the 2x2 and then the 3x3...
for n in range(1, y.shape[0] + 1):
    print(np.sum(y[:n, :n]))

# A matrix is symmetric if Aij = Aji (basically if A == np.transpose(A)
# check if one matrix is symmetric?
z = abs(y - y.transpose())
# z is symmetric
z - z.transpose() == 0
# all true
# but still not great without visually inspecting it
z - z.transpose()   # will give all zeros
