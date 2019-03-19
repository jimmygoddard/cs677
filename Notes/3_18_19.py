"""
===================================================================
Assignments
===================================================================

Let r be the percent difference between today's open and yesterday's close


All long positions here:
r > [0, 1, 2, 3, 4, 5]
r > 0 buy at open, sell at close
r > 1 buy only if open is higher than yesterday close by at least 1%
r > n buy only if open is higher than yesterday close by at least n%

For each of these values, indicate average pnl

Also all short positions:
r < [0, -1, -2, -3, -4, -5]
r < 0 sell at open, buy at close (short)
r < n but only if open is higher than yesterday close by at least -n%

graph:
r on x axis
% pnl on y axis

also, want to indicate what percentage of total trades for an R were profitable
graph:
r on x axis
% of positive trades on y axis


for r = -5, -4, -3, -2, -1, 1, 2, 3, 4, 5
first graph: r on x-axis, pnl on y-axis
second graph: r on x-axis, % profitable trades on y-axis

write two or three sentences -- what do you see from these graphs?
"What is, if any, a good trading strategy?"

Try to apply the Homework 6 inertia trading strategy (and its reverse) to two or three other years (not 2008)
See if there are other trends

Assignment do KNN "manually"
For every week in 2018 (x and y are avg return and avg volatility)
- calculate distance from one point to every other point and sort that remaining list increasing by distance
- pick up the first 3 (k = 3) and find the majority label on those three and that's the classification
Ex:
  ['red', 'red', 'green'].count('red')
  # 2

Note:
    don't both with scaling

Run the manual knn for 3 distance metrics:
1. Euclidean p = 2
2. Manhattan (street) p = 1
3. Minkowski p = 1.5

LOOK IN DATA_SCIENCE_MODULE_NUMPY.pdf (saved in drive), page 27 - 30.  calculations using numpy are on page 30

ex:
W = np.array([10, 20])
U = np.array([4, 1])
V = W - U

euclidean = np.linalg.norm(V, ord=2)
manhattan = np.linalg.norm(V, ord=1)
minkowski = np.linalg.norm(V, ord=1.5)

euclidean should basically give the same result as we got before using the sklearn knn library

write two or three lines describing if one of the distance metrics gave us better accuracy

graph each of the results
Do it for the same values of k that we used for Homework 5

Assignment summary:
1) Finish labeling
2) Day trading
3) Implement kNN for 3 distances without any sklearn (or any library)

Next week:
use naive bayesian to do classification on labeled stocks data trained on 2017 with testing on 2018
compare the results of naive bayesian vs the knn

===================================================================
Naive Bayesian Classifiers
===================================================================

dataset with height, weight, foot, age, income, label

x = (5, 150, 11, 30, 40)

compute both
P(label = 0 | h = 5, w = 150 ...)
P(label = 1 | h = 5, w = 150 ...)

choose the one that is higher

No need to do scaling for Naive Bayesian because it does not use distances


"""

import numpy as np

# distance examples
W = np.array([10, 20])
U = np.array([4, 1])
V = W - U

euclidean = np.linalg.norm(V, ord=2)
manhattan = np.linalg.norm(V, ord=1)
minkowski = np.linalg.norm(V, ord=1.5)


# Scaling manually
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
X.reshape([5, 2])
# Out[14]:
# array([[ 1,  2],
#        [ 3,  4],
#        [ 5,  6],
#        [ 7,  8],
#        [ 9, 10]])
Y = X.reshape(5, 2)
(Y - np.mean(Y, axis=0)) / np.std(Y, axis=0)
# Out[30]:
# array([[-1.41421356, -1.41421356],
#        [-0.70710678, -0.70710678],
#        [ 0.        ,  0.        ],
#        [ 0.70710678,  0.70710678],
#        [ 1.41421356,  1.41421356]])

