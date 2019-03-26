"""
Lecture 8
Quiz #3 this weekend
April 29 -- final presenations (5 - 7 minutes each)
April 29 - May 6 Final Exam (online)

Clustering (k-means), logistic regression, decision trees, and support vector machines

Naive Bayesian

P(X) = P(X|Y) * P(Y)
P(X|Y) = P(Y|X) * P(X) / P(Y)

P('red'|x*) = P(x*|'red') * P('red') / P(x*)

# the following is possible because naive bayesian assume that all of the features are independent
P(x*|'red') = P(x1|'red') * ... * P(xn|'red')


############# Assigment
Train naive bayesian on 2017 of our stocks dataset
Use naive bayesian to predict labels of 2018 of our stocks dataset
see slides naive bayesian page 19
make a comparison to our best k in our kNN

To make the comparison, make the following table:
kNN - manhattan  # pick the best k that we have
kNN - p = 1.5 # pick the best k that we have
kNN - euclidean # pick the k that we have
naive bayesian

two columns:
method, accuracy (1 - error rate)

can use sklearn for both bayesian and knn
info on using sklearn with different distance metrics:
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html
https://stats.stackexchange.com/questions/186833/find-k-nearest-neighbour-with-custom-distance-metric

#### Assignment part 2
Use Linear Regression to do classification

Take a window of 10 days
P1, P2, P3, ..., P10  # closing prices
use linear regression to estimate price on day 11
Let P* be the estimated price
If P* > P10, then tomorrow return > 0
If P* < P10, then tomorrow return < 0

Then compare the results with the real returns
In this way we generate estimates of daily returns for 2018
start with the last 10 days of 2017

Accuracy is determined by how many times your estimate of whether market is going up or down is actually the case
Compare with real daily returns and compute the accuracy
so, if P* > P10 we expect P11 return to be positive (return > 0)
if P* < P10 we expect P11 return to be negative (return < 0)

Do this for W = 10, 20, 30

We can use our choice of polyfit or linear regression

Use an "autocorrelation" function to see if a linear regression would be useless
If as n gets larger the autocorrelation functions approaches and stays at 0 then linear regression will not be useful
If at some point, there is a large spike at some value of n, there is some periodicity in the data

################### Manual Task
Prepare the following dataset
have a set of pairs (x, y)
split the dataset by y = x
points above y = x are green
points below y = x are red
x and y are mean and std dev (so our stocks data set)
we can do it manually if we want
or we can do it automatically (like I said above about y = x)
Do this for all of the points in our dataset

might want to plot the red and green scatter plot to see if there's an obvious line
walk through the data, if y > x and data is red, remove it
if y < x and the data is green, remove it

can choose any line.  x = 2, y = 5, y = 2x, etc.
"""

import numpy as np

# example of linear regressions
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([])

degree = 1
weights = np.polyfit(x, y, degree)
model = np.poly1d(weights)
predicted = model(11)

if predicted > y[-1]:
    rate = 1
else:
    rate = -1
