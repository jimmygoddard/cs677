"""
Homework 5
Need a labeled data set
2017 + 2018 - label every week as "green"
(good week to be invested)
and "red" (better be in cash)

Generate (x, y) points for each week (2017 - 2018 roughly 105 data points)
For week i: xi = np.mean([r1, r2, r3, r4, r5])
yi = np.std([r1, r2, r3, r4, r5])

Why do we need this:
(1) KNN - nearest neighbors
(2) Naive Bayesian

Data Scaling: (already in blackboard)

Numeric Python:
Basic object: np.array([1, 2, 3, 4, 5])
reshape this -> n-dimensional matrices
operations are vectorized
slicing, indexing
slicing gives you a view (not a copy)

in many applications you need to measure similarity (distance) between objects

a distance is a non-negative number
1) D(x, x) = 0
2) Symmetric: D(x, y) = D(y, x)
3) Triangle inequality: D(x, z) <= D(x, y) + D(y, z)

Examples:
    D(x, y) = 0 if x = y and 1 otherwise

We can have both categorical and numerical data.  The above can handle categorical binary data
Say we have two values for our categorical data: "sunny" and "strong".  Encode "sunny" as 0 and "strong" as 1

Distances on vectors
(x1, ..., xn)
(y1, ..., yn)

Suppose each xi and yi are categorical

"Hamming" distance
Lay out each sequence and just count how many positions where they're different

Sets: A = {x, y, z}, B = {y, z, w}, C = {x, z, w, a, b}

Jacard's similarity:
len(A.intersection(B)) / len(A.union(B)  # |A intersection B| / |A union B|
D(A, B) = 2/4 = 0.5
D(A, C) = 2/6 = 1/3

For example, distance = 1 - similarity # not a bad distance there.  would get distance 0 for identical sets

Data Scaling:

MinMax scaling:

xi = (xi - min(X)) / (max(X) - min(X))

Standard scaling (z-scaling, etc.):

xi = (xi - mean(X)) / std(X)

NOTE: Use standard scaling for any method that uses distances!!!!


HOMEWORK

QUIZ available Friday through Monday

Task 1:
Compute the value of k that gives you the best accuracy in a KNN model
Data set is a collection of weekly data, (xi, yi):  (mean, std) for that week plus labels (maybe not x = mean
and y = std)

Take 2 years of data (2017-2018)

Take 2017 as training set
Take 2018 as testing set

X = 104 rows and 3 columns
mean, std, label

52 points on training data
52 points on testing data

take k = [1, 3, 5, 7, 9, 11]

draw the following graph (similar to page 17 for IRIS)

on x you specify k
on y is the accuracy (error rate)

Find out the best k* - what is th minimum k for th lowest error rate

Last part of task 1:
use k* neighbors to predict teh first week on 2019


Naive Bayes

Assignment (not due next week because we have a quiz to do)
2017 -> training data (mean_i, std_i)
use naive bayesian classifier to train on 2017
predict your labels for 2018 and compute the error rate
compare your results with the knn for best "k"


Questions:
Suppose you have two envelopes
one contains 1 red card and 2 blue cards
the second contains 2 red cards and 1 blue card and $5

suppose you draw one card and it is blue
will it (and how) change your estimate on the probably that this was the envelope
containing $5

Before drawing -----> 0.5 (prior knowledge)

Probability that it is the first envelope given that you've selected a blue card

P(envelope_one|blue) = P(blue|envelope_one) * P(red|envelope_one) * P(envelope_one)
= 2/3 * 1/3 * 1/2 = 0.1111111111111111

P(envelope_two|blue) = P(blue|envelope_two) * P(red|envelope_two) * P(envelope_two)
= 1/3 * 2/3 * 1/2 = 0.1111111111111111

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

url = r'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, names=['sepal - length', 'sepal - width', 'petal - length', 'petal - width', 'Class'])

x = np.arange(1, 37).reshape(6, 6)

data_dict = {
    'id': [1, 2, 3, 4, 5, 6, 7, 8],
    'Label': ['green', 'green', 'green', 'green', 'red', 'red', 'red', 'red'],
    'Height': [5, 5.5, 5.33, 5.75, 6.00, 5.92, 5.58, 5.92],
    'Weight': [100, 150, 130, 150, 180, 190, 170, 165],
    'Foot': [6, 8, 7, 9, 13, 11, 12, 10]
}
data = pd.DataFrame(data_dict, columns=list(data_dict.keys()))

data.describe()
# Out[5]:
#             id    Height      Weight      Foot
# count  8.00000  8.000000    8.000000   8.00000
# mean   4.50000  5.625000  154.375000   9.50000
# std    2.44949  0.343428   28.962722   2.44949
# min    1.00000  5.000000  100.000000   6.00000
# 25%    2.75000  5.457500  145.000000   7.75000
# 50%    4.50000  5.665000  157.500000   9.50000
# 75%    6.25000  5.920000  172.500000  11.25000
# max    8.00000  6.000000  190.000000  13.00000

X = data[['Height', 'Weight']].values
Z_min_max = MinMaxScaler().fit_transform(X)
Z_std_scale = StandardScaler().fit_transform(X)
colors = [color[0] for color in data[['Label']].values]
plt.scatter(x=X[:, 0], y=X[:, 1], c=colors)
plt.show()

plt.scatter(x=Z_min_max[:, 0], y=Z_min_max[:, 1], c=colors)
plt.show()

plt.scatter(x=Z_std_scale[:, 0], y=Z_std_scale[:, 1], c=colors)
plt.show()

# Train KNN
data = pd.DataFrame(
    {'id': [1, 2, 3, 4, 5, 6],
     'Label': ['green', 'red', 'red', 'green', 'green', 'red'],
     'X': [1, 6, 7, 10, 10, 15],
     'Y': [2, 4, 5, -1, 2, 2 ]},
    columns=['id', 'Label', 'X', 'Y'])

X = data[['X', 'Y']].values
Y = data[['Label']].values

# SHOULD BE SCALING HERE BEFORE TRAINING THE MODEL.  See page 10 of KNN slides
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X, Y)

# get a prediction
new_instance = np.asmatrix([3, 2])
prediction = knn_classifier.predict(new_instance)

