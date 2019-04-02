"""
y = mx + b + e

assume mean of e is zero
assume y and e are independent
assume e is normally distributed
assume there is no strong correlation amongst any of the independent variables
https://www.statisticssolutions.com/assumptions-of-multiple-linear-regression/

Cov(X, Y) = 0 if their correlation is 0.  in other words, if the two random variables are entirely independent of
each other

Cov(X, Y) = corr * sigma(x) * sigma(y)

Gradient descent is used in cases where taking the partial derivatives of a function and setting them to zero (minimizing
them mathematically) is costly or not possible.

You basically choose a point in f(x) and calculate its tangent.  Then you move ahead of that point by a delta which is
called the learning rate.  then calculate the tangent again.  you're looking for the tangent's slope to be zero.


########################### Assignment 9 ###########################
1. Fix window size at 10. Recompute your results using explicit Python code
  a. You can use code on page 16 or any code you may want to write.  do NOT use sklearn or any other libraries
2. estimate slope and intercept using gradient descent method (page 19, add some value for L learning rate)

You will investigate accuracy for different L and different "epochs"

Plot the following graphs:

Consider 5 learning rates, L = [0.01, 0.02, 0.03, 0.04, 0.05]
np.arange(0.01, 0.06, 0.01)
# array([0.01, 0.02, 0.03, 0.04, 0.05])
Take the number of epochs = 100
Use gradient descent to estimate slope and intercept
Estimate P11 ----> return for day 11 (same as before)

for every window of 10, use gradient descent to compute slope and intercept (over 100 epochs), then use
the resulting slope and intercept to predict day 11 and compare it against day 11 return (in same way as we did in
Homework 8)

we should generate the accuracy prediction (like we did before) for every learning rate

extra credit:
plot autocorrelation function on returns for my data set

plt.acorr(df['Return'])
plt.show()

Assignment for logistic regression:
1. Take a linearly-separable dataset that you created for 2017-2018
2. Apply logistic regression to it
3. Calculate accuracy

Look at docs for logistic regression in sklearn and find out the equation that separates your points

On the same graph, plot your colored points, plot the line that you used to remove the extra points, and plot the actual
logistic regression line

http://benalexkeen.com/linear-regression-in-python-using-scikit-learn/

Part 2: look at the original dataset
use points from 2017 to train a logistic regression classifier and apply this classifier to predict
labels for each week in 2018

compute your accuracy (% of weeks that you correctly predicted the label)

# Homework Summary
Summary: homework consists of linear regression and logistic regression

Linear regression:
(a) write formulas for slope and intercept without using sklearn and recompute results for W = 10
(b) implement gradient descent approach to estimating slope and intercept and use to predict
(use W = 10 and use gradient descent to compute slope and intercept for different learning rates, L = 0.01 - 0.05,
and then estimate accuracy)

table:
L        accuracy for 2018
0.01     0.48 # for example
0.02     0.39 # for example

Logistic Regression
(a) compute the equation for the reduced dataset and plot that together with points and the line that you chose
to remove the extra points
(b) original dataset 2017, logistic regression to predict 2018

########################### Assignment 9 End ###########################

########################### Logistic Regression ###########################
2 years of data, 2017-201
(x, y, label)
X - average of 5 daily returns
Y - standard deviation of 5 returns

Logistic regression works well if your classes can be separated by a line
Logistic regression input data needs to be scaled just like kNN and linear regression (each of these deals with
distances in a way)
Naive Bayesian input data does not need to be scaled as it is not dealing with distances

########################### Remaining Material ###########################
Support Vector machines
Decision Trees
Clustering

Ensemble Learning: combining classifiers together

If time permits:
Random forests

########################### Support Vector Machines ###########################
https://scikit-learn.org/stable/modules/svm.html
The advantages of support vector machines are:

Effective in high dimensional spaces.
Still effective in cases where number of dimensions is greater than the number of samples.
Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.
Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.
The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.
SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.arange(0.01, 0.06, 0.01)
# array([0.01, 0.02, 0.03, 0.04, 0.05])

df = pd.DataFrame({'Return': [0, 1, 2, 3, 4, 5]})  # or use the gs_df
plt.acorr(df['Return'])
plt.show()

