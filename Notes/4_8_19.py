"""
Prediction, Classification

Labeled data -> predict label
Logistic regression, SVM

A: is linear regression prediction or classification?
Both?  (mostly prediction, can be used for classification.  wasn't clear why)
No, not both.  Read up on linear regression for binary outcomes:
http://thestatsgeek.com/2015/01/17/why-shouldnt-i-use-linear-regression-if-my-outcome-is-binary/

logistic is much better suited to binary outcomes

Linear regression:
y = ax + b

y_pred = a * x_i + b
e_i = y_i - (a * x_i + b) = yi - y_pred

gradient descent is the method by which you compute the minimal value of some function

What is Logistic Regression?
Motivation: separate points into two "regions" by a linear boundary

In logistic regression we use the sigmoid function to make a classification (because we want to have smooth functions)

Do we need to scale for logistic regression?
Yes

What classification methods require rescaling?
any method that uses distance somewhere (or decides on the "closeness" of the points)
kNN - yes
naive bayesian - no
regression - yes, but not as big an impact as long as the data is centered
SVM - yes

examples:

cx # multiply random variable by constant
mu_2 = c * mu_1
sigma_2 = sigma_1

cx + d # multiply by constant and add another constant
mu_2 = c * mu_1 + d
sigma_2 = sigma_1

Now assume we have two stocks: X1, X2

X1 and X2 --> w1 * X1 + w2 * X2
Y = w1 * X1 + w2 * X2

Q: how to compute mu_Y and sigma_Y?
mu_Y = w1 * mu_1 + w2 * mu_2  # simple enough
covariance needs to be computed:
COV(X1, X2) = w1 * sigma_1 + w1 * w2 * sigma_1 * sigma_2 + w2 * sigma_2 ^2

COV(X1, X2) = sum((X1 - mu_x1) * (X2 - mu_x2)) / n

pearson correlation coefficient relationship to covariance:
p = COV(X1, X2) / sigma_x1 * sigma_x2

Upcoming topics:
Support Vector Machines
Decision Trees
Clustering (k-means)

If time:
Random forests, ...

Support Vector Machine is a classifier that predicts labels (like Logistic Regression)
Basically you create "the thickest possible line that separates your points"

- use "thickest" line
- maximize "margins".  margin is the distance between the middle "decision line" and the support vectors on either side

in linear regression, the points are not labeled and you're only computing the line which minimizes the sum of square of
errors

in a SVM machine, you do have labels.  you could theoretically compute many lines to separate your labels.  the process
is to choose the "thickest" line which separates the labels

With SVM we start with an assumption of linear separability.  This can often be achieved in a number of different ways
For example, take two labels in two dimensions. If you add a third dimension and one label is at one value of the third
dimension and all observations with the other label are in a different value of the third dimension then these points
are separable through a hyperplane.

================================= Homework 10 =================================
1) Required part: use linear SVM to predict your labels

2017 labeled data set (52 rows)
(x, y, {Good, Bad})
X average of 5 daily returns for a week
Y standard dev of these five values
You don't really need to scale the values (optional)
Use 2017 to train your linear SVM, apply it to 2018 data set and compute accuracy

Optional (only change is to change the kernel and in the case of poly kernels, include a degree):
Take Gaussian SVM and poly SVM
Compute accuracy
Tell us if these are superior to the linear SVM
================================= Homework 10 end =================================

================================= Homework 11 =================================
Due in two weeks
Homework on decision trees
Very similar to SVM
Take 2017 data, train a decision tree classifier on it, and apply the classifier to 2018 and compute accuracy
Use "entropy" as the decision tree parameter
================================= Homework 11 end =================================

Decision Trees
inputs and outputs
rules
supervised learning
constructed based on information gain
classification or prediction (regression)
CART: classification or regression tree

graphically it's a tree

Organize your tree by minimizing your entropy at each level thereby maximizing information gain
There is a mathematically formula based on log and event probabilities that allows you to calculate entropy


Supervised learning:
SVM, logistic regression, decision trees, kNN are all examples of supervised learning

Unsupervised learning:
You don't know the classes of your observations in advance and want infer the classes
You do need some measure of similarity between the observations, some sort of distance metric

Given a set of N points, (x1, y1), ..., (xn, yn), how do you split them into groups?


"""