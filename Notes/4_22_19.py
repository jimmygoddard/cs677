"""
Decision tree problems:
1) Overfitting - very sensitive to the precision of your dataset.  changing one data point can have dramatic effects on
  the decision tree

Dealing with Decision tree problems:
Random forest - construct a collection of decision trees and use the entire collection to make a classification

Decision trees:
- Can be used with continuous or categorical data
- It is supervised learning (takes labels as an input)
- Used for classification (also listed prediction/regression in slide)

Decision trees are very sensitive to the data (and noise in the data).  I think that's high bias

Bias vs variance in classifier predictions
from https://towardsdatascience.com/understanding-the-bias-variance-tradeoff-165e6942b229:

What is bias?

Bias is the difference between the average prediction of our model and the correct value which we are trying to predict.
Model with high bias pays very little attention to the training data and oversimplifies the model. It always leads to
high error on training and test data.

What is variance?

Variance is the variability of model prediction for a given data point or a value which tells us spread of our data.
Model with high variance pays a lot of attention to training data and does not generalize on the data which it hasn’t
seen before. As a result, such models perform very well on training data but has high error rates on test data.

Random forests are trained using data selected using replacement.  You can specify both the number of decision trees as
well as the max depth of the decision trees (he used 5 decision trees and max depth 2 in the example. iris example
used 25 trees and max depth 5).  The more subtrees and higher depth the longer the classifier will take to run on larger
datasets


## using machine learning to construct a portfolio

matrix of covariance between two stocks = COV(X1, YX2 is the relationship between the
two stocks = std(X1) * std(X2) * pearson_coef(X1, X2) and is 0 if the two stocks are entirely independent statistically

Construct a portfolio of multiple stocks with weights, W1, ... WN, where the sum of the weights equals 1 (percent of each
stock to take)

Markowitz portfolio:
Return of portfolio:
R(W) - x_1 * r_1 + ... + w_N * r_N

Given some acceptable return R*, compute the weights that minimize risk

Then construct a series of cluster numbers for each year of stocks.
You can use a hamming distance to calculate the similarity/differences between these vectors
Select stocks that have the largest hamming distances
============================= Presentation =============================
5-6 mins
Tell us name
Tell us what we worked on
Tell us what we achieved
Tell us where we failed
"If we had more time, what would we do.  what would we add to this?"
No code.

Turn in:
- Video: introduction and present slides
- Written Description: 2-3 pages describing the project and code
- Code
- Add screenshots of the code, how to run it (maybe they can't run it)

Tips:
- Be honest with your data
- Provide working scripts (keep it simple.  better to have a simple working piece of code rather than a complex piece of
  code that doesn't run
============================= Presentation End =============================



============================= Random Forest Assignment =============================
Implement Random Forest basically same as Decision Tree (train on 2017, test on 2018)

Task 1:
Build and analyze a random forest classifier for our data
Figure out best number of estimators and the best depth
Two hyperparameters:
 1) N - number of estimators (number of (sub)trees to use)
 2) D - max depth of each subtree

Train on 2017, Test on 2018, calculate error
N = range(1, 11)
D = range(1, 6)
Graph results:
x-axis will be N, y-axis will be D
Size of point will reflect the error
Identify at least one point that is the best (there could be more than one point that's the best)
Tie breaker is the lowest error rate which also has the smallest N and D
Use same "criterion" as we used for the Decision Tree homework (which was either "entropy" or default/omitted param

Task 1.5:
- Construct a comparison table for your classifiers
-- For kNN and random forest: use the most accurate model, put hyperparameters in classifier name (or additional column):
-- just dig up the values we've already produced and put them in the table

Classifier        Parameters      Error Rate
kNN               k = 9           55%
decision tree     entropy         51%
logistic regression               49%   # did we do classification or just use this for sign?
SVM               linear/etc      61%
Random forest     n = 10, d = 3   45%
Naive Bayesian    gauss           49%

Round percentages (not 0.49999, use 49%)
============================= Random Forest Assignment End =============================


"""
