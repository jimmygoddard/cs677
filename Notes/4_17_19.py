"""
Upcoming assignment
Take the stock
Take 3 other indices
Consider constructing portfolios our stock and one of those indices
We will use one traditional approach and one machine learning approach

SVM
Transformation that you use to go to a higher dimensions space is called the "kernel"
Regularization is when you add a penalty to the SVM to allow for "soft margins" where depending on the penalty, you may
  get more or less misclassified data points
Regularization in general is adding a penalty function.  We saw this same mechanism when we were looking at higher
  degree polynomial linear regressions.  You can add a penalty function to penalize the number/content of weights being
  considered in the higher degree polynomial function

CLUSTERING

Prediction vs Classification
Clustering is used to classify (unsupervised)
Prediction is used to predict new values given that you've seen previous values (supervised)
Classification allows you to make a much simpler description of a data set.  For example, "You have one million people,
  but they may fall into 5 different classes and you can describe the million people in terms of five features"

k-means: clustering algorithm which describes each group based on their geometric mean
Informal definition of k-means clustering: The distance from a point in a cluster to the mean of that cluster must be
  less than the distance from that point to any other cluster's mean

Algorithm -- Assignment and Update:
1. Assign each data point to a cluster (called "assignment")
2. Calculate the mean of each cluster (called "update")
3. Calculate if each point in a cluster is closer to its cluster's mean or another cluster's mean
4. If data point is closer to another cluster's mean, assign that data point to that cluster and go back to step 2
5. End

One major problem with k-means is that the final clusters are fairly highly dependent on the initial centroids
Only in one dimension is this not a problem
There are a number of heuristics, and we can see them in sklearn, whose only variation is trying to inspect the data
  initially to try to determine the initial centroids better than just random choices (which is what the base k-means
  method does)
You can also specify the centroids manually

How to calculate "k"?
Pick a k, calculate clusters 100 times, take average of J and plot it
Pick a larger k, do the same
At some point, this will yield smaller and smaller benefits by increasing k
And if k = n (number of points in dataset), J will be zero by definition


How would you do it if you wanted to require that clusters had the same number of elements?
You'd have to use regularization, penalize larger sized clusters.  But how?

J = sum(sum((x - mu(Ci))^2) + lambda * size(Ci)^2
You could calculate the mean of the entire dataset, and find the largest distance of one point in the dataset to the mean
This value could be used for lambda.  Or at least something in that ballpark.  That measurement exercise at least gives
you an idea of what scale lambda should be in (fraction, tens, millions, zillions, etc.)
============================== Assignment ==============================
K-Means due monday (combine with Decision trees?)
Look at the data set and figure out "how many clusters are there".  It's not an easy question to
answer. We will not be using the labels that we've used previously since we will be using unsupervised learning

We will be implementing our own k-means algorithm in addition to using the library

-- Use k-means to cluster our weeks and see the composition of the clusters
  a) does k-means clustering find the red/green groups that we labeled before (even for k > 2)

1) We have labeled data (2017 and 2018)
Take (x1, y1), ..., (x104, y104) -- two years worth of data -- and cluster

Task 1:
  Use sklearn library to do this, use random initialization (take the defaults), take 5 clusters, and plot the results

Some graphing help: https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html

Task 2:
  For the same dataset, find out the best k to use for each k = [1, 2, 3, 4, 5, 6, 7, 8].  Plot the distortion
  (k_means.inertia_) vs k and find out the best k

Task 3: for this optimal k, examine your clusters:
  Cluster #        % red weeks        % green weeks

Task 4: implement k-means (assignment and update) for euclidean, manhattan, and minkowski distances
  https://mubaris.com/posts/kmeans-clustering/ and his slides examples

Portfolio Task (not required to be turned in, pure optional):
Take 3 ETF's (exchange traded fund)
XLE - energy ETF, XLF - financials, XLV - healthcare
XLE, XLF, and XLV are the ticker symbols

We will need to download and compute the dataframes (daily returns) for these three tickers for both 2017 and 2018

Consider 3 portfolios: P1, P2, P3:
1) X + XLE  (50/50 split)
2) X + XLF  (50/50 split)
3) X + XLV  (50/50 split)

For each portfolio compute the following:
  1) Return w1 * R(X) + w2 * R(ETF)   # R(X) is the "return of X" (should be average), w1 and w2 are the weights: 0.5
    a) this gives us the average daily returns for each portfolio
  2) Compute Risk (standard deviation of returns)
    a) var = w1^2 * sigma1^2 + 2 * w1 * w2 * sigma1 * sigma2 + w2^2 * sigma2^2
      What you need: compute covariance matrix using numpy
      Compute risk as square root of the variance
      see examples here https://s3.amazonaws.com/assets.datacamp.com/production/course_5612/slides/chapter2.pdf

For each of P1, P2, P3 compute Return, Risk
Graph return of P on x-axis, risk of P on y-axis, each portfolio is a different color

take the following 10 combinations of weights for each portfolio and plot them in the way as above.
  each portfolio is a different color:
10, 90
20, 80
30, 70
40, 60
50, 50
60, 40
70, 30
80, 20
90, 10
100, 0 (no ETF at all)

This way you will get 3 points for each pair (X, ETF) - plot these 30 points (3 colors) and explain what you got
============================== Assignment End ==============================

============================== Presentation ==============================
Say who we are
Show our conclusions (no code)
2 questions:
- most important accomplishment
- biggest failure
5 or 6 minutes

submit the description (2 or 3 pages), the code, (and some "screenshots" of the code running?)
video is a 2 or 3 minute video
============================== Presentation ==============================

"""
