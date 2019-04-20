"""
Jimmy Goddard
4/20/19
CS 677 Assignment 12
"""
import datetime
import os
import platform
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas_datareader import data as web
from sklearn.cluster import KMeans

HEADER_LONG = 'Long_MA'
HEADER_SHORT = 'Short_MA'
HEADER_WEEK = 'Week'
HEADER_RETURN = 'Return'
HEADER_DATE = 'Date'
HEADER_AVG = 'Rolling AVG'
HEADER_SD = 'Rolling SD'
HEADER_PRICE = 'Adj Close'
HEADER_YEAR = 'Year'
HEADER_LABEL = 'Week Label'
HEADER_Y_HAT = 'Y hat'


def get_stock(ticker, start_date, end_date, s_window, l_window):
    try:
        df = web.get_data_yahoo(ticker, start=start_date, end=end_date)
        df[HEADER_RETURN] = df[HEADER_PRICE].pct_change()
        df[HEADER_RETURN].fillna(0, inplace=True)
        df[HEADER_DATE] = df.index
        df[HEADER_DATE] = pd.to_datetime(df[HEADER_DATE])
        df['Month'] = df[HEADER_DATE].dt.month
        df['Year'] = df[HEADER_DATE].dt.year
        df['Day'] = df[HEADER_DATE].dt.day
        for col in ['Open', 'High', 'Low', 'Close', HEADER_PRICE]:
            df[col] = df[col].round(2)
        df['Weekday'] = df[HEADER_DATE].dt.weekday_name
        df[HEADER_SHORT] = df[HEADER_PRICE].rolling(window=s_window, min_periods=1).mean()
        df[HEADER_LONG] = df[HEADER_PRICE].rolling(window=l_window, min_periods=1).mean()
        col_list = [HEADER_DATE, 'Year', 'Month', 'Day', 'Weekday', 'Open',
                    'High', 'Low', 'Close', 'Volume', HEADER_PRICE,
                    HEADER_RETURN, HEADER_SHORT, HEADER_LONG]
        df = df[col_list]
        return df
    except Exception as error:
        print(error)
        return None


def get_last_digit(y):
        x = str(round(float(y), 2))
        x_list = x.split('.')
        fraction_str = x_list[1]
        if len(fraction_str) == 1:
            return 0
        else:
            return int(fraction_str[1])


def get_data_table(ticker='GS', start_date='2014-01-01', end_date='2018-12-31'):
    """
    Retrieves stock data, writes it to a CSV file and returns it as a matrix.  Provided to us as is by the course
    Professor

    :return: data table matrix
    """
    # ticker = 'GS'  # Goldman Sachs Group Inc
    # ticker = 'GDDY'  # GoDaddy
    # ticker = 'GM'  # General Motors
    # ticker = 'GRUB'  # GrubHub
    # start_date = '2014-01-01'
    # end_date = '2018-12-31'
    s_window = 14
    l_window = 50

    if platform.system() == 'Windows':
        home_dir = os.path.join('C:', os.path.sep, 'Users', 'jimmy_000')  # MS Windows home directory
    else:  # Assumes Linux
        home_dir = os.path.join(os.path.sep + 'home', 'jgoddard')  # Linux home directory
    input_dir = os.path.join(home_dir, 'src', 'git', 'CS677', 'datasets')
    output_file = os.path.join(input_dir, ticker + '.csv')

    if not os.path.isfile(output_file):
        df = get_stock(ticker, start_date, end_date, s_window, l_window)
        df.to_csv(output_file, index=False)
    else:
        df = pd.read_csv(output_file)
    return df


def get_week(local_date):
    date_format = '%Y-%m-%d'
    dt = datetime.datetime.strptime(local_date, date_format)
    return dt.isocalendar()[1]  # get number of week in year


def label_good_weeks(good_weeks):
    def get_label(local_date):
        if local_date in good_weeks:
            return 0  # 'green'
        else:
            return 1  # 'red'
    return get_label


gs_df = get_data_table(end_date='2019-01-08')
gs_df[HEADER_WEEK] = gs_df[HEADER_DATE].apply(get_week)
# criteria for a good week is that the sum of the returns for each day of that week were positive:
is_good_return_by_week = gs_df[HEADER_RETURN].groupby(gs_df[HEADER_WEEK]).sum() > 0
only_good = is_good_return_by_week[is_good_return_by_week == True].index
positive_week_dates = list(only_good)
gs_df[HEADER_LABEL] = gs_df[HEADER_WEEK].apply(label_good_weeks(positive_week_dates))

data_df = gs_df

returns = data_df[HEADER_RETURN].groupby(data_df[HEADER_WEEK])
means = returns.mean().values
standard_devs = returns.std().values
labels = [group.values[0] for name, group in data_df[HEADER_LABEL].groupby(data_df[HEADER_WEEK])]


new_data_df = pd.DataFrame({'mean': means, 'std': standard_devs, 'label': labels})

X = new_data_df[['mean', 'std']].values
Y = new_data_df[['label']].values.ravel()

# Task 1
k_means = KMeans(n_clusters=5)
k_means.fit(X, Y)
y_k_means = k_means.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_k_means, s=50, cmap='viridis')
centers = k_means.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.show()

# Task 2
k_values = list(range(1, 20))
distortions = []
for k in k_values:
    k_means = KMeans(n_clusters=k)
    k_means.fit(X, Y)
    distortions.append(k_means.inertia_)
    y_k_means = k_means.predict(X)
    # plt.scatter(X[:, 0], X[:, 1], c=y_k_means, s=50, cmap='viridis')
    centers = k_means.cluster_centers_
    # plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
    # plt.title(f'K = {k}')
    # plt.show()

plt.plot(k_values, distortions)
plt.xlabel('K')
plt.ylabel('Distortions')
plt.title('K vs Distortions')
plt.show()

for i, d in zip(k_values, distortions):
    print(i, d)
# 1 0.001129825779070519
# 2 0.000679458182746947
# 3 0.0004440374652797004
# 4 0.00032460120794303843
# 5 0.00023388310010424772
# 6 0.00019201865451832337
# 7 0.00015138974695874882
# 8 0.00013295968573437826
# 9 0.00011640431257100816
# 10 9.898988133085249e-05
# 11 8.658920796340571e-05
# 12 7.651767504344593e-05
# 13 6.835527765280836e-05
# 14 5.922658140028837e-05
# 15 5.408255564840768e-05
# 16 4.9034294760014274e-05
# 17 4.3550298987691445e-05
# 18 3.6707575607739105e-05
# 19 3.578562383108236e-05

# Optimal k = 6 with distortion 0.00019201865451832337.  From there out the gains to distortion by increasing k are less
# pronounced


# Task 3
k_means = KMeans(n_clusters=6)
k_means.fit(X, Y)
y_k_means = k_means.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_k_means, s=50, cmap='viridis')
centers = k_means.cluster_centers_
cluster_df = pd.DataFrame({'label': Y, 'cluster': y_k_means})
print('Cluster #\t% red\t% green')
for name, group in cluster_df.groupby('cluster'):
    group_size = len(group)
    count_red = len(group[group.label == 1])
    percent_red = count_red / group_size
    percent_green = 1 - percent_red
    print(f'{name}\t\t\t{int(percent_red * 100)}\t\t{int(percent_green * 100)}')

# Yes, the clusters did tend to pick up the same label per cluster:
# Cluster #	% red	% green
# 0			70		30
# 1			0		100
# 2			81		18
# 3			0		100
# 4			100		0
# 5			100		0


# Task 4 with help from https://mubaris.com/posts/kmeans-clustering/
# and https://www.kaggle.com/andyxie/k-means-clustering-implementation-in-python/comments
k = 6
n = X.shape[0]
# Number of features in the data
c = X.shape[1]

# Generate random centers, here we use sigma and mean to ensure it represent the whole data
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
centers = np.random.randn(k, c) * std + mean
centers_old = np.zeros(centers.shape)  # to store old centers
centers_new = deepcopy(centers)  # Store new centers

clusters = np.zeros(n)
distances = np.zeros((n, k))
error = np.linalg.norm(centers_new - centers_old)
while error != 0:
    # Measure the distance to every center
    for i in range(k):
        distances[:, i] = np.linalg.norm(X - centers[i], axis=1)
    # Assign all training data to closest center
    clusters = np.argmin(distances, axis=1)

    centers_old = deepcopy(centers_new)
    # Calculate mean for every cluster and update the center
    for i in range(k):
        centers_new[i] = np.mean(X[clusters == i], axis=0)
    error = np.linalg.norm(centers_new - centers_old)

colors = ['r', 'g', 'b', 'y', 'c', 'm']
fig, ax = plt.subplots()
for i in range(k):
    points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
    ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
ax.scatter(centers_new[:, 0], centers_new[:, 1], s=200, c='#050505')

plt.title(f'K-means for k = {k}')
plt.show()
