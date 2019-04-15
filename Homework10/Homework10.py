"""
Jimmy Goddard
4/15/19
CS 677 Assignment 10
"""
import datetime
import os
import platform

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas_datareader import data as web
from sklearn import svm
from sklearn.preprocessing import StandardScaler

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

training_df = gs_df[gs_df[HEADER_YEAR] == 2017]
testing_df = gs_df[gs_df[HEADER_YEAR] == 2018]

training_returns = training_df[HEADER_RETURN].groupby(training_df[HEADER_WEEK])
training_means = training_returns.mean().values
training_std = training_returns.std().values
training_labels = [group.values[0] for name, group in training_df[HEADER_LABEL].groupby(training_df[HEADER_WEEK])]

testing_returns = testing_df[HEADER_RETURN].groupby(testing_df[HEADER_WEEK])
testing_means = testing_returns.mean().values
testing_std = testing_returns.std().values
testing_labels = [group.values[0] for name, group in testing_df[HEADER_LABEL].groupby(testing_df[HEADER_WEEK])]

training = pd.DataFrame({'mean': training_means, 'std': training_std, 'label': training_labels})
testing = pd.DataFrame({'mean': testing_means, 'std': testing_std, 'label': testing_labels})

X_train = training[['mean', 'std']].values
Y_train = training[['label']].values.ravel()
X_test = testing[['mean', 'std']].values
Y_test = testing[['label']].values.ravel()


X_train = StandardScaler().fit_transform(X_train)

svm_linear = svm.SVC(kernel='linear')
svm_linear.fit(X_train, Y_train)
predicted = svm_linear.predict(X_test)
accuracy = np.mean(predicted == Y_test)
print(f'Linear SVM accuracy is {accuracy}')

svm_gaussian = svm.SVC(kernel='rbf')
svm_gaussian.fit(X_train, Y_train)
predicted = svm_gaussian.predict(X_test)
accuracy = np.mean(predicted == Y_test)
print(f'Gaussian SVM accuracy is {accuracy}')

degrees = [2, 5, 9]
svm_poly = []
for degree in degrees:
    svm_classifier = svm.SVC(kernel='poly', degree=degree)
    svm_classifier.fit(X_train, Y_train)
    svm_poly.append((svm_classifier, degree))
    predicted = svm_classifier.predict(X_test)
    accuracy = np.mean(predicted == Y_test)
    print(f'Polynomial of degree {degree} SVM accuracy is {accuracy}')


# The linear SVM model had the accuracy as the polynomial SVM models even though the polynomial kernels appeared quite
# different.  See Graphs for details of the different shapes each SVM model took.
# Linear SVM accuracy is 0.5192307692307693
# Gaussian SVM accuracy is 0.4807692307692308
# Polynomial of degree 2 SVM accuracy is 0.5192307692307693
# Polynomial of degree 5 SVM accuracy is 0.5192307692307693
# Polynomial of degree 9 SVM accuracy is 0.5192307692307693

# The following graphing code adapted from
# https://scikit-learn.org/stable/auto_examples/svm/plot_iris.html#sphx-glr-auto-examples-svm-plot-iris-py

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


models = (svm_linear, svm_gaussian, svm_poly[0][0], svm_poly[1][0], svm_poly[2][0])
titles = ('SVC with linear kernel',
          'SVC with gaussian kernel',
          f'SVC with polynomial degree {svm_poly[0][1]} kernel',
          f'SVC with polynomial degree {svm_poly[1][1]} kernel',
          f'SVC with polynomial degree {svm_poly[2][1]} kernel')


# Set-up 3x2 grid for plotting.
fig, sub = plt.subplots(3, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X_train[:, 0], X_train[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=Y_train, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Mean')
    ax.set_ylabel('Standard Deviation')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()
