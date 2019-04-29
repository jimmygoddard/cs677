"""
Jimmy Goddard
4/29/19
CS 677 Assignment 13
"""
import datetime
import os
import platform


import numpy as np
import pandas as pd
from pandas_datareader import data as web
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

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

# decision_tree = tree.DecisionTreeClassifier(criterion='entropy')
# decision_tree.fit(X_train, Y_train)
# prediction = decision_tree.predict(X_test)
# accuracy = np.mean(prediction == Y_test)
# print(f'Accuracy for decision tree using entropy as its criterion is {accuracy}')

N = range(1, 11)
D = range(1, 6)
errors = []
x = []
y = []
for d in D:
    for n in N:
        random_forest = RandomForestClassifier(criterion='entropy', n_estimators=n, max_depth=d)
        random_forest.fit(X_train, Y_train)
        prediction = random_forest.predict(X_test)
        error_rate = np.mean(prediction != Y_test)
        errors.append(error_rate)
        y.append(d)
        x.append(n)
        print(f'Error rate for random forest using entropy as its criterion with n={n} and d={d} is {error_rate}')

s = errors
# scale s
s = [abs(value * 2000) for value in s]
plt.title('Error rate per number of estimators and max tree depth')
plt.xlabel('Number of estimators')
plt.ylabel('Max tree depth')
plt.scatter(x=x, y=y, s=s)
plt.show()

# most optimal random forest
n = 4
d = 1
random_forest = RandomForestClassifier(criterion='entropy', n_estimators=n, max_depth=d)
random_forest.fit(X_train, Y_train)
prediction = random_forest.predict(X_test)
error_rate = np.mean(prediction != Y_test)
print(f'Random forest with {n} estimators and {d} max depth error rate is {error_rate}')

models = ['Guassian Naive Bayesian', 'kNN, k=11 distance=minkoswki', 'Logistic Regression', 'SVM linear kernel',
          'Decision tree', 'Random forest with 4 estimators and max depth of 1'],
model_data = {
    'model': ['NB', 'kNN', 'Log Reg', 'SVM', 'Decision Tree', 'Random Forest'],
    'parameters': ['Guassian', 'k=11 distance=minkowski', '', 'kernel=linear', 'criterion=entropy',
                   'estimators=4, max depth=1, criterion=entropy'],
    'error rate': [1 - 0.673077, 1 - 0.807692, 1 - 0.6666666666666666, 1 - 0.5192307692307693, 1 - 0.6538461538461539,
                   0.40384615384615385]
}

model_df = pd.DataFrame(model_data)
error_rates = [int(error_rate) for error_rate in model_df['error rate'] * 100]
model_df['error rate'] = error_rates
model_df.sort_values(by=['error rate'], ascending=False).plot.bar(x='model', y='error rate')
plt.title('Error rates for each model')
plt.show()
print(model_df.sort_values(by=['error rate']))
#            model                                    parameters  error rate
# 1            kNN                       k=11 distance=minkowski          19
# 0             NB                                      Guassian          32
# 2        Log Reg                                                        33
# 4  Decision Tree                             criterion=entropy          34
# 5  Random Forest  estimators=4, max depth=1, criterion=entropy          40
# 3            SVM                                 kernel=linear          48
