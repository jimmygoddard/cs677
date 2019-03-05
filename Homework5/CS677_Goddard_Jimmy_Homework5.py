"""
Jimmy Goddard
3/2/19
CS 677 Assignment 5
"""
import datetime
import os
import platform

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas_datareader import data as web
from sklearn.neighbors import KNeighborsClassifier
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
            return 'green'
        else:
            return 'red'
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

k_values = [1, 3, 5, 7, 9, 11]
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

error_rate = []
for k in k_values:
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train, Y_train)
    pred_k = knn_classifier.predict(X_test)
    error_rate.append(np.mean(pred_k != Y_test))

x = k_values
y = error_rate
plt.title('Error Rate vs k')
plt.xlabel('number of neighbors: k')
plt.ylabel('Error Rate')
plt.plot(x, y, '-bo')
plt.show()

df_2019 = gs_df[gs_df[HEADER_YEAR] == 2019]
returns_2019 = df_2019[HEADER_RETURN].groupby(df_2019[HEADER_WEEK])
week_one_mean = returns_2019.mean().values[0]
week_one_std = returns_2019.std().values[0]
week_one_label = [group.values[0] for name, group in df_2019[HEADER_LABEL].groupby(df_2019[HEADER_WEEK])]
# this is the knn using k = 11
print(knn_classifier.predict([[week_one_mean, week_one_std]]))
# ['red']

