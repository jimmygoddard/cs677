"""
Jimmy Goddard
3/30/19
CS 677 Assignment 8
"""
import datetime
import os
import platform

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas_datareader import data as web
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
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

distances = [
    (2, 'euclidean'),
    (1, 'manhattan'),
    (1.5, 'minkowski')
]

accuracy_by_model = {'model': [], 'accuracy': []}
for p, model in distances:
    error_rates = []
    for k in k_values:
        knn_classifier = KNeighborsClassifier(n_neighbors=k, p=p)
        knn_classifier.fit(X_train, Y_train)
        pred_k = knn_classifier.predict(X_test)
        error_rates.append(np.mean(pred_k != Y_test))

    accuracy_by_model['model'].append(model)
    accuracy_by_model['accuracy'].append(1 - min(error_rates))
    x = k_values
    y = error_rates
    plt.title(f'Error Rate vs k for {model}')
    plt.xlabel('number of neighbors: k')
    plt.ylabel('Error Rate')
    plt.plot(x, y, '-bo')
    plt.show()

# best k = 11, best metric is either minkowski or manhattan, both with accuracy just above .80

nb_classifier = GaussianNB().fit(X_train, Y_train)
prediction = nb_classifier.predict(X_test)
error_rate = np.mean(prediction != Y_test)
accuracy_by_model['model'].append('naive bayesian')
accuracy_by_model['accuracy'].append(1 - error_rate)
acc_table = pd.DataFrame(accuracy_by_model)

# Question 1 results (best accuracy across kNN and NB models
print(acc_table)
#             model  accuracy
# 0       euclidean  0.750000
# 1       manhattan  0.807692
# 2       minkowski  0.807692
# 3  naive bayesian  0.673077

# rolling linear regression
w = 10
degree = 1
df = pd.concat([training_df, testing_df]).reset_index(drop=True)
lr_acc = {'w': [], 'accuracy': []}
for w in [10, 20, 30]:
    y_hat_values = np.empty(len(df))
    y_hat_values.fill(np.nan)
    for r_idx, row in df.iterrows():
        window = df[r_idx:r_idx + w]
        x = window[HEADER_PRICE]
        y = window[HEADER_RETURN]
        weights = np.polyfit(x, y, degree)
        model = np.poly1d(weights)
        try:
            y_hat = model(df[HEADER_PRICE][r_idx + w])
            y_hat_values[r_idx + w] = y_hat
        except KeyError:
            break

    df[HEADER_Y_HAT] = y_hat_values
    df_2018 = df[df[HEADER_YEAR] == 2018]
    accuracy = len(df_2018[np.sign(df_2018[HEADER_RETURN]) == np.sign(df_2018[HEADER_Y_HAT])]) / len(df_2018)
    lr_acc['w'].append(w)
    lr_acc['accuracy'].append(accuracy)

lr_acc_table = pd.DataFrame(lr_acc)
# Question 2 results: Best linear regression return prediction accuracy for w = 10, 20, 30
print(lr_acc_table)
#     w  accuracy
# 0  10  0.565737
# 1  20  0.585657
# 2  30  0.581673

total = pd.concat([training, testing]).reset_index(drop=True)
total.plot.scatter(x='mean', y='std', c=total['label'])
plt.show()


# this works well for a diagonal line
m = 0.03 / 0.02
# y = 1.5x + 0.02
greens = total[(total['std'] < m * total['mean'] + 0.02) & (total['label'] == 'green')]
reds = total[(total['std'] > m * total['mean'] + 0.02) & (total['label'] == 'red')]
new_df = pd.concat([greens, reds]).reset_index(drop=True)
new_df.plot.scatter(x='mean', y='std', c=new_df['label'])
plt.show()

# line y = x (std = mean).  Above line is red, below is green.  Neither this nor the reverse of green above and red below
# work well
greens = total[(total['std'] < total['mean']) & (total['label'] == 'green')]
reds = total[(total['std'] > total['mean']) & (total['label'] == 'red')]
new_df = pd.concat([greens, reds]).reset_index(drop=True)
new_df.plot.scatter(x='mean', y='std', c=new_df['label'])
plt.show()

# line is x = 0 (mean = 0).  Right of line is green, left of line is red
greens = total[(total['mean'] > 0) & (total['label'] == 'green')]
reds = total[(total['mean'] < 0) & (total['label'] == 'red')]
new_df = pd.concat([greens, reds]).reset_index(drop=True)
x = new_df['mean']
y = new_df['std']
c = new_df['label']
fig, ax = plt.subplots()
plt.scatter(x=x, y=y, c=c)
plt.title('Mean vs Standard deviation')
plt.xlabel('Mean')
plt.ylabel('Standard deviation')
ax.axvline(x=0)
plt.show()
