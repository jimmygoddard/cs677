"""
Jimmy Goddard
4/6/19
CS 677 Assignment 9
"""
import datetime
import os
import platform

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas_datareader import data as web
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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


def estimate_coef(x, y):
    n = np.size(x)
    mu_x, mu_y = np.mean(x), np.mean(y)
    ss_xy = np.sum(y*x) - n * mu_y * mu_x
    ss_xx = np.sum(x*x) - n * mu_x * mu_x
    slope = ss_xy / ss_xx
    intercept = mu_y - slope * mu_x
    return slope, intercept


def gradient_descent(x, y, l, epochs=100):
    a = 0
    b = 0
    n = len(x)
    error = 0
    slope, intercept = estimate_coef(x, y)
    for i in range(epochs):
        y_pred = slope * x + intercept
        error = sum((y - y_pred) * (y - y_pred)) / n
        d_slope = (-2/n) * sum(x * (y - y_pred))
        d_intercept = (-2/n) * sum(y - y_pred)
        a = a - l * d_slope
        b = b - l * d_intercept
    return a, b, error


w = 10
df = pd.concat([training_df, testing_df]).reset_index(drop=True)
lr_acc = {'l': [], 'accuracy': []}
y_hat_values = np.empty(len(df))
y_hat_values.fill(np.nan)
learning_rates = np.arange(0.01, 0.06, 0.01)
for l in learning_rates:
    print(f'Calculating gradient descent linear regressions for for learning rate {l}')
    for r_idx, row in df.iterrows():
        window = df[r_idx:r_idx + w]
        x = window[HEADER_PRICE]
        y = window[HEADER_RETURN]
        m, b, error = gradient_descent(x, y, l)
        if np.isnan(m) or np.isnan(b):
            raise Exception(f'm={m} or b={b} is NaN')
        try:
            y_hat = m * df[HEADER_PRICE][r_idx + w] + b
            y_hat_values[r_idx + w] = y_hat
        except KeyError:
            break

    df[HEADER_Y_HAT] = y_hat_values
    df_2018 = df[df[HEADER_YEAR] == 2018]
    accuracy = len(df_2018[np.sign(df_2018[HEADER_RETURN]) == np.sign(df_2018[HEADER_Y_HAT])]) / len(df_2018)
    lr_acc['l'].append(l)
    lr_acc['accuracy'].append(accuracy)

lr_acc_table = pd.DataFrame(lr_acc)

# Question 1.  I couldn't seem to get any of the different learning rates to converge.  For each the slope, intercept,
# and error became very small very quickly.
print(lr_acc_table)
#      l  accuracy
# 0  0.1  0.517928
# 1  0.2  0.517928
# 2  0.3  0.517928
# 3  0.4  0.517928
# 4  0.5  0.517928


total = pd.concat([training, testing]).reset_index(drop=True)

# this works well for a diagonal line
m = 0.03 / 0.02
b = 0.02
# y = 1.5x + 0.02
greens = total[(total['std'] < m * total['mean'] + b) & (total['label'] == 'green')]
reds = total[(total['std'] > m * total['mean'] + b) & (total['label'] == 'red')]
new_df = pd.concat([greens, reds]).reset_index(drop=True)


X = new_df[['std', 'mean']]
Y = new_df['label']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=3)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
log_reg_classifier = LogisticRegression()
log_reg_classifier.fit(X_train, Y_train)
prediction = log_reg_classifier.predict(X_test)
accuracy = np.mean(prediction == Y_test)
print(f'accuracy = {accuracy}')
# accuracy = 0.6666666666666666


x = new_df['mean']
y = new_df['std']
c = new_df['label']
fig, ax = plt.subplots()
plt.scatter(x=x, y=y, c=c)
plt.title('Mean vs Standard deviation')
plt.xlabel('Mean')
plt.ylabel('Standard deviation')

# line I chose to split red and green
y_vals = b + m * x
plt.plot(x, y_vals, 'b-')
buffer = 0.015
x_min, x_max = X.values[:, 0].min() - buffer, X.values[:, 0].max() + buffer
y_min, y_max = X.values[:, 1].min() - buffer, X.values[:, 1].max() + buffer
h = .02  # step size in the mesh

# this is right out of the sklearn documentation:
# https://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html
# but I can't get it to work properly
# it's supposed to show the decision boundary for the logistic regression
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = log_reg_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
# plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(x, y, c=c)
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# plt.xlim(xx.min(), xx.max())
# plt.ylim(yy.min(), yy.max())
# plt.xticks(())
# plt.yticks(())

plt.show()
