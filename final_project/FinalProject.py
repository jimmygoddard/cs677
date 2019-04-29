import os
import platform
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.metrics import accuracy_score

plt.rcParams['figure.figsize'] = (10, 6)

RUN_ALL_MODELS = False

marvel_file_name = 'marvel-wikia-data.csv'
dc_file_name = 'dc-wikia-data.csv'

if platform.system() == 'Windows':
    home_dir = os.path.join('C:', os.path.sep, 'Users', 'jimmy_000')  # MS Windows home directory
else:  # Assumes Linux
    home_dir = os.path.join(os.path.sep + 'home', 'jgoddard')  # Linux home directory
input_dir = os.path.join(home_dir, 'src', 'git', 'CS677', 'final_project', 'datasets', 'fivethirtyeight')

marvel_input_file = os.path.join(input_dir, marvel_file_name)
marvel_df = pd.read_csv(marvel_input_file)
marvel_df.columns = marvel_df.columns.str.lower()
marvel_df['publisher'] = 'Marvel'

dc_input_file = os.path.join(input_dir, dc_file_name)
dc_df = pd.read_csv(dc_input_file)
dc_df['publisher'] = 'DC'
dc_df.columns = dc_df.columns.str.lower()

df = pd.concat([marvel_df, dc_df], ignore_index=True)

year = df['year'].dropna().astype(int)
sns.distplot(year)
plt.title('New Heroes Histogram (both publishers)')
plt.ylabel('Probability')
plt.xlabel('Year')
plt.show()

marvel_year = df.year[df.publisher == 'Marvel']
dc_year = df.year[df.publisher == 'DC']

fig, ax = plt.subplots()
sns.distplot(dc_year.dropna(), ax=ax, label='DC')
sns.distplot(marvel_year.dropna(), ax=ax, label='Marvel')
plt.title('New Heroes Histogram By Publisher')
plt.ylabel('Probability')
plt.xlabel('Year')
plt.legend()
plt.show()


sns.catplot(x='year', y='sex', data=df, height=6, aspect=1.6)
plt.title('New Heroes per Year by Gender')
plt.show()

sns.catplot(x='year', y='sex', data=df, kind='boxen', height=6, aspect=1.6)
plt.title('New Heroes per Year by Gender')
plt.show()

sns.catplot(x='year', y='sex', data=df, hue='gsm', height=6, aspect=1.6, legend=False)
plt.legend(loc='lower left')
plt.title('New Heroes per Year by Gender and Sexuality')
plt.show()

g = sns.countplot(x=year)
labels = g.get_xticklabels()
for i, l in enumerate(labels):
    if i % 5 != 0:
        labels[i] = ''
g.set_xticklabels(labels, rotation=90)
plt.title('Count of New Heroes per Year (both publishers)')
plt.show()

g = sns.catplot(x='year', data=df, kind='count', hue='publisher', legend=False, height=6, aspect=1.6)
for ax in g.axes.flat:
    labels = ax.get_xticklabels()
    for i, l in enumerate(labels):
        if i % 5 != 0:
            labels[i] = ''
        else:
            # must be a better way to go from '1935.0' to 1935
            labels[i] = int(float(labels[i].get_text()))
    ax.set_xticklabels(labels, rotation=90)
plt.legend(loc='upper left')
plt.title('Count of New Heroes per Year By Publisher')
plt.show()

# gender analytics
sex_year_df = df[['year', 'sex']]
sex_year_df_grouped = sex_year_df.groupby(['year', 'sex']).size().reset_index(name='counts')

male_df = df[df.sex == 'Male Characters']
female_df = df[df.sex == 'Female Characters']
male_female_df = pd.concat([male_df, female_df], ignore_index=True)
male_female_df_grouped = male_female_df.groupby(['year', 'sex']).size().reset_index(name='counts')
sns.lineplot(x='year', y='counts', hue='sex', data=male_female_df_grouped)
plt.title('New heroes per year by gender (only male/female)')
plt.show()


# number of each type of gender plotted per year, might want to also show just male/female
sns.lineplot(x='year', y='counts', hue='sex', data=sex_year_df_grouped)
plt.title('New heroes per year by gender')
plt.show()



# maybe also show the same type of plot, but count id (secret, public, etc.) alignment, eye/hair, or alive?

# can do appearances by gender too
app_by_gender_grouped = df.groupby(['sex'])['appearances'].count().reset_index()

# appearances by sexuality
app_by_gsm = df.groupby(['gsm'])['appearances'].count().reset_index()

# build classifiers to classify gender:
# 1) a knn - https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
# 2) neural net classifier - https://medium.com/datadriveninvestor/building-neural-network-using-keras-for-classification-3a3656c726c1
# 3) naive bayesian - https://scikit-learn.org/stable/modules/naive_bayes.html

sns.catplot(x='year', y='sex', data=male_female_df, kind='boxen', height=6, aspect=1.6)
plt.title('New Heroes per Year by Gender')
plt.show()

# classification
input_data = male_female_df.drop(['page_id', 'urlslug', 'first appearance', 'name', 'sex'], axis=1)
dummies = [pd.get_dummies(male_female_df[c]) for c in input_data.columns]
encoded_data = pd.concat(dummies, axis=1)

X = encoded_data.values
le = LabelEncoder()
Y = le.fit_transform(male_female_df['sex'].values)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=3)


# This takes a LONG time to complete.
def find_optimal_k_for_knn():
    k_values = [1, 3, 5, 7, 9, 11]

    accuracy_rate = []
    for k in k_values:
        knn_clf = KNeighborsClassifier(n_neighbors=k)
        start = time.time()
        knn_clf.fit(X_train, Y_train)
        end = time.time()
        print(f'kNN, k = {k}, model fitting took {end - start} seconds')
        start = time.time()
        predictions = knn_clf.predict(X_test)
        end = time.time()
        print(f'kNN, k = {k}, model predictions took {end - start} seconds')
        accuracy_rate.append(np.mean(predictions == Y_test))

    x = k_values
    y = accuracy_rate
    plt.title('Accuracy vs k')
    plt.xlabel('Number of neighbors: k')
    plt.ylabel('Accuracy')
    plt.plot(x, y, '-bo')
    plt.show()

    for tup in zip(k_values, accuracy_rate):
        print(tup)
    # (1, 0.6520291550436426)
    # (3, 0.6909025465670836)
    # (5, 0.7142985692432287)
    # (7, 0.725006748852695)
    # (9, 0.7300458921983263)
    # (11, 0.7316656168451363)


# k = 9 appears to be the optimal value of k for kNN
# kNN classifier
def run_knn():
    k = 9
    knn_clf = KNeighborsClassifier(n_neighbors=k)
    start = time.time()
    knn_clf.fit(X_train, Y_train)
    end = time.time()
    print(f'kNN model, k = {k}, fitting took {end - start} seconds')
    start = time.time()
    predictions = knn_clf.predict(X_test)
    end = time.time()
    print(f'kNN model, k = {k}, predictions took {end - start} seconds')
    accuracy = np.mean(predictions == Y_test)
    print(f'kNN, k = {k}, accuracy = {accuracy}')
    # kNN model, k = 9, fitting took 0.6297478675842285 seconds
    # kNN model, k = 9, predictions took 123.2866427898407 seconds
    # kNN, k = 9, accuracy = 0.7300458921983263

# Logistic Regression
lr_clf = LogisticRegression()
lr_clf.fit(X_train, Y_train)
predictions = lr_clf.predict(X_test)
accuracy = np.mean(predictions == Y_test)
print(f'Logistic regression accuracy = {accuracy}')
# Logistic regression accuracy = 0.7507423737964546

# Naive Bayesian
nb_clf = MultinomialNB()
nb_clf.fit(X_train, Y_train)
predictions = nb_clf.predict(X_test)
accuracy = np.mean(predictions == Y_test)
print(f'Naive Bayesian accuracy = {accuracy}')
# Naive Bayesian accuracy = 0.7466930621794295


# This takes a LONG time to complete.
# SVM
def train_svm():
    svm_clf = svm.SVC(kernel='linear')
    start = time.time()
    svm_clf.fit(X_train, Y_train)
    end = time.time()
    print(f'Model fitting took {end - start} seconds')
    start = time.time()
    predictions = svm_clf.predict(X_test)
    end = time.time()
    print(f'Model prediction took {end - start} seconds')
    accuracy = np.mean(predictions == Y_test)
    print(f'Linear SVM accuracy = {accuracy}')
    # Linear SVM accuracy = 0.7472329703950329


# Decision tree
dt_clf = tree.DecisionTreeClassifier(criterion='entropy')
dt_clf.fit(X_train, Y_train)
prediction = dt_clf.predict(X_test)
accuracy = np.mean(prediction == Y_test)
print(f'Accuracy for decision tree using entropy as its criterion is {accuracy}')
# Accuracy for decision tree using entropy as its criterion is 0.6153846153846154

# Random forest
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

# optimal hyperparameters appear to be n = 2, d = 1
n = 2
d = 1
rf_clf = RandomForestClassifier(criterion='entropy', n_estimators=n, max_depth=d)
rf_clf.fit(X_train, Y_train)
prediction = rf_clf.predict(X_test)
accuracy = np.mean(prediction == Y_test)
print(f'Accuracy for random forest using entropy as its criterion with n={n} and d={d} is {accuracy}')
# Accuracy for random forest using entropy as its criterion with n=2 and d=1 is 0.7448933681274184


# kMeans clustering
def find_optimal_k_means():
    k_values = list(range(1, 20))
    distortions = []
    for k in k_values:
        k_means = KMeans(n_clusters=k)
        start = time.time()
        k_means.fit(X, Y)
        end = time.time()
        print(f'kMeans fitting took {end - start} seconds for k = {k}')
        distortions.append(k_means.inertia_)

    distortions = [
        104555.03595050643,
        94799.71566732477,
        89037.77925461614,
        85242.52477457703,
        82702.49217260712,
        80825.34740966791,
        79432.78405873015,
        77797.7634927912,
        76710.79528013534,
        74688.83636048707,
        74079.70647002068,
        73751.91239553502,
        72843.42378733958,
        72210.39894473055,
        71582.40275349027,
        70903.10163810465,
        70700.07871516803,
        70419.49591542348,
        69423.1690742116
    ]

    plt.plot(k_values, distortions)
    plt.xlabel('K')
    plt.ylabel('Distortions')
    plt.title('K vs Distortions')
    plt.show()

    for tup in zip(k_values, distortions):
        print(tup)


# optimal k looks like k = 16
def perform_k_means():
    k = 16
    k_means = KMeans(n_clusters=k)
    start = time.time()
    k_means.fit(X, Y)
    end = time.time()
    print(f'kMeans model, k = {k}, fitting took {end - start} seconds')
    start = time.time()
    clusters = k_means.predict(X)
    end = time.time()
    print(f'kMeans model, k = {k}, prediction took {end - start} seconds')
    centers = k_means.cluster_centers_
    cluster_df = pd.DataFrame({'label': Y, 'cluster': clusters})
    print('Cluster #\t% Male\t% Female')
    for name, group in cluster_df.groupby('cluster'):
        group_size = len(group)
        count_male = len(group[group.label == 1])
        percent_male = count_male / group_size
        percent_female = 1 - percent_male
        print(f'{name}\t\t\t{int(percent_male * 100)}\t\t{int(percent_female * 100)}')

    # kMeans model, k = 16, fitting took 16.888083696365356 seconds
    # Cluster #	% Male	% Female
    # 0			63		36
    # 1			67		32
    # 2			69		30
    # 3			65		34
    # 4			64		35
    # 5			81		18
    # 6			70		29
    # 7			82		17
    # 8			82		17
    # 9			70		29
    # 10		81		18
    # 11		82		17
    # 12		77		22
    # 13		61		38
    # 14		85		14
    # 15		76		23


# Do a voting classifier
def run_voting_classifier():
    knn_clf = KNeighborsClassifier(n_neighbors=9)
    lr_clf = LogisticRegression()
    nb_clf = MultinomialNB()
    svm_clf = svm.SVC(kernel='linear')
    dt_clf = tree.DecisionTreeClassifier(criterion='entropy')

    voting_clf = VotingClassifier(
        estimators=[('knn', knn_clf), ('lr', lr_clf), ('nb', nb_clf), ('svm', svm_clf), ('dt', dt_clf)],
        voting='hard'
    )

    for clf in (knn_clf, lr_clf, nb_clf, svm_clf, dt_clf, voting_clf):
        clf_name = clf.__class__.__name__
        print(f'Running {clf_name} classifier')
        clf.fit(X_train, Y_train)
        start = time.time()
        prediction = clf.predict(X_test)
        end = time.time()
        print(f'{clf_name} prediction took {end - start} seconds')
        print(clf_name, np.mean(Y_test == prediction))
        print(clf_name, accuracy_score(Y_test, prediction))


# run_voting_classifier()
# Running KNeighborsClassifier classifier
# KNeighborsClassifier prediction took 135.62122893333435 seconds
# KNeighborsClassifier 0.7300458921983263
# Running LogisticRegression classifier
# LogisticRegression prediction took 0.02280592918395996 seconds
# LogisticRegression 0.7507423737964546
# Running MultinomialNB classifier
# MultinomialNB prediction took 0.03211832046508789 seconds
# MultinomialNB 0.7466930621794295
# Running SVC classifier
# SVC prediction took 56.472883224487305 seconds
# SVC 0.7472329703950329
# Running DecisionTreeClassifier classifier
# DecisionTreeClassifier prediction took 0.008718013763427734 seconds
# DecisionTreeClassifier 0.6594079006568884
# Running VotingClassifier classifier
# VotingClassifier prediction took 186.8331654071808 seconds
# VotingClassifier 0.7558715018446864

models = {
    'model': ['kNN', 'Logistic Reg', 'NB', 'SVM', 'Decision Tree', 'Random Forest', 'Voting Classifier'],
    'accuracy': [0.7300458921983263, 0.7507423737964546, 0.7466930621794295, 0.7472329703950329, 0.6594079006568884,
                 0.7448933681274184, 0.7558715018446864]
}

models_df = pd.DataFrame(models)
sns.barplot(x='model', y='accuracy', data=models_df.sort_values(by=['accuracy']), palette='Blues_d')
plt.title('Classification model accuracy')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.show()
print(models_df.sort_values(by=['accuracy']))
#                  model  accuracy
# 4        Decision Tree  0.659408
# 0                  kNN  0.730046
# 2                   NB  0.746693
# 3                  SVM  0.747233
# 1  Logistic Regression  0.750742
# 5    Voting Classifier  0.755872


if RUN_ALL_MODELS:
    find_optimal_k_for_knn()
    run_knn()
    train_svm()
    find_optimal_k_means()
    perform_k_means()

