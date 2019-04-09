import os
import platform

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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

sns.pairplot(df)
plt.show()

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


sns.catplot(x='year', y='sex', data=df)
plt.title('New Heroes per Year by Gender')
plt.show()

sns.catplot(x='year', y='sex', data=df, kind='boxen')
plt.title('New Heroes per Year by Gender')
plt.show()

sns.catplot(x='year', y='sex', data=df, hue='gsm')
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

g = sns.catplot(x='year', data=df, kind='count', hue='publisher', legend=False)
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

male_df = df[['year', 'sex']][df.sex == 'Male Characters']
female_df = df[['year', 'sex']][df.sex == 'Female Characters']
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
app_by_gender_grouped = df.groupby(['sex'])['appearances'].sum().reset_index()

# appearances by sexuality
app_by_gsm = df.groupby(['gsm'])['appearances'].sum().reset_index()

# build classifiers to classify gender:
# 1) a knn - https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
# 2) neural net classifier - https://medium.com/datadriveninvestor/building-neural-network-using-keras-for-classification-3a3656c726c1
# 3) naive bayesian - https://scikit-learn.org/stable/modules/naive_bayes.html

