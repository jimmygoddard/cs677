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

# sns.pairplot(df)
# plt.show()

year = df['year'].dropna().astype(int)
sns.distplot(year)
plt.title('Year Histogram')
plt.ylabel('Probability')
plt.xlabel('Year')
plt.show()

marvel_year = df.year[df.publisher == 'Marvel']
dc_year = df.year[df.publisher == 'DC']

fig, ax = plt.subplots()
sns.distplot(dc_year.dropna(), ax=ax, label='DC')
sns.distplot(marvel_year.dropna(), ax=ax, label='Marvel')
plt.title('Year Histogram By Publisher')
plt.ylabel('Probability')
plt.xlabel('Year')
plt.legend()
plt.show()


sns.catplot(x='year', y='sex', data=df)
plt.show()

sns.catplot(x='year', y='sex', data=df, kind='boxen')
plt.show()

sns.catplot(x='year', y='sex', data=df, hue='gsm')
plt.show()

g = sns.countplot(x=year)
labels = g.get_xticklabels()
for i, l in enumerate(labels):
    if i % 5 != 0:
        labels[i] = ''
g.set_xticklabels(labels, rotation=90)
plt.show()

g = sns.catplot(x='year', data=df, kind='count', hue='publisher', legend=False)
for ax in g.axes.flat:
    labels = ax.get_xticklabels()
    for i, l in enumerate(labels):
        if i % 5 != 0:
            labels[i] = ''
    ax.set_xticklabels(labels, rotation=90)
plt.legend(loc='upper left')
plt.title('')
plt.show()

# gender analytics
sex_year_df = df[['year', 'sex']]
sex_year_df_grouped = sex_year_df.groupby(['year', 'sex']).size().reset_index(name='counts')

# number of each type of gender plotted per year, might want to also show just male/female
sns.lineplot(x='year', y='counts', hue='sex', data=sex_year_df_grouped)
plt.show()

# maybe also show the same type of plot, but count id (secret, public, etc.) hair color, eye color

# can do appearances by gender too
app_by_gender = df[['']]
