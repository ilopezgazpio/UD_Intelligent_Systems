# Tutorial from https://towardsdatascience.com/going-from-stata-to-pandas-706888525acf
# Common Stata operations in python

import pandas as pd

auto = 'http://www.stata-press.com/data/r15/auto2.dta'
df = pd.read_stata(auto)

# Visualizations
df.head()
df.columns

# Cross-tabulation
df['rep78'].unique()
df['foreign'].unique()
df['foreign'].cat.categories

pd.crosstab(df['rep78'], df['foreign'])
df.groupby(["rep78","foreign"]).size()

# To convert NA values into an extra category we need 2 extra lines
df['rep78'].cat.add_categories('No Record', inplace=True)
df['rep78'] = df['rep78'].fillna('No Record')

# Summary statistics
df.describe().transpose()


# More examples
df[['price','mpg','foreign']].groupby('foreign').describe()


# Basic plots
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

auto = 'http://www.stata-press.com/data/r15/auto2.dta'
df = pd.read_stata(auto)
sns.jointplot(x='weight',y='mpg',data=df, kind='scatter')
plt.show()


sns.pairplot(df[['mpg','weight','length','price']])
plt.show()


sns.pairplot(df[['mpg','weight','length','price','foreign']],
             kind='scatter', plot_kws={'alpha': 0.5},
             hue='foreign')
plt.show()



# Simple Linear regression in Python
import pandas as pd
from sklearn.linear_model import LinearRegression

auto = 'http://www.stata-press.com/data/r15/auto2.dta'
df = pd.read_stata(auto)

numerical_features = df.select_dtypes(['number']).columns
categorical_features = df.columns.difference(numerical_features)

X_train = df.drop(categorical_features, axis=1, inplace=False).drop("price", axis=1, inplace=False)

lr = LinearRegression(normalize=True)
lr.fit(X_train, df['price'])

df['yhat'] = lr.predict(X_train)
df['resid'] = df.price - df.yhat

df.head(50)['price yhat resid'.split(' ')]


# More complex regression tests

# econometric tools : http://www.danielmsullivan.com/econtools/#data-manipulation-tools

# linear models : https://bashtage.github.io/linearmodels

# statsmodels : https://www.statsmodels.org/stable/index.html


# Generalized Method of Moments Estimation and Instrumental Variable (IV) regression in Python
from statsmodels.sandbox.regression import gmm
# The main models that are currently available are: GMM, IV2SLS, IVGMM, LinearIVGMM, NonlinearIVGMM