#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing

url = "https://raw.githubusercontent.com/lgazpio/UD_Intelligent_Systems/master/Datasets/people.csv"
data = pd.read_csv(url)
data.size
data.shape
data.columns
data.head()

data['custcat'].value_counts()
# 281 Plus Service, 266 Basic-service, 236 Total Service, and 217 E-Service customers

data.groupby(['age','income', 'custcat']).size().unstack()



X = data.drop("custcat", axis=1, inplace=False).values
X = preprocessing.StandardScaler().fit_transform(X.astype(float))
X.shape
data.shape

y = data['custcat'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=123)
print ('Train set: {} {}'.format(X_train.shape,  y_train.shape))
print ('Test set: {} {}'.format(X_test.shape,  y_test.shape))

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
k = 4
k4 = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
k4

yhat = k4.predict(X_test)

print("Train set Accuracy: ", metrics.accuracy_score(y_train, k4.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))

K_limit = 20
mean_acc = np.zeros((K_limit))
std_acc = np.zeros((K_limit))

for n in range(1,K_limit):
    current = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat = current.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)
    std_acc[n-1] = np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc.max()
mean_acc.argmax()

mean_acc = mean_acc[:-1]
std_acc = std_acc[:-1]
plt.plot(range(1,K_limit), mean_acc, 'g')
plt.fill_between(range(1,K_limit), mean_acc - 1 * std_acc, mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1,K_limit), mean_acc - 3 * std_acc, mean_acc + 3 * std_acc, alpha=0.10,color="green")
plt.legend(('Accuracy ', '+/- 1 std','+/- 3 std'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()