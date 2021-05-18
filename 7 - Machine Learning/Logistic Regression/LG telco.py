#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
import matplotlib.pyplot as plt

data = pd.read_csv("https://raw.githubusercontent.com/lgazpio/UD_Intelligent_Systems/master/Datasets/telco.csv")
data.head()
data.size
data.shape
data.columns

some_columns = ['tenure', 'age', 'address', 'income', 'ed', 'employ', 'churn']

data = data[some_columns]
data['churn'] = data['churn'].astype('int')
data.head()
data.size
data.shape
data.columns

X = np.asarray(data.drop("churn", axis=1, inplace=False))
X.shape

y = np.asarray(data['churn'])
y.shape

from sklearn import preprocessing
X = preprocessing.StandardScaler().fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set: {} {}'.format(X_train.shape,  y_train.shape))
print ('Test set: {} {}'.format(X_test.shape,  y_test.shape))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
LR

y_hat = LR.predict(X_test)
y_hat

yhat_prob = LR.predict_proba(X_test)
yhat_prob

from sklearn.metrics import classification_report, confusion_matrix
import itertools

def plot_confusion_matrix(cm,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



print(confusion_matrix(y_test, y_hat, labels=[1,0]))
cnf_matrix = confusion_matrix(y_test, y_hat, labels=[1,0])
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')
plt.show()
print (classification_report(y_test, y_hat))

