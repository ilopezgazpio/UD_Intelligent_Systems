#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv("https://raw.githubusercontent.com/lgazpio/UD_Intelligent_Systems/master/Datasets/cancer.csv")
data.size
data.shape
data.head()
data.columns

ax = data[data['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='Malignant cancer');
data[data['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='Benign cancer', ax=ax);
plt.show()

data.dtypes
data = data[pd.to_numeric(data['BareNuc'], errors='coerce').notnull()]
data['BareNuc'] = data['BareNuc'].astype('int')
data.dtypes

X = data.drop("ID", axis=1,inplace=False).drop("Class", axis=1, inplace=False)
X = np.asarray(X)
X.size
X.shape
X.head()

data['Class'].value_counts().to_frame()
data['Class'] = data['Class'].astype('int')
y = np.asarray(data['Class'])

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=123)
print ('Train set: {} {}'.format(X_train.shape,  y_train.shape))
print ('Test set: {} {}'.format(X_test.shape,  y_test.shape))

from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train) 
y_hat = clf.predict(X_test)

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

cnf_matrix = confusion_matrix(y_test, y_hat, labels=[2,4])
np.set_printoptions(precision=2)

print(classification_report(y_test, y_hat))
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Benign(2)','Malignant(4)'],normalize= False,  title='Confusion matrix')
plt.show()
