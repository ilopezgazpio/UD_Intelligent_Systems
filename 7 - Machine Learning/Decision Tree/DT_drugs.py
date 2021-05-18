#!/usr/bin/env python
# coding: utf-8

# Example to import the Scikit Learn Decision-Tree module

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Parse data using pandas dataframe:

url = "https://raw.githubusercontent.com/lgazpio/UD_Intelligent_Systems/master/Datasets/drug200.csv"
data = pd.read_csv("https://raw.githubusercontent.com/lgazpio/UD_Intelligent_Systems/master/Datasets/drug200.csv", delimiter=",")
data.size
data.head()
data.columns
data.mean(axis = 0, numeric_only=True)
data.groupby(["Age","Sex"]).size()
data.groupby(["Age","Sex"]).size().unstack()

# Get parts of the file for training
columns_of_interest = ['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']
X = data[columns_of_interest].values
X.shape
X.size

# Categorical to numerical transformation
from sklearn import preprocessing
num_sex = preprocessing.LabelEncoder()
num_sex.fit(['F','M'])
X[:,1] = num_sex.transform(X[:,1])
X[:5,1]

num_BP = preprocessing.LabelEncoder()
num_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = num_BP.transform(X[:,2])
X[:5,2]

num_Chol = preprocessing.LabelEncoder()
num_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = num_Chol.transform(X[:,3])
X[:5,3]

# Set the targets for supervised classification
y = data["Drug"]
y[0:5]

# Cross validation training
from sklearn.model_selection import train_test_split
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

tree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
print(tree)

tree.fit(X_trainset, y_trainset)
y_hat = tree.predict(X_testset)

# Evaluation

# By hand :)
accuracy = sum(y_hat == y_testset.values) / y_hat.size
accuracy

matrix = (y_hat == y_testset).value_counts().to_frame().unstack()
matrix

# Contingency matrix
cm = pd.concat( [pd.Series(y_hat) , pd.Series(y_testset.values)] , axis=1 )
cm.groupby([0,1]).size().unstack()

# Using scikit-learn goodness
from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: {}".format(metrics.accuracy_score(y_testset, y_hat)))

