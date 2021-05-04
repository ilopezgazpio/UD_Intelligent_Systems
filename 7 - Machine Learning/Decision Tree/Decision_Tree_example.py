#!/usr/bin/python3

#-------------------------------------------
# Decision Trees Concepts
# Intelligent Systems - University of Deusto
# Inigo Lopez-Gazpio
#-------------------------------------------

#----------------------------------
# PART 1: Environment and libraries
#----------------------------------
# We'll use scikit-learn, numpy, pandas and matplotlib libraries for Machine Learning projects
# These libraries are amongst the strongest ones for data scientists
# All of them can be installed through conda environments, pip or pip3

from math import *
epsilon = -1e-100
#---------------------------------------------------------------------------
# PART 2: Formulas
#---------------------------------------------------------------------------

# Entropy of a dataset with 50-50 elements of different classes (worst case, maximum entropy)
entropy = -0.50 * log2(0.50) - 0.50 * log2(0.50)
print(entropy)

# Entropy of a dataset with 100-0 elements of different classes (best case, minimum entropy)
entropy = epsilon * log2(epsilon) - 1.00 * log2(1.00)
print(entropy)

# Entropy of intermediate distributions
entropy = -0.75*log2(0.75) - 0.25*log2(0.25)
print(entropy)

entropy = -0.01*log2(0.01) - 0.99*log2(0.99)
print(entropy)

# Defining a function
def entropy (a, b):
  total = a + b
  prob_a = a / total
  prob_b = b / total
  if prob_a == 0 or prob_b == 0:
      return 0
  else:
      return -prob_a * log2(prob_a) - prob_b * log2(prob_b)

# Imagine we have an initial datase with 10 and 10 elements
entropy(10, 10)

# We can split into a 7-3 and 3-7 datasets... and start computing information gain
# IG = H(class) - H ( class | attributes)

gain1 = entropy(10, 10) - ( (10/20) * entropy(3,7) + (10/20) * entropy(7,3) )
print(gain1)

# We can split into a 1-9 and 9-1 datasets...
gain2 = entropy(10,10) - ( (10/20) * entropy(1,9) + (10/20) * entropy(9,1) )
print(gain2)

# We can split into a 9-9 and 1-1 datasets...
gain3 = entropy(10,10) - ( (18/20) * entropy(9,9) + (2/20) * entropy(1,1) )
print(gain3)

# We can split into a 9-3 and 1-7 datasets...
gain4 = entropy(10,10) - ( (12/20) * entropy(9,3) + (8/20) * entropy(1,7) )
print(gain4)

# We can split into a 10-0 and 0-10 datasets...
gain5 = entropy(10,10) - ( (10/20) * entropy(10,0) + (10/20) * entropy(0,10) )
print(gain5)

# Entropy is implemented in scikit-learn... https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html


#---------------------------------------------------------------------------
# PART 3: (General porpuses) Formulas
#---------------------------------------------------------------------------

# Define the entropy as a function able to receive any set of partitions

import numpy as np
import pandas as pd
import requests

def entropy (s : np.array):
    # s is a numpy array with counts per class
    probs = s / np.sum(s)
    logprobs = np.log2(probs)
    logprobs[logprobs == np.inf * -1 ] = 0
    return sum(-1 * probs * logprobs)


# Defining information gain as a function able to receive any set of partitions
def gain (dataframe : pd.DataFrame, attr : str, target : str):
  values = dataframe.groupby([attr, target]).size().unstack().values
  values = np.nan_to_num(values)

  # to compute class entropy H(class)
  class_variable_counts = np.sum(values, axis = 0)

  # class given value entropy H(class | attribute)
  attribute_variable_counts = np.sum(values, axis=1)
  attribute_variable_probs = attribute_variable_counts / np.sum(values)
  entropy_given_attribute  = np.apply_along_axis(entropy, 1, values)

  return entropy(class_variable_counts) - np.sum(attribute_variable_probs * entropy_given_attribute)


url = "https://raw.githubusercontent.com/lgazpio/UD_Intelligent_Systems/master/Datasets/agaricus-lepiota.csv"
#source = requests.get(url).content
data = pd.read_csv(url)
data.columns = [
    "class",
    "cap.shape",
    "cap.surface",
    "cap.color",
    "bruises",
    "odor",
    "gill.attachment",
    "gill.spacing",
    "gill.size",
    "gill.color",
    "stalk.shape",
    "stalk.root",
    "stalk.surface.above.ring",
    "stalk.surface.below.ring",
    "stalk.color.above.ring",
    "stalk.color.below.ring",
    "veil.type",
    "veil.color",
    "ring.number",
    "ring.type",
    "spore.print.color",
    "population",
    "habitat"
]

data.size
data.head()
data.columns

# Which is the attribute with more "gain"?
gain(data, "cap.shape", "class")
gain(data, "ring.type", "class")
gain(data, "cap.color", "class")
gain(data, "odor", "class")


# For each partition we should keep on with this process recursively...
# If data is categorical instead of numerical we need to transform it to numbers by discretizing...

# Mutual information classifier from scikit-learn performs similar job
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html#sklearn.feature_selection.mutual_info_classif

#---------------------------------------------------------------------------
# PART 4: Decision tree is (Fortunately) implemented in Python
#---------------------------------------------------------------------------
# https://scikit-learn.org/stable/modules/tree.html

from sklearn import tree
import matplotlib.pyplot as plt

data_onehot = pd.get_dummies( data )
trainingSet = data_onehot.values[:,2:]
trainingSet.shape

labels = data.values[:,0]
labels.shape

clf = tree.DecisionTreeClassifier()
clf = clf.fit(trainingSet, labels)

tree.plot_tree(clf.fit(trainingSet, labels))
plt.show()

# export with graphviz
import graphviz
tree_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(tree_data)
graph.render("Decision_tree_example")