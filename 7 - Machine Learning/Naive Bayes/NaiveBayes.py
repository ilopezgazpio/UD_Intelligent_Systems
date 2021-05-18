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
import pandas as pd
import numpy as np

epsilon = -1e-100

url = "https://raw.githubusercontent.com/lgazpio/UD_Intelligent_Systems/master/Datasets/restaurants.csv"
data = pd.read_csv(url)

data.size
data.shape
data.head()
data.columns

# Naive Bayes is a generative supervised classifier that predict Probability of the class attribute given features (Posterior)
# It takes into account the probability of the class (Prior) and the conditional probability of the features given the class (likelihood)

# Can be resumed as P(Class | features) = P(Class) * P(Features | Class), this equation can be derived from the Bayes theorem

# Computing the prior
# P(Class = True)
# P(Class = False)
data['target_wait'].value_counts()
data.shape[0]

# Computing likelihood
data.groupby(["alternate", "target_wait"]).size()
# P(alternate = False | Class = True)
# P(alternate = True | Class = True)
# P(alternate = False | Class = True)
# P(alternate = True | Class = True)

data.groupby([data.columns[0], "target_wait"]).size()

# Important note: we are treating each feature as independently distributed from the rest of the features! which might not be true
# this is known as the Markov Assumption and it gives raise to the Naive Bayes classification model
# We can use more complicated conditional distributions and implement a non-Naive Bayes classifier


# Finally, the posterior P(Class = ? | features) is computed as a generative probabilistic model and argmax is taken

# P(Class = True | features) = P(Class = True) * P(Feature1 | Class = True) * P(Feature2 | Class = True) ....
# P(Class = False | features) = P(Class = False) * P(Feature1 | Class = False) * P(Feature2 | Class = False) ....

# More on https://scikit-learn.org/stable/modules/naive_bayes.html

from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing

le_patrons = preprocessing.LabelEncoder().fit(data["patrons"])
le_type = preprocessing.LabelEncoder().fit(data["type"])
le_price = preprocessing.LabelEncoder().fit(data["price"])
le_wait_estimate = preprocessing.LabelEncoder().fit(data["wait_estimate"])

data["patrons"] = le_patrons.transform(data["patrons"])
data["type"] = le_type.transform(data["type"])
data["price"] =  le_price.transform(data["price"])
data["wait_estimate"] = le_wait_estimate.transform(data["wait_estimate"])

gnb = GaussianNB()
gnb.fit(data.drop("target_wait", axis=1, inplace=False), data["target_wait"])

new_sample = {
    'alternate': False,
    'bar': True,
    'party_day': True,
    'hungry': True,
    'patrons': le_patrons.transform(["Full"]),
    'price': le_price.transform(["$"]),
    'raining': True,
    'reservation': False,
    'type': le_type.transform(['Thai']),
    'wait_estimate': le_wait_estimate.transform(['>60'])
}

gnb.predict(pd.Series(data=new_sample).values.reshape(1,-1))

gnb.class_count_
gnb.class_prior_
gnb.theta_
gnb.sigma_