#!/usr/bin/python3

#---------------------------------------------------------------------------
# Linear Regression Concepts
# Intelligent Systems - University of Deusto
# Enrique Onieva Caracuel and transformed to Python by Inigo Lopez-Gazpio
#---------------------------------------------------------------------------
# This script aims to present the concept and functioning operation of
# linear regression
#---------------------------------------------------------------------------

#---------------------------------------------------------------------------
# PART 1: Preparation of the environment and libraries
#---------------------------------------------------------------------------

# We'll use scikit-learn, numpy, pandas and matplotlib libraries for Machine Learning projects
# These libraries are amongst the strongest ones these days for data scientists
# All of them can be installed through conda environments, pip or pip3

from math import *
import numpy as np
import matplotlib.pyplot as plt

#---------------------------------------------------------------------------
# PART 2: Definition of dataset and error measure
#---------------------------------------------------------------------------
# Define the training set (for 2 degrees)

trainingSet = np.arange(1, 20, 0.20)
bias = np.ones(trainingSet.size)
target = 2 * trainingSet - 3

plt.plot(trainingSet, target, linewidth = 5 , color = "k", label = "Real data")
plt.legend()
plt.show()

########################################################
# First attempt: Linear model is the mean of the curve
########################################################

LinearModel1 = np.mean(target)
plt.plot(trainingSet, target, linewidth = 5 , color = "k", label = "Real data")
plt.plot(trainingSet, LinearModel1.repeat(trainingSet.size), linewidth = 5 , color = "b", label = "Mean regression")
plt.legend()
plt.show()

# We can also compute the error of this approximation

def MSE (gold, predicted):
    return np.sum( np.square( gold - predicted)) / gold.size


MSE(target,  LinearModel1.repeat(trainingSet.size))

# Actually, the mean is a quite Naive approximation, despite the fact, it is used to compute the R^2 evaluation metric
def R2(gold, predicted):
    SS_residuals = gold - predicted
    SS_totals = gold - gold.mean()
    return 1 - (SS_residuals.dot(SS_residuals) / SS_totals.dot(SS_totals))

# Can you infer goodness of model according to R2 parameter? R2 parameter can be between -1 and 1


R2(target,  LinearModel1.repeat(target.size))
# R == 0 for a Naive mean regression

R2(target,  target)
# R == 1 for a perfect curve

R2(target, np.random.random(target.size))
# R <= 0 for a worse than naive curve

#########################################################
# Second attempt: Lets try a polynomial fit with python
#########################################################

LinearModel1 = np.mean(target)

degree = 1
LinearModel2 = np.polyfit(trainingSet, target, degree)

plt.plot(trainingSet, target, linewidth = 8 , color = "k", label = "Real data")
plt.plot(trainingSet, LinearModel1.repeat(trainingSet.size), linewidth = 5 , color = "b", label = "Mean regression")
plt.plot(trainingSet, np.poly1d(LinearModel2)(trainingSet), linewidth = 2 , color = "red", label = "Polyfit regression")

plt.legend()
plt.show()

# Seems like it is doing great job, lets compute MSE and R2
MSE(target, np.poly1d(LinearModel2)(trainingSet) )
R2(target,  np.poly1d(LinearModel2)(trainingSet))

# :)


###################################################################
# Third attempt: Lets try a regression based on equation systems
###################################################################

# Y = X W
# X_T Y = X_T X W
# (X_T X)^-1 X_T Y = W

trainingSetBias = np.append ( np.ones(trainingSet.size).reshape(trainingSet.size, 1) , trainingSet.reshape(trainingSet.size, 1) , axis = 1)

weights = np.linalg.solve( np.dot( trainingSetBias.T , trainingSetBias) , np.dot(trainingSetBias.T , target))
LinearModel3 = weights


plt.plot(trainingSet, target, linewidth = 8 , color = "k", label = "Real data")
plt.plot(trainingSet, LinearModel1.repeat(trainingSet.size), linewidth = 5 , color = "b", label = "Mean regression")
plt.plot(trainingSet, np.dot(trainingSetBias, weights), linewidth = 2 , color = "red", label = "Equation system regression")

plt.legend()
plt.show()

# Seems like it is doing great job, lets compute MSE and R2
MSE(target, np.dot(trainingSetBias, weights) )
R2(target, np.dot(trainingSetBias, weights) )

# This also rules!

# The problem with equation systems is we can not always compute # (X_T X)^-1, because not all matrices are invertible
# This is why we need other methods to train weights, such as: backpropagation


#---------------------------------------------------------------------------
# PART 3: Hypothesis in Linear Regression
#---------------------------------------------------------------------------
# In linear regression, the hypothesis is a model in the way h = w1*x + w0
# So, different values on w0 and w1 will provide different predictions (and errors)


weights = np.array((0,0))
predicted = np.dot( trainingSetBias , weights)
plt.plot(trainingSet, target, linewidth = 8 , color = "k", label = "Real data")
plt.plot(trainingSet, LinearModel1.repeat(trainingSet.size), linewidth = 5 , color = "b", label = "Mean regression")
plt.plot(trainingSet, np.dot(trainingSetBias, weights), linewidth = 2 , color = "red", label = "Weights = " + str(weights))
plt.legend()
plt.show()

weights = np.array((10,-1))
predicted = np.dot( trainingSetBias , weights)
plt.plot(trainingSet, target, linewidth = 8 , color = "k", label = "Real data")
plt.plot(trainingSet, LinearModel1.repeat(trainingSet.size), linewidth = 5 , color = "b", label = "Mean regression")
plt.plot(trainingSet, np.dot(trainingSetBias, weights), linewidth = 2 , color = "red", label = "Weights = " + str(weights))
plt.legend()
plt.show()

weights = np.array((-10,2.5))
predicted = np.dot( trainingSetBias , weights)
plt.plot(trainingSet, target, linewidth = 8 , color = "k", label = "Real data")
plt.plot(trainingSet, LinearModel1.repeat(trainingSet.size), linewidth = 5 , color = "b", label = "Mean regression")
plt.plot(trainingSet, np.dot(trainingSetBias, weights), linewidth = 2 , color = "red", label = "Weights = " + str(weights))
plt.legend()
plt.show()



#---------------------------------------------------------------------------
# PART 4: Coeficients adjustments
#---------------------------------------------------------------------------
# Define weights for the learning hypotesis and alpha (usually random)
weights0 = np.array((0,0))
alpha = 0.01

# Calculate the hypothesis using current weights and current error
predicted = np.dot( trainingSetBias , weights0)
delta_error = target - predicted

# Update weigths
weights[0] = weights0[0] + alpha * np.sum(delta_error * trainingSetBias[:,0]) / trainingSetBias.shape[0]
weights[1] = weights0[1] + alpha * np.sum(delta_error * trainingSetBias[:,1]) / trainingSetBias.shape[0]

plt.plot(trainingSet, target, linewidth = 8 , color = "k", label = "Real data")
plt.plot(trainingSet, np.dot(trainingSetBias, weights0), linewidth = 8 , color = "blue", label = "Weights = " + str(weights0))
plt.plot(trainingSet, np.dot(trainingSetBias, weights), linewidth = 8 , color = "red", label = "Weights = " + str(weights))
plt.legend()
plt.show()




# Now, we put the procedure into a loop
weights = np.array((0.0,0.0))
alpha = 0.005

for i in range(10000):
    weights0 = weights.copy()
    print("Old weights: {0:.3g} {0:3g}".format(weights0[0], weights0[1]))
    predicted = np.dot(trainingSetBias, weights0)
    delta_error = target - predicted

    iteration_error = MSE(target, predicted)
    print("MSE:" + str(iteration_error))

    # Update weigths
    weights[0] = weights0[0] + alpha * np.sum(delta_error * trainingSetBias[:, 0]) / trainingSetBias.shape[0]
    print(weights[0])
    weights[1] = weights0[1] + alpha * np.sum(delta_error * trainingSetBias[:, 1]) / trainingSetBias.shape[0]
    print(weights[1])
    print("New weights: {0:.3g} {0:.3g}".format(weights[0], weights[1]))

    plt.plot(trainingSet, target, linewidth=8, color="k", label="Real data")
    plt.plot(trainingSet, np.dot(trainingSetBias, weights), linewidth=8, color="red", label="Updated Weights = " + str(weights))
    plt.plot(trainingSet, np.dot(trainingSetBias, weights0), linewidth=8, color="blue", label="Old Weights = " + str(weights0))
    plt.legend()
    plt.title("Iteration: " + str(i) + " . MSE: " + str (iteration_error))
    plt.show()


    input()


# QUESTIONS to check: which is the effect of changing the alpha value?
# QUESTIONS to check: which is the effect of changing the inital w0 and w1 values?


#---------------------------------------------------------------------------
# PART 5: Coeficients adjustments (more complex hypothesis)
#---------------------------------------------------------------------------


trainingSetX1 = np.arange(1, 20, 0.75)
trainingSetX2 = np.square(trainingSetX1)
bias = np.ones(trainingSetX1.size)
trainingSet = np.concatenate( ( bias.reshape(trainingSetX1.size, 1), trainingSetX1.reshape(trainingSetX1.size, 1), trainingSetX2.reshape(trainingSetX1.size, 1)) , axis = 1)

target  = -1 * np.square(trainingSetX1) + trainingSetX1 + 3
target = target * np.random.rand()

linearModel1 = np.mean(target)
plt.plot(trainingSet[:,1], target, linewidth = 5 , color = "k", label = "Real data")
plt.plot(trainingSet[:,1], linearModel1.repeat(trainingSetX1.shape[0]), linewidth = 5 , color = "b", label = "Mean regression")
plt.legend()
plt.show()


# Define weights for the learning hypotesis and alpha
iterations = 10000
weights = np.array((0.0, 0.0, 0.0))
alpha = 0.00001

for i in range(100000):
    predicted = np.dot(trainingSet, weights)
    delta_error = target - predicted

    iteration_error = MSE(target, predicted)
    print("MSE:" + str(iteration_error))

    # Update weigths
    weights[0] = weights[0] + alpha * np.sum(delta_error * trainingSet[:, 0]) / trainingSet.shape[0]
    weights[1] = weights[1] + alpha * np.sum(delta_error * trainingSet[:, 1]) / trainingSet.shape[0]
    weights[2] = weights[2] + alpha * np.sum(delta_error * trainingSet[:, 2]) / trainingSet.shape[0]

plt.plot(trainingSet[:,1], target, linewidth=8, color="k", label="Real data")
plt.plot(trainingSet[:,1], np.dot(trainingSet, weights), linewidth=4, color="red", label="Updated Weights = " + str(weights))
plt.legend()
plt.show()

# QUESTION to check: change alpha = 0.0001 and iterations = 1000. What happens? why?.




#---------------------------------------------------------------------------
# PART 6: Linear Regression is (Fortunately) implemented in scikit-learn
#---------------------------------------------------------------------------

from sklearn.linear_model import LinearRegression
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# y = 1 * x_0 + 2 * x_1 + 3
y = np.dot(X, np.array([1, 2])) + 3
reg = LinearRegression().fit(X, y)
reg.score(X, y)
reg.coef_
reg.intercept_
reg.predict(np.array([[3, 5]]))

# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html