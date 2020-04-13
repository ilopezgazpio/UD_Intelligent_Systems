#---------------------------------------------------------------------------
# Linear Regression Concepts
# Intelligent Systems - University of Deusto
# Enrique Onieva Caracuel
#---------------------------------------------------------------------------
# This script aims to present the concept and functioning operation of 
# linear regression
#---------------------------------------------------------------------------

#---------------------------------------------------------------------------
# PART 1: Preparation of the enviroment and libraries
#---------------------------------------------------------------------------
rm(list=ls())
cat("\014")
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()
dir()

library(ggplot2)

#---------------------------------------------------------------------------
# PART 2: Definition of dataset and error measure
#---------------------------------------------------------------------------
# Define the training set (for 2 degrees)
training.set = data.frame(x1=seq(1, 20, by = 0.25))
training.set$x0 = 1 # this is auxiliar
training.set$y  = 2 * training.set$x1 - 3

ggplot(training.set, aes(x = x1, y = y)) + geom_point()

# Imagine, we provide a prediction (or hypothesis). Mean value in this case
training.set$h = mean(training.set$y)

# So, we can calculate error in each point
training.set$e = training.set$y - training.set$h

head(training.set)

ggplot(training.set) + 
  geom_point(aes(x=x1, y=y, col="Data")) + 
  geom_point(aes(x=x1, y=h, col="Hypothesis")) +geom_line(aes(x=x1, y=h, col = "Hypothesis"))


#---------------------------------------------------------------------------
# PART 3: Hypothesis in Linear Regression
#---------------------------------------------------------------------------
# In linear regression, the hypothesis is a model in the way h = w1*x + w0
# So, different values on w0 and w1 will provide different predictions (and errors)
w0 = 0
w1 = 0
training.set$h = training.set$x1 * w1 + training.set$x0 * w0
ggplot(training.set) + 
  geom_point(aes(x=x1, y=y, col="Data")) + 
  geom_point(aes(x=x1, y=h, col="Hypothesis"))  + geom_line(aes(x=x1, y=h, col = "Hypothesis")) +
  labs(caption=paste0("w1=", round(w1, digits=3),"  w0=", round(w0, digits=3)))

w0 = 10
w1 = -1
training.set$h = training.set$x1 * w1 + training.set$x0 * w0
ggplot(training.set) + 
  geom_point(aes(x=x1, y=y, col="Data")) + 
  geom_point(aes(x=x1, y=h, col="Hypothesis"))  + geom_line(aes(x=x1, y=h, col = "Hypothesis")) +
  labs(caption=paste0("w1=", round(w1, digits=3),"  w0=", round(w0, digits=3)))

w0 = -10
w1 = 2.5
training.set$h = training.set$x1 * w1 + training.set$x0 * w0
ggplot(training.set) + 
  geom_point(aes(x=x1, y=y, col="Data")) + 
  geom_point(aes(x=x1, y=h, col="Hypothesis"))  + geom_line(aes(x=x1, y=h, col = "Hypothesis")) +
  labs(caption=paste0("w1=", round(w1, digits=3),"  w0=", round(w0, digits=3)))



#---------------------------------------------------------------------------
# PART 4: Coeficients adjustments
#---------------------------------------------------------------------------
# Define weights for the learning hypotesis and alpha (usually random)
w0 = 0
w1 = 0
alpha = 0.01

# Calculate the hypothesis using current weights and current error
training.set$h = training.set$x1 * w1 + training.set$x0 * w0
training.set$e = training.set$y - training.set$h
head(training.set)

# Update weigths 
w1 = w1 + alpha * sum(training.set$e * training.set$x1) / nrow(training.set)
w0 = w0 + alpha * sum(training.set$e * training.set$x0) / nrow(training.set)

# Plot resulting hypothesis (for visualization purposes)
ggplot(training.set) + 
  geom_point(aes(x=x1, y=y, col="Data")) + 
  geom_point(aes(x=x1, y=h, col="Hypothesis")) + geom_line(aes(x=x1, y=h, col = "Hypothesis")) +
  labs(caption=paste0("w1=", round(w1, digits=3), "  w0=", round(w0, digits=3)))


# Now, we put the procedure into a loop
iterations = 10000
w0 = 10
w1 = 10
alpha = 0.01
# training.set$y = training.set$y + runif(nrow(training.set), 0, 4)
for (i in 1:iterations) {
  w1 = w1 + alpha * sum(training.set$e * training.set$x1) / nrow(training.set)
  w0 = w0 + alpha * sum(training.set$e * training.set$x0) / nrow(training.set)
  training.set$h = training.set$x1 * w1 + training.set$x0 * w0
  training.set$e = training.set$y - training.set$h
  print(paste0(i, " w1=", round(w1, digits=3),
               "  w0=", round(w0, digits=3),
               "  error=", round(mean(abs(training.set$e)), digits=3)))
}

# Plot data and hypothesis after finishing the learning process
ggplot(training.set) + 
  geom_point(aes(x=x1, y=y, col="Data")) + 
  geom_point(aes(x=x1, y=h, col="Hypothesis")) + geom_line(aes(x=x1, y=h, col = "Hypothesis")) +
  labs(title=paste0("Iteration ", i), caption=paste0("w1=", round(w1, digits=3),"  w0=", round(w0, digits=3)))

# QUESTIONS to check: which is the effect of changing the alpha value?
# QUESTIONS to check: which is the effect of changing the inital w0 and w1 values?
# QUESTION to check: uncomment line after "alpha = 0.01" and run the process. Comment results.




#---------------------------------------------------------------------------
# PART 5: Coeficients adjustments (more complex hypothesis)
#---------------------------------------------------------------------------
training.set = data.frame(x1=rep(seq(1, 20, by = 0.75),3))
training.set$x0 = 1 # this is auxiliar
training.set$x2 = training.set$x1 ^ 2 # this is auxiliar
training.set$y  = -training.set$x1 ^ 2 + training.set$x1 + 3
training.set$y = training.set$y * runif(nrow(training.set), 0.8, 1.2)

training.set$h = mean(training.set$y)

head(training.set)

ggplot(training.set) + 
  geom_point(aes(x=x1, y=y, col="Data")) + 
  geom_point(aes(x=x1, y=h, col="Hypothesis")) + geom_line(aes(x=x1, y=h, col = "Hypothesis"))


# Define weights for the learning hypotesis and alpha
iterations = 10000
w0 = 0
w1 = 0
w2 = 0
alpha = 0.0000001

# Calculate the hypothesis using current weights and current error
training.set$h = training.set$x2 * w2 +training.set$x1 * w1 + training.set$x0 * w0
training.set$e = training.set$y - training.set$h

for (i in 1:iterations){
  # Update the weights
  w2 = w2 + alpha * sum(training.set$e * training.set$x2)#/nrow(training.set)
  w1 = w1 + alpha * sum(training.set$e * training.set$x1)#/nrow(training.set)
  w0 = w0 + alpha * sum(training.set$e * training.set$x0)#/nrow(training.set)
  # Update the hypothesis value and the error
  training.set$h = training.set$x2 * w2 +training.set$x1 * w1 + training.set$x0 * w0
  training.set$e = training.set$y - training.set$h
  # Print the evolution of the weigths and error during the learnig process
  print(paste0(i,"  w2=",round(w2, digits=3),
               "  w1=",round(w1, digits=3),
               "  w0=",round(w0, digits=3),
               "  error=",round(mean(abs(training.set$e)), digits=3)))
}

ggplot(training.set) + 
  geom_point(aes(x=x1, y=y, col = "Data")) + 
  geom_point(aes(x=x1, y=h, col = "Hypothesis")) + geom_line(aes(x=x1, y=h, col = "Hypothesis")) +
  labs(title=paste0("Iteration ",i), 
       caption=paste0("w2=", round(w2, digits=3),"  w1=", round(w1, digits=3),"  w0=", round(w0, digits=3)))

# QUESTION to check: change alpha = 0.0001 and iterations = 1000. What happens? why?.




#---------------------------------------------------------------------------
# PART 6: Linear Regression is (Fortunately) implemented in R
#---------------------------------------------------------------------------
model1 = lm(y ~ x1, training.set)
model1

model2 = lm(y ~ poly(x1, 5), training.set)
model2

length(unique(training.set$x1))
training.set$h = predict(model2, training.set)

ggplot(training.set) + 
  geom_point(aes(x=x1, y=y, col = "Data")) + 
  geom_point(aes(x=x1, y=h, col = "Hypothesis")) + geom_line(aes(x=x1, y=h, col = "Hypothesis")) +
  labs(caption=paste0("w2=", round(w2, digits=3),"  w1=", round(w1, digits=3),"  w0=", round(w0, digits=3)))

# QUESTION to check: try a polynomal with degree = 15. Comment results