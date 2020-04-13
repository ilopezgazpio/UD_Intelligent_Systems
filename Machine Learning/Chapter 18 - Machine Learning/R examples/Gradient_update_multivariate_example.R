# Remove all the variables in the workspace
rm(list = ls())
# Clear console
cat("\014")

# Comment this lines after intalling the packages
#install.packages("ggplot2")
#install.packages("rstudioapi")

# Load ggplot2 library
library(ggplot2)

# Sets working directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# # Define the training set (for 2 degrees)
training.set <- data.frame(x1=seq(1:20))
training.set$x0 <- 1 # this is auxiliar
training.set$x2 <- training.set$x1 ^ 2 # this is auxiliar
training.set$y  <- -training.set$x1 ^ 2 + training.set$x1 + 3

# Define weights for the learning hypotesis and alpha
w0 <- 0
w1 <- 0
w2 <- 0
alpha <- 0.00001

# Calculate the hypothesis using current weights
training.set$h <- training.set$x2 * w2 + training.set$x1 * w1 + training.set$x0 * w0
# Calculate the current error
training.set$e <- training.set$y - training.set$h

# Show training set in the command line
training.set

# Plot data and hypothesis before applaying Linear Regression
ggplot(training.set) + 
  geom_point(aes(x=x1, y=y, col="Data")) + 
  geom_point(aes(x=x1, y=h, col="Hypothesis")) + 
  labs(title=paste0("Initial Situation"), 
       caption=paste0("w2=", round(w2, digits=3),"  w1=", round(w1, digits=3),"  w0=", round(w0, digits=3)))

# Update weigths one time
w2 <- w2 + alpha * sum(training.set$e * training.set$x2)/nrow(training.set)
w1 <- w1 + alpha * sum(training.set$e * training.set$x1)/nrow(training.set)
w0 <- w0 + alpha * sum(training.set$e * training.set$x0)/nrow(training.set)

# Calculate again the hypothesis and the error after updating the weights
training.set$h <- training.set$x2 * w2 + training.set$x1 * w1 + training.set$x0 * w0
training.set$e <- training.set$y - training.set$h

# Show training set in the command line
training.set

# Plot data and hypothesis after updating the weights one time
ggplot(training.set) + 
  geom_point(aes(x=x1, y=y, col="Data")) + 
  geom_point(aes(x=x1, y=h, col="Hypothesis")) + 
  labs(title=paste0("After 1 update"), 
       caption=paste0("w2=", round(w2, digits=3), "  w1=", round(w1, digits=3),"  w0=", round(w0, digits=3)))

# Set the number of iterations for the learning process
iterations <- 1000

for (i in 1:iterations){
  # Update the weights
  w2 <- w2 + alpha * sum(training.set$e * training.set$x2)/nrow(training.set)
  w1 <- w1 + alpha * sum(training.set$e * training.set$x1)/nrow(training.set)
  w0 <- w0 + alpha * sum(training.set$e * training.set$x0)/nrow(training.set)
  # Update the hypothesis value and the error
  training.set$h <- training.set$x2 * w2 +training.set$x1 * w1 + training.set$x0 * w0
  training.set$e <- training.set$y - training.set$h
  # Print the evolution of the weigths and error during the learnig process
  print(paste0(i,"  w2=",round(w2, digits=3),
               "  w1=",round(w1, digits=3),
               "  w0=",round(w0, digits=3),
               "  error=",round(mean(abs(training.set$e)), digits=3)))
}

# Plot data and hypothesis after finishing the learning process
ggplot(training.set) + 
  geom_point(aes(x=x1, y=y, col = "Data")) + 
  geom_point(aes(x=x1, y=h, col = "Hypothesis")) + 
  labs(title=paste0("Iteration ",i), 
       caption=paste0("w2=", round(w2, digits=3),"  w1=", round(w1, digits=3),"  w0=", round(w0, digits=3)))

