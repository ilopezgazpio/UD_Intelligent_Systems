# Remove all the variables in the workspace
rm(list = ls())
# Clear console
cat("\014")

# Comment this lines after intalling the packages
#install.packages("gridExtra")
#install.packages("ggplot2")
#install.packages("rstudioapi")

# Load ggplot2 library
library(ggplot2)
library(gridExtra)

# Sets working directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Define the training set (10 examples)
data = data.frame(alt=c(T,T,F,T,T,F,F,F,F,T,F,T),
                  bar=c(F,F,T,F,F,T,T,F,T,T,F,T), # this is auxiliar
                  fri=c(F,F,F,T,T,F,F,F,T,T,F,T),
                  hun=c(T,T,F,T,F,T,F,T,F,T,F,T),
                  pat=c("Some","Full","Some","Full","Full","Some","None","Some","Full","Full","None","Full"),
                  price=c("$$$","$","$","$","$$$","$$","$","$$","$","$$$","$","$"),
                  rain=c(F,F,F,F,F,T,T,T,T,F,F,F),
                  res=c(T,F,F,F,T,T,F,T,F,T,F,F),
                  type=c("French","Thai","Burger","Thai","French","Italian","Burger","Thai","Burger","Italian","Thai","Burger"),
                  est=c("0-10","30-60","0-10","10-30",">60","0-10","0-10","0-10",">60","10-30","0-10","30-60"),
                  target=c(T,F,T,T,F,T,F,T,F,F,F,T))

# Define the structures to plot the distribution of the examples using each attribute
plot.alt <- ggplot(data, aes(x=alt, fill=target)) + geom_bar(position="stack") + labs(title="Alternate")
plot.bar <- ggplot(data, aes(x=bar, fill=target)) + geom_bar(position="stack") + labs(title="Bar")
plot.fri <- ggplot(data, aes(x=fri, fill=target)) + geom_bar(position="stack") + labs(title="Fri/Sat")
plot.hun <- ggplot(data, aes(x=hun, fill=target)) + geom_bar(position="stack") + labs(title="Hungry")
plot.pat <- ggplot(data, aes(x=pat, fill=target)) + geom_bar(position="stack") + labs(title="Patrons")
plot.price <- ggplot(data, aes(x=price, fill=target)) + geom_bar(position="stack") + labs(title="Price")
plot.rain <- ggplot(data, aes(x=rain, fill=target)) + geom_bar(position="stack") + labs(title="Rainning")
plot.res <- ggplot(data, aes(x=res, fill=target)) + geom_bar(position="stack") + labs(title="Reservation")
plot.type <- ggplot(data, aes(x=type, fill=target)) + geom_bar(position="stack") + labs(title="Type")
plot.est <- ggplot(data, aes(x=est, fill=target)) + geom_bar(position="stack") + labs(title="WaitEstimate")

# Plot examples distribution
grid.arrange(plot.alt, plot.bar, plot.fri, plot.hun, plot.pat, plot.price, plot.rain, plot.res, plot.type, plot.est, nrow=5)

# Define the entropy as a function
entropy = function (s) 
{
  # get the probability vector for each value on the feature
  p <- (s/sum(s))
  # get the log2 of probability
  logp <- log2(p)
  # 0 * log2(0) is 0 in entropy calculation
  logp[logp==-Inf] <- 0
  
  return(-1 * sum(p * logp))
}

# Defining information gain as a function
gain = function (attr, target) 
{
  values <- table(attr, target)
  # Get the vector for the feature values
  ps <- rowSums(values) / sum(values)
  # Get the entropy for target given each variable value
  es <- apply(values, 1, entropy)
  # IG = H(target) - H(Target | feature)
  gain <- entropy(colSums(values)) - sum(ps * es)
  return(gain)
}

# Let's calculate the entropy of the original dataset
values <- table(data$target)
p <- (values / sum(values))
logp <- log2(p)
sum(-p * logp)

# Extract pat values to calculate information gain of "pat" attribute
values = table(data$pat, data$target)

values
colSums(values)
rowSums(values)

pat.ps <- rowSums(values) / sum(values)
pat.es <- apply(values, 1, entropy)
# IG = H(target) - H(Target | patrons)
pat.gain <- entropy(colSums(values)) - sum(pat.ps * pat.es)
pat.gain

# Calculate information Gain for each attribute
print(paste0("Alternate gain=", gain(data$alt, data$target)))
print(paste0("Bar gain=", gain(data$bar, data$target)))
print(paste0("Fri/Sat gain=", gain(data$fri, data$target)))
print(paste0("Hungry gain=", gain(data$hun, data$target)))
print(paste0("Patrons gain=", gain(data$pat, data$target)))
print(paste0("Price gain=", gain(data$price, data$target)))
print(paste0("Raining gain=", gain(data$rain, data$target)))
print(paste0("Reservation gain=", gain(data$res, data$target)))
print(paste0("Type gain=", gain(data$type, data$target)))
print(paste0("WaitEstimate gain=", gain(data$est, data$target)))

