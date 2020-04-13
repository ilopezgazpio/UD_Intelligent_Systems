#---------------------------------------------------------------------------
# Decision Trees Concepts
# Intelligent Systems - University of Deusto
# Enrique Onieva Caracuel
#---------------------------------------------------------------------------
# This script aims to present the concept and functioning operation of 
# Decision Trees
#---------------------------------------------------------------------------

#---------------------------------------------------------------------------
# PART 1: Preparation of the environment and libraries
#---------------------------------------------------------------------------
rm(list=ls())
cat("\014")
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()
dir()

library(mlbench)
library(dplyr)
library(rpart)

#---------------------------------------------------------------------------
# PART 2: Formulas
#---------------------------------------------------------------------------
# Entropy of a dataset with 50-50 elements of different classes (worst case)
entropy = -0.50*log2(0.50) - 0.50*log2(0.50)

# Entropy of a dataset with 100-0 elements of different classes (best case)
entropy = -0.00*log2(0.00) - 1.00*log2(1.00)
# (needed to be changed by 0 since log2(0.00)=-INF)

# Entropy of intermediate distributions
entropy = -0.75*log2(0.75) - 0.25*log2(0.25)
entropy = -0.01*log2(0.01) - 0.99*log2(0.99) 

# Defining a function
entropy = function (a,b)
{
  total = a+b
  a = a / total
  b = b / total
  ent = -a * log2(a) - b * log2(b)
  if (is.nan(ent)){ent = 0}
  return(ent)
}

# Imagine we have an initial datase with 10 and 10 elements
entropy(10,10)

# We can split into a 7-3 and 3-7 datasets...
gain1 = entropy(10,10) - ((10/20)*entropy(3,7) + (10/20)*entropy(7,3))

# We can split into a 1-9 and 9-1 datasets...
gain2 = entropy(10,10) - ((10/20)*entropy(1,9) + (10/20)*entropy(9,1))

# We can split into a 9-9 and 1-1 datasets...
gain3 = entropy(10,10) - ((18/20)*entropy(9,9) + (2/20)*entropy(1,1))

# We can split into a 9-3 and 1-7 datasets...
gain4 = entropy(10,10) - ((12/20)*entropy(9,3) + (8/20)*entropy(1,7))

# We can split into a 10-0 and 0-10 datasets...
gain5 = entropy(10,10) - ((10/20)*entropy(10,0) + (10/20)*entropy(0,10))

# QUESTION: Which one to pick

#---------------------------------------------------------------------------
# PART 3: (General porpuses) Formulas
#---------------------------------------------------------------------------

# Define the entropy as a function able to receive any set of partitions
entropy = function (s) 
{
  p  = (s/sum(s))
  logp  = log2(p)
  logp[logp==-Inf]  = 0
  return(sum(-p * logp))
}

# Defining information gain as a function able to receive any set of partitions
gain = function (attr, target) 
{
  values  = table(attr, target)
  ps  = rowSums(values) / sum(values)
  es  = apply(values, 1, entropy)
  gain  = entropy(colSums(values)) - sum(ps * es)
  
  return(gain)
}



# Import data from UCI ML repo
theURL = "http://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
# Explicitly adding the column headers from the data dictionary
data =  read.csv(file = theURL, header = FALSE, sep = ",",strip.white = TRUE,
                         stringsAsFactors = TRUE, 
                         col.names = c("class","cap-shape","cap-surface","cap-color","bruises",
                                       "odor","gill-attachment","gill-spacing","gill-size",
                                       "gill-color","stalk-shape","stalk-root","stalk-surface-above-ring",
                                       "stalk-surface-below-ring","stalk-color-above-ring","stalk-color-below-ring",
                                       "veil-type","veil-color","ring-number","ring-type","spore-print-color",
                                       "population","habitat"))

data.levels = cbind.data.frame(Variable=names(data), Total_Levels=sapply(data,function(x){as.numeric(length(levels(x)))}))
print(data.levels)

# Explicitly changing levels names from the data dictionary
levels(data$class) = c("edible","poisonous")
levels(data$cap.shape) =c("bell","conical","flat","knobbed","sunken","convex") 
levels(data$cap.surface) = c("fibrous","grooves","smooth","scaly")
levels(data$cap.color) = c("buff","cinnamon","red","gray","brown","pink","green","purple","white","yellow")
levels(data$bruises) = c("bruisesno","bruisesyes")
levels(data$odor) =c("almond","creosote","foul","anise","musty","nosmell","pungent","spicy","fishy")
levels(data$gill.attachment) = c("attached","free")
levels(data$gill.spacing) = c("close","crowded")
levels(data$gill.size) =c("broad","narrow")
levels(data$gill.color) = c("buff","red","gray","chocolate","black","brown","orange","pink","green","purple","white","yellow")
levels(data$stalk.shape) = c("enlarging","tapering")
levels(data$stalk.root) = c("missing","bulbous","club","equal","rooted")
levels(data$stalk.surface.above.ring) =c("fibrous","silky","smooth","scaly")
levels(data$stalk.surface.below.ring) =c("fibrous","silky","smooth","scaly")
levels(data$stalk.color.above.ring) = c("buff","cinnamon","red","gray","brown","orange","pink","white","yellow")
levels(data$stalk.color.below.ring) = c("buff","cinnamon","red","gray","brown","orange","pink","white","yellow")
levels(data$veil.type) = c("partial")
levels(data$veil.color) = c("brown","orange","white","yellow")
levels(data$ring.number) =c("none","one","two")
levels(data$ring.type) = c("evanescent","flaring","large","none","pendant")
levels(data$spore.print.color) = c("buff","chocolate","black","brown","orange","green","purple","white","yellow")
levels(data$population) = c("abundant","clustered","numerous","scattered","several","solitary")
levels(data$habitat) = c("woods","grasses","leaves","meadows","paths","urban","waste")

# Which is the attribute with more "gain"?
gain(data$cap.shape, data$class)
gain(data$ring.type, data$class)
gain(data$cap.color, data$class)
gain(data$odor, data$class)

# Lets compare two of them
table(data$odor, data$class)
table(data$cap.shape, data$class)

# (Imagine) odor is the attribute selected. Then, we have to do the same operation for all the generated partitions
table(data$odor)
data1 = data[data$odor == "almond",]
data2 = data[data$odor == "creosote",]
data3 = data[data$odor == "foul",]
data4 = data[data$odor == "anise",]
data5 = data[data$odor == "musty",]
data6 = data[data$odor == "nosmell",]
data7 = data[data$odor == "pungent",]
data8 = data[data$odor == "spicy",]
data9 = data[data$odor == "fishy",]


# And to repeat the procedure independently in each part. But only needed for odor = "nosmell"
gain(factor(data6$stalk.root), data6$class)
gain(factor(data6$spore.print.color), data6$class)
# (and so on...)


# When data is numeric...
data("PimaIndiansDiabetes")
data = PimaIndiansDiabetes

# We "check for every" possible cut
gain(data$pregnant>0, data$diabetes)
gain(data$pregnant>1, data$diabetes)
gain(data$pregnant>2, data$diabetes)
gain(data$pregnant>3, data$diabetes)
gain(data$pregnant>4, data$diabetes)
gain(data$pregnant>5, data$diabetes)
gain(data$pregnant>6, data$diabetes)
gain(data$pregnant>7, data$diabetes)
gain(data$pregnant>8, data$diabetes)
gain(data$pregnant>9, data$diabetes)
# (...)


#---------------------------------------------------------------------------
# PART 5: Decision tree is (Fortunately) implemented in R
#---------------------------------------------------------------------------
model1 = rpart(diabetes ~ .,data)
model1

h = predict(model1, data, type = "class")

table(h, data$diabetes)
