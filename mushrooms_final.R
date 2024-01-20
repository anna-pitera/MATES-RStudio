set.seed(1000) # set seed to ensure that split is the same every time file is run
library(tidyverse)
library(ggthemes)
library(caret) # for one-hot encoding categorical variables (the majority of variables are categorical in this dataset)
library(dataMaid) # for auto EDA
library(DataExplorer) # for auto EDA
library(caTools) # for splitting data into training and test set
library(rpart) # for decision tree (CART)
library(rpart.plot) # for plotting decision trees
library(tree) # FILL IN
library(vip) # for feature importance (aka how big of an impact each feature has on the target prediction)
library(pdp) # for feature effects (aka the effect of changes in the value of each feature on the model's predictions)

# read in and view dataset - 8124 rows and 23 columns
mushrooms <- read_csv("mushrooms.csv")
view(mushrooms)

# check for na values - there are none
colSums(is.na(mushrooms))

# convert boolean column ("bruises") into binary ints
mushrooms$bruises <- as.integer(as.logical(mushrooms$bruises))
view(mushrooms)

# target column ("poisonous") is written as p = poisonous and e = edible
# but it'd be better if it were just binary, with poisonous = 1 and edible = 0
# convert target column ("poisonous") into binary ints
mushrooms$poisonous <- factor(mushrooms$poisonous, levels=c("p", "e"), labels=c(1, 0)) # first convert vectors to factors, then replace with labels
view(mushrooms)

# one-hot encoding categorical variables using caret
colnames(mushrooms) # get col names
# at this point, i figured out that "veil_type" only has one value ("p") across the entire dataset
# after looking at stack overflow for a bit, i decided i should drop "veil_type" because
# R isn't going to let me encode it or use it in machine learning algorithms (because it has only one value)
# and dropping it isn't going to hurt anything because it's the same value ("p") for every row
mushrooms <- subset(mushrooms, select = -c(veil_type)) # drop "veil_type"
# back to encoding
cols <- c("cap_shape", "cap_surface", "cap_color", "bruises",
          "odor", "gill_attachment", "gill_spacing",
          "gill_size", "gill_color", "stalk_shape",
          "stalk_root", "stalk_surface_above_ring",
          "stalk_surface_below_ring", "stalk_color_above_ring",
          "stalk_color_below_ring", "veil_color",
          "ring_number", "ring_type", "spore_print_color",
          "population", "habitat"
          ) # every column except "poisonous" because it's the target
# so "poisonous" aka the target column has been dropped at this point
mushrooms[cols] <- lapply(mushrooms[cols], factor) # convert selected columns to factors for encoding
dv <- caret::dummyVars(" ~ cap_shape + cap_surface + cap_color + bruises +
                       odor + gill_attachment + gill_spacing +
                       gill_size + gill_color + stalk_shape +
                       stalk_root + stalk_surface_above_ring +
                       stalk_surface_below_ring + stalk_color_above_ring +
                       stalk_color_below_ring + veil_color +
                       ring_number + ring_type + spore_print_color +
                       population + habitat", data = mushrooms) # assign dummy variables to each column
mushrooms <- data.frame(predict(dv, newdata = mushrooms)) # convert to data frame so it's viewable
view(mushrooms) # columns are all encoded

# # auto EDA with dataMaid - outputs as html
# makeDataReport(mushrooms, output = "html", replace = TRUE)
# 
# # auto EDA with DataExplorer - outputs as html
# create_report(mushrooms)

# for this particular dataset, auto EDA isn't super useful because of the way the dataset is set up
# but leaving it in there just in case

# splitting data into training and test set with caTools
# 80% train to 20% test ratio
sample <- sample.split(mushrooms$cap_shape.b, SplitRatio = 0.8)
train <- subset(mushrooms, sample == TRUE)
test <- subset(mushrooms, sample == FALSE)
view(train) # 6500 rows
view(test) # 1624 rows

# converting data (which is a list) into a numeric vector so the model can understand it
train <- data.frame(cols=unlist(cols(train)))

# now it's time for the actual machine learning algorithms!

# 1. decision tree (CART) model
# i'll be using the "rpart" package, which is regarded as being much faster than other packages like "tree"
dt1 <- rpart(
  formula = cap_shape.b ~ .,
  data    = train,
  method  = "anova" # by default, rpart tries its best to assume what fitting method it uses, but it's best to specify in the beginning
)

dt1

rpart.plot(dt1)