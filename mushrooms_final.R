library(tidyverse)
library(ggthemes)
library(caret) # for one-hot encoding categorical variables (the majority of variables are categorical in this dataset)
library(dataMaid) # for auto EDA
library(DataExplorer) # for auto EDA
library(caTools) # for splitting data into training and test set

# read in and view dataset - 8124 rows and 23 columns
mushrooms <- read_csv("mushrooms.csv")
view(mushrooms)

# convert boolean column ("bruises") into binary ints
mushrooms$bruises <- as.integer(as.logical(mushrooms$bruises))
view(mushrooms)

# target column ("poisonous") is written as p = poisonous and e = edible
# but it'd be better if it were just binary, with poisonous = 1 and edible = 0
# convert target column ("poisonous") into binary ints
mushrooms$poisonous <- factor(mushrooms$poisonous, levels=c("p", "e"), labels=c(1, 0))
view(mushrooms)

# one-hot encoding categorical variables using caret


# drop target column (first column, labeled "poisonous")
mushrooms <- subset(mushrooms, select = -c(poisonous))
view(mushrooms)

# auto EDA with dataMaid - outputs as html 
makeDataReport(mushrooms, output = "html", replace = TRUE)

# auto EDA with DataExplorer - outputs as html
create_report(mushrooms)

# for this particular dataset, auto EDA isn't super useful because of the way the dataset is set up
# but leaving it in there just in case

# set seed to ensure that split is the same every time file is run
set.seed(72)

# splitting data into training and test set with caTools
# 80% train to 20% test ratio
sample <- sample.split(mushrooms$cap_shape, SplitRatio = 0.8)
train <- subset(mushrooms, sample == TRUE)
test <- subset(mushrooms, sample == FALSE)

