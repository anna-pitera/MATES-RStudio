set.seed(1000) # set seed to ensure that split is the same every time file is run
library(tidyverse)
library(ggthemes)
library(caret) # for one-hot encoding categorical variables (the majority of variables are categorical in this dataset)
library(dataMaid) # for auto EDA
library(DataExplorer) # for auto EDA
library(caTools) # for splitting data into training and test set
library(rpart) # for decision tree (CART)
library(tidymodels) # for decision tree (CART)
library(tidyr) # for decision tree (CART)
library(rpart.plot) # for plotting decision trees
library(vip) # for feature importance (aka how big of an impact each feature has on the target prediction)
library(class) # for k-nearest neighbors (KNN)

# read in and view dataset - 8124 rows and 23 columns
mushrooms <- read_csv("mushrooms.csv") # includes target column
view(mushrooms)

# check for na values - there are none
colSums(is.na(mushrooms))

# convert boolean column ("bruises") into binary ints
mushrooms$bruises <- as.integer(as.logical(mushrooms$bruises))
view(mushrooms)

# grab poisonous column to save for later
poisonous_col <- data.frame(mushrooms["poisonous"])
view(poisonous_col)

# one-hot encoding categorical variables using caret
colnames(mushrooms) # get col names
# at this point, i figured out that "veil_type" only has one value ("p") across the entire dataset
# after looking at stack overflow for a bit, i decided i should drop "veil_type" because
# R isn't going to let me encode it or use it in machine learning algorithms (because it has only one value)
# and dropping it isn't going to hurt anything because it's the same value ("p") for every row
mushrooms <- subset(mushrooms, select = -c(veil_type, poisonous)) # drop "veil_type"
# back to encoding
cols <- c("cap_shape", "cap_surface", "cap_color", "bruises",
          "odor", "gill_attachment", "gill_spacing",
          "gill_size", "gill_color", "stalk_shape",
          "stalk_root", "stalk_surface_above_ring",
          "stalk_surface_below_ring", "stalk_color_above_ring",
          "stalk_color_below_ring", "veil_color",
          "ring_number", "ring_type", "spore_print_color",
          "population", "habitat"
)
view(mushrooms)
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

mushrooms["poisonous"] <- data.frame(poisonous_col["poisonous"]) # add target column back to dataset because R allows you to specify the target column during fitting
view(mushrooms)

# # auto EDA with dataMaid - outputs as html
# makeDataReport(mushrooms, output = "html", replace = TRUE)
# 
# # auto EDA with DataExplorer - outputs as html
# create_report(mushrooms)

# for this particular dataset, auto EDA isn't super useful because of the way the dataset is set up
# but leaving it in there just in case

# splitting data into training and test set
# # 80% train to 20% test ratio
data_split <- initial_split(mushrooms, prop = 0.8)
train <- training(data_split)
test <- testing(data_split)
view(train) # 6499 rows
view(test) # 1625 rows

# 1. decision tree (CART)
tree_spec <- decision_tree() %>% # decision tree model specifications
  set_engine("rpart") %>% # "rpart" is a package that is regarded as being faster than other older decision tree packages like "tree"
  set_mode("classification") # using classification not regression because target column is categorical

# converting target column to factor so accuracy can be assessed with metric_set() which requires either factors for classification or numerics for regression
train["poisonous"] <- lapply(train["poisonous"], factor) # lapply() applies a function over a list or vector, so in this case we applied the factor function
test["poisonous"] <- lapply(test["poisonous"], factor)

# fitting and predicting model
tree_fit <- tree_spec %>%
  fit(poisonous ~ ., data = train) # " ~ . " means "any columns from data that are otherwise not used"
dt_predictions <- tree_fit %>%
  predict(test) %>%
  pull() # pull() extracts a single column - in the guide i'm using, it pulls a column called ".pred", but that wouldn't work for me, so this is the way i got it to work

# assessing accuracy and performance of model
dt_metrics <- metric_set(accuracy, kap) # accuracy and kap are the classification metrics
# accuracy is from 0-1.0, with higher number being more accurate
# kap (short for kappa) is similar to accuracy (from 0-1.0), but is normalized by the accuracy that would be expected by chance alone
# kap is apparently very useful when one or more classes have large frequency distributions
dt_model_performance <- test %>%
  mutate(dt_predictions = factor(dt_predictions)) %>%
  dt_metrics(truth = poisonous, estimate = factor(dt_predictions))
print(dt_model_performance) # accuracy = 0.993, kap = 0.985
# this high accuracy could be because the dataset is so simple and doesn't have as many feature columns as bigger ML datasets
# let's look into the model and its rules to see how it got so accurate

# decision tree plot (not very aesthetically pleasing - unfortunately this seems to be the only option for customizing a decision tree plot in R)
rpart.plot(tree_fit$fit, type = 1, extra = 100, under = TRUE, box.palette = "Rd", branch.lty = 1, branch.col = "gray40", split.lwd = 10)
# written-out rules for the model
dt_rules <- rpart.rules(tree_fit$fit)
print(dt_rules) # based on this output, it looks like there's not many rules to the model, probably because it's so simple
# variable importance (aka how much of an impact each variable has on the model's outcome)
dt_var_importance <- vip(tree_fit, num_features = 10) # looks at top 10 most important features
print(dt_var_importance) # just like the tree plot and the written-out rules showed, odor is the most important variable by a lot
# adding a title and changing the color of the bars
dt_var_importance <- vip(tree_fit, num_features = 10, aes = list(fill = "springgreen2")) +
  ggtitle("Decision Tree Variable Importance")
print(dt_var_importance)

# 2. k-nearest neighbors (KNN)

# for this knn approach, we need the target column to be in binary ints
# this is because the scale function doesn't like categorical columns, even if you exclude it with something like [-1]
train$poisonous <- factor(train$poisonous, levels=c("p", "e"), labels=c(1, 0)) # first convert vectors to factors, then replace with labels
test$poisonous <- factor(test$poisonous, levels=c("p", "e"), labels=c(1, 0))
view(train)
view(test)

# TRY DROPPING POISONOUS THEN DOING IT WITHOUT POISONOUS IG?

# then convert binary ints poisonous column into a numerical column (because otherwise it's basically being read as a string)
train$poisonous <- as.numeric(as.factor(train$poisonous))
test$poisonous <- as.numeric(as.factor(test$poisonous))

view(train)

# scale features of train and test sets
train_scaled <- scale(train[,-117])
test_scaled <- scale(test[,-117])

view(train_scaled)



# training KNN and predicting
knn_test_pred <- knn(
  train = train_scaled, 
  test = test_scaled,
  cl = train$poisonous, 
  k=10
)

