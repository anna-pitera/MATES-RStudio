set.seed(1000) # set seed to ensure that split is the same every time file is run
library(tidyverse)
library(ggthemes)
library(caret) # for one-hot encoding categorical variables (the majority of variables are categorical in this dataset)
library(dataMaid) # for auto EDA
library(DataExplorer) # for auto EDA
library(caTools) # for splitting data into training and test set
library(rpart) # for decision tree (CART)
library(rpart.plot) # for plotting decision trees
library(vip) # for feature importance (aka how big of an impact each feature has on the target prediction)
library(pdp) # for feature effects (aka the effect of changes in the value of each feature on the model's predictions)
library(tidymodels) # for second attempt at decision tree (CART)
library(tidyr) # for second attempt at decision tree

# read in and view dataset - 8124 rows and 23 columns
mushrooms <- read_csv("mushrooms.csv") # includes target column (that's why it's called "all")
view(mushrooms)

# check for na values - there are none
colSums(is.na(mushrooms))

# convert boolean column ("bruises") into binary ints
mushrooms$bruises <- as.integer(as.logical(mushrooms$bruises))
view(mushrooms)

# grab poisonous column to save for later
poisonous_col <- data.frame(mushrooms["poisonous"])
view(poisonous_col)

# # target column ("poisonous") is written as p = poisonous and e = edible
# # but it'd be better if it were just binary, with poisonous = 1 and edible = 0
# # convert target column ("poisonous") into binary ints
# mushrooms_all$poisonous <- factor(mushrooms_all$poisonous, levels=c("p", "e"), labels=c(1, 0)) # first convert vectors to factors, then replace with labels
# view(mushrooms)

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

view(poisonous_col)

mushrooms["poisonous"] <- data.frame(poisonous_col["poisonous"])
view(mushrooms)

# # auto EDA with dataMaid - outputs as html
# makeDataReport(mushrooms, output = "html", replace = TRUE)
# 
# # auto EDA with DataExplorer - outputs as html
# create_report(mushrooms)

# for this particular dataset, auto EDA isn't super useful because of the way the dataset is set up
# but leaving it in there just in case

# FIRST ATTEMPT AT SPLITTING DATA INTO TRAINING AND TEST SET
# # splitting data into training and test set with caTools
# # 80% train to 20% test ratio
# sample <- sample.split(mushrooms$cap_shape.b, SplitRatio = 0.8)
# train <- subset(mushrooms, sample == TRUE)
# test <- subset(mushrooms_all, sample == FALSE)
# view(train) # 6500 rows
# view(test) # 1624 rows

# splitting data into training and test set
# # 80% train to 20% test ratio
data_split <- initial_split(mushrooms, prop = 0.8)
train <- training(data_split)
test <- testing(data_split)
view(train)


# # FIRST ATTEMPT AT DECISION TREE
# # converting data (which is a list) into a numeric vector so the model can use it
# train <- lapply(train, as.numeric)

# # 1. decision tree (CART) model
# # i'll be using the "rpart" package, which is regarded as being much faster than other packages like "tree"
# dt1 <- rpart(
#   formula = cap_shape.b ~ ., # "~ . " selects all features as well as cap_shape.b
#   data    = train,
#   method  = "anova" # by default, rpart tries its best to assume what fitting method to use, but it's best to specify in the beginning
# )
# dt1 # run decision tree
# rpart.plot(dt1) # plot decision tree
# summary(dt1) # tells us the number of observations in each node, splits, terminal nodes, etc
# plotcp(dt1) # plots elbow chart using rpart's automatic pruning
# # but let's see what the elbow chart looks like without any pruning
# dt2 <- rpart(
#   formula = cap_shape.b ~ .,
#   data    = train,
#   method  = "anova",
#   control = list(cp = 0, xval = 10) # generates full tree
# )
# dt2
# plotcp(dt2) # this plot shows that the error percentage plateaus at 13 terminal nodes
# # so it's best optimized at 13 terminal nodes

# 1. decision tree (CART) model but better
# this method is better for a few reasons, mostly simplicity and convenience
# but also because it doesn't require the dataset to be a numeric vector
# so it's simpler

view(train)

tree_spec <- decision_tree() %>% # decision tree model specifications
  set_engine("rpart") %>% # same thing as rpart from the first attempt, just a different way
  set_mode("classification")

train["poisonous"] <- lapply(train["poisonous"], factor)
test["poisonous"] <- lapply(test["poisonous"], factor)

tree_fit <- tree_spec %>%
  fit(poisonous ~ ., data = train)

predictions <- tree_fit %>%
  predict(test) %>%
  pull()

metrics <- metric_set(accuracy, kap)
model_performance <- test %>%
  mutate(predictions = factor(predictions)) %>%
  metrics(truth = poisonous, estimate = factor(predictions))
print(model_performance)

rpart.plot(tree_fit$fit, type = 4, extra = 101, under = TRUE, cex = 0.8, box.palette = "auto")
rules <- rpart.rules(tree_fit$fit)
print(rules)
new_data <- tribble(
  ~crim, ~zn, ~indus, ~chas, ~nox, ~rm, ~age, ~dis, ~rad, ~tax, ~ptratio, ~black, ~lstat,
  0.03237, 0, 2.18, 0, 0.458, 6.998, 45.8, 6.0622, 3, 222, 18.7, 394.63, 2.94
)