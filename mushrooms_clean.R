set.seed(1000) # set seed to ensure that split is the same every time file is run
library(tidyverse)
library(ggthemes)
library(extrafont) # extra fonts for plots
library(hrbrthemes) # for extra ggplot2 themes
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
library(MLmetrics) # for calculating KNN accuracy
library(neuralnet) # for neural network
library(nnet) # for neural network
library(devtools) # for running source_url() to get the plot.nnet function for neural network
source_url('https://gist.githubusercontent.com/fawda123/7471137/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet_plot_update.r')

# import windows default fonts for graphs
loadfonts(device = "win")

# read in and view dataset - 8124 rows and 23 columns
mushrooms <- read_csv("mushrooms.csv") # includes target column
view(mushrooms)

# check for na values - there are none
sum(is.na(mushrooms))

# convert boolean column ("bruises") into binary ints
mushrooms$bruises <- as.integer(as.logical(mushrooms$bruises))
view(mushrooms)

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
                       population + habitat", data = mushrooms, drop2nd = TRUE) # assign dummy variables to each column
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

#-----------------------------------------------------------------------------------------------------------------------------------------------------

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
  ggtitle("Top 10 Decision Tree Variables") +
  ylab("Importance") +
  xlab("") +
  theme(plot.title = element_text(size = 18, family = "Calibri"),
        axis.title = element_text(family = "Calibri"),
        axis.line = element_line(colour = "black"))
print(dt_var_importance)

#-----------------------------------------------------------------------------------------------------------------------------------------------------

# 2. k-nearest neighbors (KNN)

# drop all cap_surface variables because for some reason they were messing up the scaling
# probably will affect the model a little bit, but not to a crazy degree
train <- subset(train, select = -c(cap_surface.f, cap_surface.g, cap_surface.s, cap_surface.y))
test <- subset(test, select = -c(cap_surface.f, cap_surface.g, cap_surface.s, cap_surface.y))

# scale features of train and test sets
train_scaled <- scale(train[-113]) # remove 117th column aka target column
test_scaled <- scale(test[-113])
view(train_scaled)
view(test_scaled)

# training KNN and predicting
knn_test_pred <- knn(
  train = train_scaled,
  test = test_scaled,
  cl = train$poisonous, # cl = class labels
  k = 10 # number of nearest neighbors to consider when making predictions
)

# model evaluation
target <- test$poisonous # grab test set target column
knn_cm <- table(target, knn_test_pred) # cm = confusion matrix
view(knn_cm)

# calculate accuracy from confusion matrix
knn_accuracy <- sum(diag(knn_cm))/length(target)
sprintf("Accuracy: %.2f%%", knn_accuracy*100) # sprintf is a wrapper that returns a character vector containing a formatted combination of text and variable values

# k = 10 got us 99.94% accuracy which is obviously great, but let's see how increasing k can affect the accuracy
# plot time!
k_to_try = 1:100 # k = 1 to 100
acc_k = rep(x = 0, times = length(k_to_try))

for(i in seq_along(k_to_try)) { # seq_along function loops over a vector that stores non-consecutive numbers
  knn_test_pred = knn(train = train_scaled,
                      test = test_scaled,
                      cl = train$poisonous,
                      k = k_to_try[i])# sequence through k_to_try
  acc_k[i] = MLmetrics::Accuracy(target, knn_test_pred) # saves accuracy (calculated using Accuracy function from MLmetrics) into acc_k
}
acc_k <- data.frame(acc_k) # converts to data frame for plotting
acc_k$k_num <- 1:100 # adds a 1-100 column for plotting
view(acc_k)
ggplot(acc_k, aes(x = k_num, y = acc_k)) +
  geom_point(color = "dodgerblue1", size = 2) +
  geom_line() +
  ggtitle("KNN Accuracy vs Neighbors (K)") +
  xlab("Number of Neighbors (K)") +
  ylab("Accuracy") +
  theme(plot.title = element_text(hjust = 0.5, size = 22, family = "Calibri"), # center align title
        axis.title = element_text(family = "Calibri"),
        axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(), # remove grid lines
        panel.background = element_rect(fill = "white",
                                        colour = "white",
                                        size = 0.5, linetype = "solid")) # make background panel white

#-----------------------------------------------------------------------------------------------------------------------------------------------------

# 3. simple neural network

# calling this "simple" because the package i'm using ("neuralnet") is apparently outdated
# it's not fully customizable and isn't the best, but it'll be interesting to see how it compares to the decision tree and KNN

# convert target column to binary ints for conversion to numeric later (but only for train set, we're doing something different for test set)
train$poisonous <- factor(train$poisonous, levels = c("p", "e"), labels = c(1, 0)) # first convert vectors to factors, then replace with labels
test$poisonous <- factor(test$poisonous, levels = c("p", "e"), labels = c("1", "0"))
view(train)
view(test)

# convert poisonous binary ints from factors to numerics
train <- lapply(train, as.numeric)
test <- lapply(test, as.numeric)

columns_formula <- poisonous ~ . # save columns for later

view(train)
nn_model <- neuralnet(
  columns_formula,
  data = train,
  hidden = c(4,2), # 2 hidden layers: 1st layer has 4 neurons, 2nd layer has 2 neurons
  linear.output = FALSE
)

plot(nn_model, rep = "best") # plot neural network (not visually appealing at all)

# try another way of plotting neural network (still doesn't look good)
nn_plot <- nnet(columns_formula, train$poisonous, data = train, size = 1)
plot.nnet(nn_plot)

view(test)

nn_pred <- predict(nn_model, test)
labels <- c("p", "e")

prediction_label <- data.frame(nn_pred = labels[max.col(nn_pred)]) %>%
  mutate(nn_pred_label = labels[max.col(nn_pred)]) %>%
  select(2) %>%
  unlist()
view(nn_pred)
view(max.col(nn_pred))
view(prediction_label)
print(table(test$poisonous, prediction_label)) # print confusion matrix