library(tidyverse)

data <- read.csv(file.choose(), head = T)

head(data)

# we will start solving the problem by using glm - binary case

str(data)

# factorizing KPIs met and is_promoted: Undoubtedly our target variable is 
data$KPIs_met..80. <- as.factor(data$KPIs_met..80.)
data$is_promoted <- as.factor(data$is_promoted)


analysis1 <- glm(is_promoted ~ ., family = binomial(), data = data)

summary(analysis1)

#analysis: gender is not really significant - that means it doesn't matter if you're M or F, if you have right characteristics 
#now we'll use randomForest only deploying the most significant characteristics

#################################

library(randomForest)

#spilitting the dataset into 80% training and 20% testing purposes by selecting only the required variables

data_new <- data %>% select(employee_id, no_of_trainings, age, previous_year_rating, length_of_service, KPIs_met..80., awards_won., avg_training_score, is_promoted)

set.seed(1234)
data_new1 <- sample(2, nrow(data_new), replace = T, prob = c(0.8,0.2))
train <- data_new[data_new1 == 1,]
test <- data_new[data_new1 == 2,]

## checking for missing values

is.na(train) # there are missing values in previous year training variable - removing these values
train <- na.omit(train)
test <- na.omit(test)


##model
rf <- randomForest(is_promoted ~., train,
                   mtry = 2)

#storing the results

# let us see our predictions... how good is our model?

library(caret)


pred3 <- predict(rf, train)
matrix <- confusionMatrix(pred3, train$is_promoted)

pred1 <- predict(rf, test)
matrix1 <- confusionMatrix(pred1, test$is_promoted)

## we have achieved an accuracy of 93% for the training data and 92% for the test data.... let us try improve the accuracy by improving mtry number. 

tune <- tuneRF(train[,-9], train[,9],
               stepFactor = 1,
               plot = T,
               ntreeTry = 30,
               trace = T,
               improve = 1)

### mtry isn't really improving when tried
### let's look at some important variables

varImpPlot(rf, sort = T)


###################### we have used RF let us try XGBoost as our next algorithm to practice on!
# we already have two sets - training and test datasets

library(xgboost)
library(Matrix)

# Matrix - one hot encoding

train$is_promoted <- as.integer(train$is_promoted) #converting the dependent variable back to integer
train$is_promoted <- train$is_promoted - 1
train$is_promoted <- as.integer(train$is_promoted)

train$KPIs_met..80.<- as.integer(train$KPIs_met..80.)




test$is_promoted <- as.integer(test$is_promoted) #converting the dependent variable back to integer
test$is_promoted <- test$is_promoted - 1
test$is_promoted <- as.integer(test$is_promoted)

test$KPIs_met..80.<- as.integer(test$KPIs_met..80.)


######################################################################



train1 <- sparse.model.matrix(is_promoted ~ .-1, data = train)
train_label <- train[,"is_promoted"]
train_matrix <- xgb.DMatrix(data = as.matrix(train1), label = train_label)





# we'll do the same with test data set 

test1 <- sparse.model.matrix(is_promoted ~ .-1, data = test)
test_label <- test[,"is_promoted"]
test_matrix <- xgb.DMatrix(data = as.matrix(test1), label = test_label)

# parameters to evaluate the model effectively

noc <- length(unique(train_label))
xgb_params <- list("objective" = "multi:softprob",
                   "eval_metric" = "mlogloss",
                   "num_class" = noc)
watchlist = list(train = train_matrix, test = test_matrix)

# xgBoosting model

model <- xgb.train(params = xgb_params,
                   data = train_matrix,
                   nrounds = 1000,
                   watchlist = watchlist)

## confusion matrix

p <- predict(model, newdata = test_matrix)
pred <- matrix(p, nrow = noc, ncol = length(p)/noc) %>%
  t() %>%
  data.frame() %>%
  mutate(label = test_label, max_prob = max.col(., "last")-1)
table(Prediction = pred$max_prob, Actual = pred$label)

###### we have obtained same accuracy with xgboost as well...


imp <- xgb.importance(colnames(train_matrix), model = model)
print(imp)
xgb.plot.importance(imp)


#both algorithms give importance to similar variables.
