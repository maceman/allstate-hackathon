# Set working directory
setwd("/Users/maceyma/Desktop/Allstate Hackathon/Data")

# Multiclass log loss - validation metric

# load in libraries
install.packages("data.table")
install.packages("gbm")
install.packages("Metrics")
install.packages("glmnet")
install.packages("MASS")
library(data.table) # tutorial assumes  v1.9.7
library(gbm) # CRAN v2.1.1
library(Metrics)
library(glmnet)
library(MASS)

# Read in training and test data
data_all <- fread('/Users/maceyma/Desktop/Allstate Hackathon/Data/train.csv')
ids_test <- fread('/Users/maceyma/Desktop/Allstate Hackathon/Data/test.csv')

### create entire dataset format ####

data_all[, dummy_counter:=1]

data_all2 = data_all
# find the first/last event for each policy in training data
data_all2[, max_timestamp_by_policy:=max(timestamp), by=id]
data_all2[, min_timestamp_by_policy:=min(timestamp), by=id]

# create last event feature
data_all_last_element = subset(data_all2, timestamp == max_timestamp_by_policy)
data_all_last_element$timestamp = NULL
data_all_last_element$dummy_counter = NULL
data_all_last_element$max_timestamp_by_policy = NULL
data_all_last_element$min_timestamp_by_policy = NULL
setnames(data_all_last_element, "event", "last_event")

# create first even feature
data_all_first_element = subset(data_all2, timestamp==min_timestamp_by_policy)
data_all_first_element$timestamp = NULL
data_all_first_element$dummy_counter = NULL
data_all_first_element$max_timestamp_by_policy = NULL
data_all_first_element$min_timestamp_by_policy = NULL
setnames(data_all_first_element, "event", "first_event")

# featurize data, without the last element, by looking at counts of events
data_allButLast <- subset(data_all2, timestamp != max_timestamp_by_policy)
data_ohe <- dcast(data_allButLast, 
                        id ~ event, 
                        fun.aggregate=sum, 
                        value.var="dummy_counter")

# create a feature that counts how many events occured in total before the last event
data_ohe[, total_events:=apply(data_ohe[, -c("id"), with=FALSE], 1, sum)]

data_ohe <- merge(data_ohe, data_all_first_element, by=c("id"))
data_ohe$first_event <- as.factor(data_ohe$first_event)

data_all_merged <- merge(data_ohe, data_all_last_element, by=c("id"))


##created entire dataset in correect format above ^



# create dummy counter variable
data_all[, dummy_counter:=1]

data_train <- subset(data_all, !(id %in% ids_test$id))

# find the first/last event for each policy in training data
data_train[, max_timestamp_by_policy:=max(timestamp), by=id]
data_train[, min_timestamp_by_policy:=min(timestamp), by=id]

# create last event feature
data_train_last_element = subset(data_train, timestamp == max_timestamp_by_policy)
data_train_last_element$timestamp = NULL
data_train_last_element$dummy_counter = NULL
data_train_last_element$max_timestamp_by_policy = NULL
data_train_last_element$min_timestamp_by_policy = NULL
setnames(data_train_last_element, "event", "last_event")

# create first even feature
data_train_first_element = subset(data_train, timestamp==min_timestamp_by_policy)
data_train_first_element$timestamp = NULL
data_train_first_element$dummy_counter = NULL
data_train_first_element$max_timestamp_by_policy = NULL
data_train_first_element$min_timestamp_by_policy = NULL
setnames(data_train_first_element, "event", "first_event")

# featurize data, without the last element, by looking at counts of events
data_train_allButLast <- subset(data_train, timestamp != max_timestamp_by_policy)
data_train_ohe <- dcast(data_train_allButLast, 
                        id ~ event, 
                        fun.aggregate=sum, 
                        value.var="dummy_counter")

# create a feature that counts how many events occured in total before the last event
data_train_ohe[, total_events:=apply(data_train_ohe[, -c("id"), with=FALSE], 1, sum)]

data_train_ohe <- merge(data_train_ohe, data_train_first_element, by=c("id"))
data_train_ohe$first_event <- as.factor(data_train_ohe$first_event)

data_train_merged <- merge(data_train_ohe, data_train_last_element, by=c("id"))

########################
# train a gbm model
# WARNING: Run time can exceed 10-15 minutes. For faster run times, lower n.trees=10
model <- gbm(last_event ~ . - id, 
             distribution="multinomial", 
             data=data_train_merged, 
             interaction.depth=5,
             n.trees=100,
             shrinkage=0.1,
             train.fraction=0.8,
             verbose=TRUE)

gbm.perf(model)
summary(model, las=2)

model2 <- gbm(last_event ~ . - id, 
             distribution="multinomial", 
             data=data_train_merged, 
             interaction.depth=10,
             n.trees=200,
             shrinkage=0.001,
             train.fraction=0.7,
             verbose=TRUE)

gbm.perf(model2)
summary(model2, las=2)

### Test Data Transformation

# create the test dataset
data_test <- subset(data_all, id %in% ids_test$id)
data_test[, min_timestamp_by_policy:=min(timestamp), by=id]

# create the first event feature
data_test_first_event <- subset(data_test, timestamp==min_timestamp_by_policy)
data_test_first_event$timestamp = NULL
data_test_first_event$dummy_counter = NULL
data_test_first_event$min_timestamp_by_policy = NULL
setnames(data_test_first_event, "event", "first_event")

data_test_ohe <- dcast(data_test, id~event, fun.aggregate=sum, value.var="dummy_counter")
# add the total events feature
data_test_ohe[, total_events:=apply(data_test_ohe[, -c("id"), with=FALSE], 1, sum)]

# add the first event feature
data_test_ohe <- merge(data_test_ohe, data_test_first_event, by=c("id"))

# score the model
predictions_raw <- predict(model, data_test_ohe, type="response")

#package it into a format that can be scored
predictions <- as.data.table(predictions_raw[,,1])
setnames(predictions, names(predictions), colnames(predictions_raw))
predictions[, id:=data_test_ohe$id]

pred_columns <- colnames(predictions)
pred_columns <- c('id', sort(pred_columns[-length(pred_columns)]))
setcolorder(predictions, pred_columns)
setnames(predictions, c(pred_columns[1], 
                        paste('event_', pred_columns[-1], sep='')))

write.csv(predictions, file="/Users/maceyma/Desktop/Allstate Hackathon/Data/to_upload.csv", quote=TRUE, row.names=FALSE)

#### Prediction - more code
library(glmnet)
X = model.matrix(last_event~ . - id ,data_all_merged)
Xtrain = model.matrix(last_event ~. -id, data_train_merged)
Xtest = model.matrix(last_event ~.-id, data_test_all)
y = data_all_merged$last_event
yTrain = data_train_merged$last_event
grid = 10^seq(10,-2,length=100)


data_train_all <- subset(data_all_merged, !(id %in% ids_test$id))
data_test_all <- subset(data_all_merged, id %in% ids_test$id)

ridge.mod = glmnet(Xtrain,yTrain,alpha=0,lambda=grid,thresh=1e-12,family="multinomial")
cv.out = cv.glmnet(Xtrain,yTrain,alpha=0)
plot(cv.out)
bestlam = cv.out$lambda.min
bestlam
ridge.predBest = predict(ridge.mod,s=bestlam,type="response",newx=Xtest)



## manipulate for ridge regression

data_test <- subset(data_all, id %in% ids_test$id)
data_test[, min_timestamp_by_policy:=min(timestamp), by=id]

# create the first event feature
data_test_first_event <- subset(data_test, timestamp==min_timestamp_by_policy)
data_test_first_event$timestamp = NULL
data_test_first_event$dummy_counter = NULL
data_test_first_event$min_timestamp_by_policy = NULL
setnames(data_test_first_event, "event", "first_event")

data_test_ohe <- dcast(data_test, id~event, fun.aggregate=sum, value.var="dummy_counter")
# add the total events feature
data_test_ohe[, total_events:=apply(data_test_ohe[, -c("id"), with=FALSE], 1, sum)]

# add the first event feature
data_test_ohe <- merge(data_test_ohe, data_test_first_event, by=c("id"))

# score the model
predictions_raw <- ridge.predBest

#package it into a format that can be scored
predictions_raw = as.data.frame(predictions_raw)
predictions_raw$id = data_test_ohe$id
predictions_raw = predictions_raw[,c( "id" ,"30018.1", "30021.1" ,"30024.1" ,"30027.1" ,"30039.1" ,"30042.1" ,"30045.1", "30048.1", "36003.1", "45003.1")]
names(predictions_raw) = c( "id" ,"event_30018", "event_30021" ,"event_30024" ,"event_30027" ,"event_30039" ,"event_30042" ,"event_30045", "event_30048", "event_36003", "event_45003")

ridgemodelfinal = predictions_raw

### Updated code
write.csv(ridgemodelfinal, file="/Users/maceyma/Desktop/Allstate Hackathon/Data/secondtry.csv", quote=TRUE, row.names=FALSE)
write.csv(data_test_ohe, file="/Users/maceyma/Desktop/Allstate Hackathon/Data/jmp_test.csv", quote=TRUE, row.names=FALSE)

## neural network now

library(nnet)

data_all_merged$id = as.factor(data_all_merged$id)

data_train_all <- subset(data_all_merged, !(id %in% ids_test$id))
data_test_all <- subset(data_all_merged, id %in% ids_test$id)

data_all_merged$last_event = as.factor(data_all_merged$last_event)
cylModel <- multinom(last_event~ . - id, data=data_all_merged, maxit=500, trace=T)

preds1 <- predict(cylModel, type="probs", newdata=data_test_all)
# score the model
predictions_raw2 <- preds1

#package it into a format that can be scored
predictions_raw2 = as.data.frame(predictions_raw2)
predictions_raw2$id = data_test_all$id
predictions_raw2 = predictions_raw2[,c( "id" ,"30018", "30021" ,"30024" ,"30027" ,"30039" ,"30042" ,"30045", "30048", "36003", "45003")]
names(predictions_raw2) = c( "id" ,"event_30018", "event_30021" ,"event_30024" ,"event_30027" ,"event_30039" ,"event_30042" ,"event_30045", "event_30048", "event_36003", "event_45003")

nnfinal2 = predictions_raw2
write.csv(nnfinal2, file="/Users/maceyma/Desktop/Allstate Hackathon/Data/nnfinal2.csv", quote=TRUE, row.names=FALSE)

library(MLmetrics)
MultiLogLoss(y_true = data_test_all$last_event,y_pred=ridgemodelfinal[,-1])


library(randomForest)
names(data_all_merged) = c( "id" ,"event_30018", "event_30021" ,"event_30024" ,"event_30027" ,"event_30039" ,"event_30042" ,"event_30045", "event_30048", "event_36003", "event_45003","total_events","first_event","last_event")

data_all_merged$last_event = as.factor(data_all_merged$last_event)
bag.model = randomForest(last_event~ . - id, data=data_all_merged,mtry=8,importance=TRUE)
yhatbag = predict(bag.model,newdata=data_test_all,type="prob")


#[1] 0.6737617
