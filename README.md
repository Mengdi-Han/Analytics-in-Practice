# Analytics-in-Practice
Using data mining models to develop customer transaction prediction system

Using different models to find the best performance
```{r message=FALSE}
install.packages("tidyverse")
library(tidyverse)
```

## Load the full datafile
```{r message=FALSE}
full_data <- read.csv("datafile_full.csv")
```
# Data Preparation


## Step1: Check descriptive statistics of the whole dataset
### 1.Obtain a brief overview of the dataset: attribute distribution, outliers, missing value
```{r}
str(full_data)
View(summary(full_data))
```


### 2.Visualize distribution and outliers of each attribute: Provide boxplot of attributes,take variable 60 to 80 as example
```{r message=FALSE}
boxplot(full_data[,63:83],boxwex = 0.25,col = "blue",
         main = "Checking distribution of each attribute",
         xlab = "Attributes 60:80",
         ylab = "Value of attribute entries",
         ylim = c(0, 100), yaxs = "i")
```

## Findings: 
### a. ID_ Code is a string of regular integers, which makes no contribution to the prediction of the target variable, and thus should be removed..
### b. Type of the target variable is Integer, which needs to be changed to Factor.
### c. All variables, except for the target variable, are numeric attributes.
### d. Missing values and outliers exist in some attributes, and these require further remedies.

# 
### 3.Changing type of the target variable:
```{r}
full_data$target <- as.factor(full_data$target)
```

### 4.Removing the attribute of "ID": it has no power on predicting values of the target variable.
```{r message=FALSE}
full_data <- full_data[,-1]
```
## Step2: Cleaning Training dataset:

### 1.Dealing with Missing value:
### Method 1: removing all missing values of the dataset
```{r message=FALSE}
new_full_data1 <- na.omit(full_data)
nrow(full_data) - nrow(new_full_data1)
```

### Findings: 
### a. When removing all observations with missing values, 107 observations are removed. Comparing to 100000 observations of the full dataset, 107 is relatively a small number, which might not cause a serious problem of information loss. 
### b. However, other methods are worth trying to identify whether different methods will lead to better performance of modeling.

# 
### Method 2: fill missing values with average value of each column,sinvce average value will not change the distribution of original dataset.

```{r}
new_full_data2 <-full_data
for(i in 2:201) {
new_full_data2[is.na(new_full_data2[, i]), i] <-mean(new_full_data2[, i], na.rm = T)
}
```



### 2.Removing duplicate observations
```{r}
#missing value method 1
new_full_data1 <- distinct(new_full_data1)
#missing value method 2
new_full_data2 <- distinct(new_full_data2)
```

### 3.Removing outliers: 

### It is highly possible for outliers to be valid observations, which could refer to people who have more wealth than ordinary customers. However, since the meaning of variables with outliers are unknown, these outliers are rather risky to keep, and thus are removed. 
### Boxplot is made up of the maximum, minimum, median, upper quartile (Q1) and lower quartile (Q3) in the data set. It is mainly used to reflect the distribution of original data. If an observation is higher than [(Q1)+1.5(Q1-Q3)] or lower than [(Q3)-1.5(Q1-Q3)], this value can be defined as outliers. 

```{r message=FALSE}
# missing value method 1
dateoutlier1 <- for (i in 2:201) {
  outliers1 <- boxplot(new_full_data1[, i], plot = FALSE)$out
}
new_full_data1<- new_full_data1[-which(new_full_data1[,i] %in% outliers1),]

```


```{r message=FALSE}
# missing value method 2
dateoutlier2 <- for (i in 2:201) {
  outliers2 <- boxplot(new_full_data2[, i], plot = FALSE)$out
}
new_full_data2<- new_full_data2[-which(new_full_data2[,i] %in% outliers2),]
```



## Step3:Dimensionality reduction:

### 1.Checking the correlation between variables:

### If the correlations between variables are large, it maybe worthwhile to apply dimensionality reduction, though not knowing the meaning of variables would cause difficulty in explaining the impact.
```{r message=FALSE}
install.packages("stats")
library(stats)
install.packages("Hmisc")
library(Hmisc)
```

```{r message=FALSE}
# missing value method 1
correlation1 <-cor(new_full_data1[,2:201],new_full_data1[,2:201]) 
# Check the matrix of correlations
corsig1 <- rcorr(as.matrix(new_full_data1)) 
# Check the significance of the correlations
print(corsig1)
```

```{r message=FALSE}
# missing value method 2
correlation2 <-cor(new_full_data2[,2:201],new_full_data2[,2:201]) 
# Check the matrix of correlations
corsig2 <- rcorr(as.matrix(new_full_data2)) 
# Check the significance of the correlations
print(corsig2)
```

### Findings: 
### a. As indicated by the p value, correlations between variables are insignificant. 
### b. Since Pearson coefficient is only effectient in linear correlation situation, it is better to use other methods to apply feature selection.

# 
## Step4: Normalization:

### All variables except for the target variable are numeric, and each of them has a relatively different value scale. To avoid impact of scale of data to some models built in later stage, it is necessary to use both the original dataset and dataset with all independent variables normalized to check whether model performance will imporve.
```{r message=FALSE}
# install.packages("caret")
library(caret)
```

```{r message=FALSE}
# Following missing value method 1
head(new_full_data1,)
center_DF1 <- preProcess(new_full_data1,method="range")
normdata1 <-predict(center_DF1,new_full_data1)
head(normdata1)
```

```{r message=FALSE}
# Following missing value method 2
head(new_full_data2,)
center_DF2 <- preProcess(new_full_data2,method="range")
normdata2 <-predict(center_DF2,new_full_data2)
head(normdata2)
```

# Modelling

## Step1:Data Partitioning: Spliting the whole dataset to a training dataset and a testing dataset

### Testing dataset simulates unseen data, showing true predictive power of models. Thus, the testing dataset should not be manipulated or cleaned, and data partitioning is done before cleaning the training dataset.
```{r}
install.packages("caTools")
library(caTools)
```

```{r message=FALSE}
# missing value method 1
set.seed(123)
split1 = sample.split(new_full_data1$target, SplitRatio = 0.70) # Tuning SplitRatio is necessary to identify a spliting threshold that contributes to the best performance of the models.
training_set1 = subset(new_full_data1, split1 == TRUE) 
testing_set1 = subset(new_full_data1, split1 == FALSE) 
```


```{r message=FALSE}
# missing value method 2
set.seed(123)
split2 = sample.split(new_full_data2$target, SplitRatio = 0.70) # Tuning SplitRatio is necessary to identify a spliting threshold that contributes to the best performance of the models.
training_set2 = subset(new_full_data2, split2 == TRUE) 
testing_set2 = subset(new_full_data2, split2 == FALSE) 
```
```{r message=FALSE}
# missing value method 1 & Norm
set.seed(123)
split3 = sample.split(normdata1$target, SplitRatio = 0.70) # Tuning SplitRatio is necessary to identify a spliting threshold that contributes to the best performance of the models.
training_set3 = subset(normdata1, split3 == TRUE) 
testing_set3 = subset(normdata1, split3 == FALSE) 
```

```{r message=FALSE}
# missing value method 2 & Norm
set.seed(123)
split4 = sample.split(normdata2$target, SplitRatio = 0.70) # Tuning SplitRatio is necessary to identify a spliting threshold that contributes to the best performance of the models.
training_set4 = subset(normdata2, split4 == TRUE) 
testing_set4 = subset(normdata2, split4 == FALSE) 
```

## Step2:Feature Selection

### Method 1: Using Information Gain to generate feature informativeness ranking and select the features ranked high.
```{r message=FALSE}
# install.packages("FSelector")
# install.packages("rJava")
library(rJava)
library(FSelector)
```

```{r message=FALSE}
# missing value method 1
ig_variable1 <- information.gain(target~.,training_set1)
print(ig_variable1)
filter(ig_variable1,attr_importance> 0)
# For filter variables that have higher information gain, 0 is changable
```

```{r message=FALSE}
# n for number of features according to filter function(n of variables greater than 0), changable as needed
informative_attribute1 <- cutoff.k(ig_variable1, 137)   

data_for_modelling_1 <- training_set1[informative_attribute1]
data_for_modelling_1$target <- training_set1$target
# put the list of variables in informative_attribute1 into data_for_modeling dataset
```

```{r message=FALSE}
# missing value method 2
ig_variable2 <- information.gain(target~.,training_set2)

filter(ig_variable2,attr_importance>0)
# For filter variables that have higher information gain, 0 is changable
```

```{r message=FALSE}
# n for number of features according to filter function(n of variables greater than 0), changable as needed
informative_attribute2 <- cutoff.k(ig_variable2, 134)   

data_for_modelling_2 <- training_set2[informative_attribute2]
data_for_modelling_2$target <- training_set2$target
# put the list of variables in informative_attribute1 into data_for_modeling dataset
```

## Step3:Data Balancing

### Literatures argue about the effect of different data balancing techniques on various models. In this modeling process, three data balancing methods are tried on the training dataset to compare the performance of models bulit.

```{r message=FALSE}
# install.packages("mltools")
library(mltools)
# install.packages("rpart")
library(rpart)
# install.packages("ROSE")
library(ROSE)
```

```{r message=FALSE}
# data balancing

# missing value method 1
balanceddata_both1 <- ovun.sample(target~., data = data_for_modelling_1, method = "both", p=0.5, seed=123)$data
balanceddata_both2 <- ovun.sample(target~., data = data_for_modelling_2, method = "both", p=0.5, seed=123)$data
balanceddata_both3 <- ovun.sample(target~., data = data_for_modelling_3, method = "both", p=0.5, seed=123)$data
balanceddata_both4 <- ovun.sample(target~., data = data_for_modelling_4, method = "both", p=0.5, seed=123)$data
balanceddata_both1$target <- as.factor(balanceddata_both1$target)
balanceddata_both2$target <- as.factor(balanceddata_both2$target)
balanceddata_both3$target <- as.factor(balanceddata_both3$target)
balanceddata_both4$target <- as.factor(balanceddata_both4$target)
```
## Step4: SVM Modelling

```{r  message=FALSE}
# install.package("e1071")
library(e1071)
```
```{r  message=FALSE}
svm_both1_01  <- svm(target ~. , data = balanceddata_both1, kernel = "radial", scale = TRUE, probability = TRUE, cost= 0.1)
svm_both1_02  <- svm(target ~. , data = balanceddata_both1, kernel = "radial", scale = TRUE, probability = TRUE, cost= 1)
svm_both1_03  <- svm(target ~. , data = balanceddata_both1, kernel = "radial", scale = TRUE, probability = TRUE, cost= 10)
svm_both2_01  <- svm(target ~. , data = balanceddata_both2, kernel = "radial", scale = TRUE, probability = TRUE, cost= 0.1)
svm_both2_02  <- svm(target ~. , data = balanceddata_both2, kernel = "radial", scale = TRUE, probability = TRUE, cost= 1)
svm_both2_03  <- svm(target ~. , data = balanceddata_both2, kernel = "radial", scale = TRUE, probability = TRUE, cost= 10)
svm_both3_01  <- svm(target ~. , data = balanceddata_both3, kernel = "radial", scale = TRUE, probability = TRUE, cost= 0.1)
svm_both3_02  <- svm(target ~. , data = balanceddata_both3, kernel = "radial", scale = TRUE, probability = TRUE, cost= 1)
svm_both3_03  <- svm(target ~. , data = balanceddata_both3, kernel = "radial", scale = TRUE, probability = TRUE, cost= 10)
svm_both4_01  <- svm(target ~. , data = balanceddata_both4, kernel = "radial", scale = TRUE, probability = TRUE, cost= 0.1)
svm_both4_02  <- svm(target ~. , data = balanceddata_both4, kernel = "radial", scale = TRUE, probability = TRUE, cost= 1)
svm_both4_03  <- svm(target ~. , data = balanceddata_both4, kernel = "radial", scale = TRUE, probability = TRUE, cost= 10)
```

## Step5:Naive Bayesian modeling

```{r message=FALSE}
nb.model1 <- naiveBayes(target~.,data = balanceddata_both1)
nb.model2 <- naiveBayes(target~.,data = balanceddata_both2)
nb.model3 <- naiveBayes(target~.,data = balanceddata_both3)
nb.model4 <- naiveBayes(target~.,data = balanceddata_both4)
```
## Step6: Random Forest

```{r message=FALSE}
# install.packages("randomForest")
library(randomForest)
```
```{r}
model_RF1 <- randomForest(target~.,balanceddata_both1,ntree=500, mtry=3)
model_RF2 <- randomForest(target~.,balanceddata_both2,ntree=500, mtry=3)
model_RF3 <- randomForest(target~.,balanceddata_both3,ntree=500, mtry=3)
model_RF4 <- randomForest(target~.,balanceddata_both4,ntree=500, mtry=3)
```

## Step7: XGBoost
```{r}
# install.packages("xgboost")
library(xgboost)
```

```{r}
traindata1_1 <- data.matrix(balanceddata_both1[,-1]) 
traindata1_2 <- Matrix(traindata1_1,sparse=T) 
traindata1_3 <- as.numeric(balanceddata_both1[,1])
traindata1 <- list(data=traindata1_2,label=traindata1_3) 
traindata2_1 <- data.matrix(balanceddata_both2[,-1]) 
traindata2_2 <- Matrix(traindata2_1,sparse=T) 
traindata2_3 <- as.numeric(balanceddata_both2[,1])
traindata2 <- list(data=traindata2_2,label=traindata2_3) 
traindata3_1 <- data.matrix(balanceddata_both3[,-1]) 
traindata3_2 <- Matrix(traindata3_1,sparse=T) 
traindata3_3 <- as.numeric(balanceddata_both3[,1])
traindata3 <- list(data=traindata3_2,label=traindata3_3) 
traindata4_1 <- data.matrix(balanceddata_both4[,-1]) 
traindata4_2 <- Matrix(traindata4_1,sparse=T) 
traindata4_3 <- as.numeric(balanceddata_both4[,1])
traindata4 <- list(data=traindata4_2,label=traindata4_3) 
dtrain1 <- xgb.DMatrix(data = traindata1$data, label = traindata1$label)
dtrain2 <- xgb.DMatrix(data = traindata2$data, label = traindata2$label)
dtrain3 <- xgb.DMatrix(data = traindata3$data, label = traindata3$label)
dtrain4 <- xgb.DMatrix(data = traindata4$data, label = traindata4$label)
testset1_1 <- data.matrix(testing_set1[,-1]) 
testset1_2 <- Matrix(testset1_1,sparse=T) 
testset1_3 <- as.numeric(testing_set2[,1]) 
testset1 <- list(data=testset1_2,label=testset1_3) 
testset2_1 <- data.matrix(testing_set2[,-1]) 
testset2_2 <- Matrix(testset2_1,sparse=T) 
testset2_3 <- as.numeric(testing_set2[,1]) 
testset2 <- list(data=testset2_2,label=testset2_3) 
testset3_1 <- data.matrix(testing_set3[,-1]) 
testset3_2 <- Matrix(testset3_1,sparse=T) 
testset3_3 <- as.numeric(testing_set3[,1]) 
testset3 <- list(data=testset3_2,label=testset3_3) 
testset4_1 <- data.matrix(testing_set4[,-1]) 
testset4_2 <- Matrix(testset4_1,sparse=T) 
testset4_3 <- as.numeric(testing_set4[,1]) 
testset4 <- list(data=testset4_2,label=testset4_3) 
dtest1 <- xgb.DMatrix(data = testset1$data, label = testset1$label) 
dtest2 <- xgb.DMatrix(data = testset2$data, label = testset2$label) 
dtest3 <- xgb.DMatrix(data = testset3$data, label = testset3$label) 
dtest4 <- xgb.DMatrix(data = testset4$data, label = testset4$label) 

xgb_spam1 <- xgboost(data = dtrain1,max_depth=6, eta=0.5,  objective='binary:logistic', nround=25)
xgb_spam2 <- xgboost(data = dtrain2,max_depth=6, eta=0.5,  objective='binary:logistic', nround=25)
xgb_spam3 <- xgboost(data = dtrain3,max_depth=6, eta=0.5,  objective='binary:logistic', nround=25)
xgb_spam4 <- xgboost(data = dtrain4,max_depth=6, eta=0.5,  objective='binary:logistic', nround=25)
```

# Evaluation

## Measurement 1: Expected Value
### Expected value of each model will be calculated, and the one model with the highest Expected value is selected.
### The Expected value measurement reflects all aspects of the prediction by a model 
### When comparing models' performances, Expected value calculation take into account the opportunity cost resulted from false negative prediction

```{r}
### Profit of contacting a customer who is going to make a transaction = revenue bank earns from a customer making the transaction(Value of the transaction: 100000000; net interest margin = 1.59%) - cost of the customer satisfaction program (say, 10000) - Other costs, such as personnal cost, resulting from banking helping realize the transaction (say, 100)
Profit <- 100000000*0.0159-10000-100
# Opportunity cost of FN per customer:
Oppo_cost <- -(Profit)
# Cost of a customer satisfaction program for a customer:
Cost_SatisProgram <- -10000
```

## Step1: SVM
```{r  message=FALSE}
prediction_SVM_1_01 <- predict(svm_both1_01, testing_set1, probability = TRUE)
CM_SVM01 <- confusionMatrix(prediction_SVM_1_01, testing_set1$target, positive='1', mode = "prec_recall")
CM_SVM01[[2]]
CM_SVM01_TN <- CM_SVM01[[2]][1]/nrow(testing_set1)
CM_SVM01_FP <- CM_SVM01[[2]][2]/nrow(testing_set1)
CM_SVM01_FN <- CM_SVM01[[2]][3]/nrow(testing_set1)
CM_SVM01_TP <- CM_SVM01[[2]][4]/nrow(testing_set1)

Expected_value_SVM01 <-
  CM_SVM01_TN*0+CM_SVM01_FP*Cost_SatisProgram+CM_SVM01_FN*Oppo_cost+CM_SVM01_TP*Profit
Expected_value_SVM01
```
```{r  message=FALSE}
prediction_SVM_1_02 <- predict(svm_both1_02, testing_set1, probability = TRUE)
CM_SVM02 <- confusionMatrix(prediction_SVM_1_02, testing_set1$target, positive='1', mode = "prec_recall")
CM_SVM02[[2]]
CM_SVM02_TN <- CM_SVM02[[2]][1]/nrow(testing_set1)
CM_SVM02_FP <- CM_SVM02[[2]][2]/nrow(testing_set1)
CM_SVM02_FN <- CM_SVM02[[2]][3]/nrow(testing_set1)
CM_SVM02_TP <- CM_SVM02[[2]][4]/nrow(testing_set1)

Expected_value_SVM02 <-
  CM_SVM02_TN*0+CM_SVM02_FP*Cost_SatisProgram+CM_SVM02_FN*Oppo_cost+CM_SVM02_TP*Profit
Expected_value_SVM02
```
```{r  message=FALSE}
prediction_SVM_1_03 <- predict(svm_both1_03, testing_set1, probability = TRUE)
CM_SVM03 <- confusionMatrix(prediction_SVM_1_03, testing_set1$target, positive='1', mode = "prec_recall")
CM_SVM03[[2]]
CM_SVM03_TN <- CM_SVM03[[2]][1]/nrow(testing_set1)
CM_SVM03_FP <- CM_SVM03[[2]][2]/nrow(testing_set1)
CM_SVM03_FN <- CM_SVM03[[2]][3]/nrow(testing_set1)
CM_SVM03_TP <- CM_SVM03[[2]][4]/nrow(testing_set1)

Expected_value_SVM03 <-
  CM_SVM03_TN*0+CM_SVM03_FP*Cost_SatisProgram+CM_SVM03_FN*Oppo_cost+CM_SVM03_TP*Profit
Expected_value_SVM03
```
```{r  message=FALSE}
prediction_SVM_2_01 <- predict(svm_both2_01, testing_set2, probability = TRUE)
CM_SVM04 <- confusionMatrix(prediction_SVM_2_01, testing_set2$target, positive='1', mode = "prec_recall")
CM_SVM04[[2]]
CM_SVM04_TN <- CM_SVM04[[2]][1]/nrow(testing_set2)
CM_SVM04_FP <- CM_SVM04[[2]][2]/nrow(testing_set2)
CM_SVM04_FN <- CM_SVM04[[2]][3]/nrow(testing_set2)
CM_SVM04_TP <- CM_SVM04[[2]][4]/nrow(testing_set2)

Expected_value_SVM04 <-
  CM_SVM04_TN*0+CM_SVM04_FP*Cost_SatisProgram+CM_SVM04_FN*Oppo_cost+CM_SVM04_TP*Profit
Expected_value_SVM04
```
```{r  message=FALSE}
prediction_SVM_2_02 <- predict(svm_both2_02, testing_set2, probability = TRUE)
CM_SVM05 <- confusionMatrix(prediction_SVM_2_02, testing_set2$target, positive='1', mode = "prec_recall")
CM_SVM05[[2]]
CM_SVM05_TN <- CM_SVM05[[2]][1]/nrow(testing_set2)
CM_SVM05_FP <- CM_SVM05[[2]][2]/nrow(testing_set2)
CM_SVM05_FN <- CM_SVM05[[2]][3]/nrow(testing_set2)
CM_SVM05_TP <- CM_SVM05[[2]][4]/nrow(testing_set2)

Expected_value_SVM05 <-
  CM_SVM05_TN*0+CM_SVM05_FP*Cost_SatisProgram+CM_SVM05_FN*Oppo_cost+CM_SVM05_TP*Profit
Expected_value_SVM05
```

```{r  message=FALSE}
prediction_SVM_2_03 <- predict(svm_both2_03, testing_set2, probability = TRUE)
CM_SVM06 <- confusionMatrix(prediction_SVM_2_03, testing_set2$target, positive='1', mode = "prec_recall")
CM_SVM06[[2]]
CM_SVM06_TN <- CM_SVM06[[2]][1]/nrow(testing_set2)
CM_SVM06_FP <- CM_SVM06[[2]][2]/nrow(testing_set2)
CM_SVM06_FN <- CM_SVM06[[2]][3]/nrow(testing_set2)
CM_SVM06_TP <- CM_SVM06[[2]][4]/nrow(testing_set2)

Expected_value_SVM06 <-
  CM_SVM06_TN*0+CM_SVM06_FP*Cost_SatisProgram+CM_SVM06_FN*Oppo_cost+CM_SVM06_TP*Profit
Expected_value_SVM06
```
```{r  message=FALSE}
prediction_SVM_3_01 <- predict(svm_both3_01, testing_set3, probability = TRUE)
CM_SVM07 <- confusionMatrix(prediction_SVM_3_01, testing_set3$target, positive='1', mode = "prec_recall")
CM_SVM07[[2]]
CM_SVM07_TN <- CM_SVM07[[2]][1]/nrow(testing_set3)
CM_SVM07_FP <- CM_SVM07[[2]][2]/nrow(testing_set3)
CM_SVM07_FN <- CM_SVM07[[2]][3]/nrow(testing_set3)
CM_SVM07_TP <- CM_SVM07[[2]][4]/nrow(testing_set3)

Expected_value_SVM07 <-
  CM_SVM07_TN*0+CM_SVM07_FP*Cost_SatisProgram+CM_SVM07_FN*Oppo_cost+CM_SVM07_TP*Profit
Expected_value_SVM07
```
```{r  message=FALSE}
prediction_SVM_3_02 <- predict(svm_both3_02, testing_set3, probability = TRUE)
CM_SVM08 <- confusionMatrix(prediction_SVM_3_02, testing_set3$target, positive='1', mode = "prec_recall")
CM_SVM08[[2]]
CM_SVM08_TN <- CM_SVM08[[2]][1]/nrow(testing_set3)
CM_SVM08_FP <- CM_SVM08[[2]][2]/nrow(testing_set3)
CM_SVM08_FN <- CM_SVM08[[2]][3]/nrow(testing_set3)
CM_SVM08_TP <- CM_SVM08[[2]][4]/nrow(testing_set3)

Expected_value_SVM08 <-
  CM_SVM08_TN*0+CM_SVM08_FP*Cost_SatisProgram+CM_SVM08_FN*Oppo_cost+CM_SVM08_TP*Profit
Expected_value_SVM08
```
```{r  message=FALSE}
prediction_SVM_3_03 <- predict(svm_both3_03, testing_set3, probability = TRUE)
CM_SVM09 <- confusionMatrix(prediction_SVM_3_03, testing_set3$target, positive='1', mode = "prec_recall")
CM_SVM09[[2]]
CM_SVM09_TN <- CM_SVM09[[2]][1]/nrow(testing_set3)
CM_SVM09_FP <- CM_SVM09[[2]][2]/nrow(testing_set3)
CM_SVM09_FN <- CM_SVM09[[2]][3]/nrow(testing_set3)
CM_SVM09_TP <- CM_SVM09[[2]][4]/nrow(testing_set3)

Expected_value_SVM09 <-
  CM_SVM09_TN*0+CM_SVM09_FP*Cost_SatisProgram+CM_SVM09_FN*Oppo_cost+CM_SVM09_TP*Profit
Expected_value_SVM09
```
```{r  message=FALSE}
prediction_SVM_4_01 <- predict(svm_both4_01, testing_set4, probability = TRUE)
CM_SVM10 <- confusionMatrix(prediction_SVM_4_01, testing_set4$target, positive='1', mode = "prec_recall")
CM_SVM010[[2]]
CM_SVM10_TN <- CM_SVM10[[2]][1]/nrow(testing_set4)
CM_SVM10_FP <- CM_SVM10[[2]][2]/nrow(testing_set4)
CM_SVM10_FN <- CM_SVM10[[2]][3]/nrow(testing_set4)
CM_SVM10_TP <- CM_SVM10[[2]][4]/nrow(testing_set4)

Expected_value_SVM10 <-
  CM_SVM10_TN*0+CM_SVM10_FP*Cost_SatisProgram+CM_SVM10_FN*Oppo_cost+CM_SVM10_TP*Profit
Expected_value_SVM10
```
```{r  message=FALSE}
prediction_SVM_4_02 <- predict(svm_both4_02, testing_set4, probability = TRUE)
CM_SVM11 <- confusionMatrix(prediction_SVM_4_02, testing_set4$target, positive='1', mode = "prec_recall")
CM_SVM11[[2]]
CM_SVM11_TN <- CM_SVM11[[2]][1]/nrow(testing_set4)
CM_SVM11_FP <- CM_SVM11[[2]][2]/nrow(testing_set4)
CM_SVM11_FN <- CM_SVM11[[2]][3]/nrow(testing_set4)
CM_SVM11_TP <- CM_SVM11[[2]][4]/nrow(testing_set4)

Expected_value_SVM11 <-
  CM_SVM11_TN*0+CM_SVM11_FP*Cost_SatisProgram+CM_SVM11_FN*Oppo_cost+CM_SVM11_TP*Profit
Expected_value_SVM11
```
```{r  message=FALSE}
prediction_SVM_4_03 <- predict(svm_both4_03, testing_set4, probability = TRUE)
CM_SVM12 <- confusionMatrix(prediction_SVM_4_03, testing_set4$target, positive='1', mode = "prec_recall")
CM_SVM12[[2]]
CM_SVM12_TN <- CM_SVM12[[2]][1]/nrow(testing_set4)
CM_SVM12_FP <- CM_SVM12[[2]][2]/nrow(testing_set4)
CM_SVM12_FN <- CM_SVM12[[2]][3]/nrow(testing_set4)
CM_SVM12_TP <- CM_SVM12[[2]][4]/nrow(testing_set4)

Expected_value_SVM12 <-
  CM_SVM12_TN*0+CM_SVM12_FP*Cost_SatisProgram+CM_SVM12_FN*Oppo_cost+CM_SVM12_TP*Profit
Expected_value_SVM12
```
## Step2:Naive Bayesian
```{r}
nb_predict1 <- predict(nb.model1,newdata = testing_set1)
CM_NB1 <- confusionMatrix(nb_predict1,testing_set1$target,positive="1",mode="prec_recall")
CM_NB1[[2]]
CM_NB1_TN <- CM_NB1[[2]][1]/nrow(testing_set1)
CM_NB1_FP <- CM_NB1[[2]][2]/nrow(testing_set1)
CM_NB1_FN <- CM_NB1[[2]][3]/nrow(testing_set1)
CM_NB1_TP <- CM_NB1[[2]][4]/nrow(testing_set1)

Expected_value_NB1 <-
  CM_NB1_TN*0+CM_NB1_FP*Cost_SatisProgram+CM_NB1_FN*Oppo_cost+CM_NB1_TP*Profit
Expected_value_NB1
```
```{r}
nb_predict2 <- predict(nb.model2,newdata = testing_set2)
CM_NB2 <- confusionMatrix(nb_predict2,testing_set2$target,positive="1",mode="prec_recall")
CM_NB2[[2]]
CM_NB2_TN <- CM_NB2[[2]][1]/nrow(testing_set2)
CM_NB2_FP <- CM_NB2[[2]][2]/nrow(testing_set2)
CM_NB2_FN <- CM_NB2[[2]][3]/nrow(testing_set2)
CM_NB2_TP <- CM_NB2[[2]][4]/nrow(testing_set2)

Expected_value_NB2 <-
  CM_NB2_TN*0+CM_NB2_FP*Cost_SatisProgram+CM_NB2_FN*Oppo_cost+CM_NB2_TP*Profit
Expected_value_NB2
```
```{r}
nb_predict3 <- predict(nb.model3,newdata = testing_set3)
CM_NB3 <- confusionMatrix(nb_predict3,testing_set3$target,positive="1",mode="prec_recall")
CM_NB3[[2]]
CM_NB3_TN <- CM_NB3[[2]][1]/nrow(testing_set3)
CM_NB3_FP <- CM_NB3[[2]][2]/nrow(testing_set3)
CM_NB3_FN <- CM_NB3[[2]][3]/nrow(testing_set3)
CM_NB3_TP <- CM_NB3[[2]][4]/nrow(testing_set3)

Expected_value_NB3 <-
  CM_NB3_TN*0+CM_NB3_FP*Cost_SatisProgram+CM_NB3_FN*Oppo_cost+CM_NB3_TP*Profit
Expected_value_NB3
```
```{r}
nb_predict4 <- predict(nb.model4,newdata = testing_set4)
CM_NB4 <- confusionMatrix(nb_predict4,testing_set4$target,positive="1",mode="prec_recall")
CM_NB4[[2]]
CM_NB4_TN <- CM_NB4[[2]][1]/nrow(testing_set4)
CM_NB4_FP <- CM_NB4[[2]][2]/nrow(testing_set4)
CM_NB4_FN <- CM_NB4[[2]][3]/nrow(testing_set4)
CM_NB4_TP <- CM_NB4[[2]][4]/nrow(testing_set4)

Expected_value_NB4 <-
  CM_NB4_TN*0+CM_NB4_FP*Cost_SatisProgram+CM_NB4_FN*Oppo_cost+CM_NB4_TP*Profit
Expected_value_NB4
```

## Step3: Random Forest
```{r}
prediction_RF1 <- predict(model_RF1,testing_set1)
CM_RF1 <- confusionMatrix(class_prediction_RF1,testing_set1$target,positive="1",mode="prec_recall")
CM_RF1[[2]]
CM_RF1_TN <- CM_RF1[[2]][1]/nrow(testing_set1)
CM_RF1_FP <- CM_RF1[[2]][2]/nrow(testing_set1)
CM_RF1_FN <- CM_RF1[[2]][3]/nrow(testing_set1)
CM_RF1_TP <- CM_RF1[[2]][4]/nrow(testing_set1)
Expected_value_RF1 <-
  CM_RF1_TN*0+CM_RF1_FP*Cost_SatisProgram+CM_RF1_FN*Oppo_cost+CM_RF1_TP*Profit
Expected_value_RF1
```
```{r}
prediction_RF2 <- predict(model_RF2,testing_set2)
CM_RF2 <- confusionMatrix(class_prediction_RF2,testing_set2$target,positive="1",mode="prec_recall")
CM_RF2[[2]]
CM_RF2_TN <- CM_RF2[[2]][1]/nrow(testing_set2)
CM_RF2_FP <- CM_RF2[[2]][2]/nrow(testing_set2)
CM_RF2_FN <- CM_RF2[[2]][3]/nrow(testing_set2)
CM_RF2_TP <- CM_RF2[[2]][4]/nrow(testing_set2)
Expected_value_RF2 <-
  CM_RF2_TN*0+CM_RF2_FP*Cost_SatisProgram+CM_RF2_FN*Oppo_cost+CM_RF2_TP*Profit
Expected_value_RF2
```
```{r}
prediction_RF3<- predict(model_RF3,testing_set3)
CM_RF3 <- confusionMatrix(class_prediction_RF3,testing_set3$target,positive="1",mode="prec_recall")
CM_RF3[[2]]
CM_RF3_TN <- CM_RF3[[2]][1]/nrow(testing_set3)
CM_RF3_FP <- CM_RF3[[2]][2]/nrow(testing_set3)
CM_RF3_FN <- CM_RF3[[2]][3]/nrow(testing_set3)
CM_RF3_TP <- CM_RF3[[2]][4]/nrow(testing_set3)
Expected_value_RF3 <-
  CM_RF3_TN*0+CM_RF3_FP*Cost_SatisProgram+CM_RF3_FN*Oppo_cost+CM_RF3_TP*Profit
Expected_value_RF3
```
```{r}
prediction_RF4<- predict(model_RF4,testing_set4)
CM_RF4 <- confusionMatrix(class_prediction_RF4,testing_set4$target,positive="1",mode="prec_recall")
CM_RF4[[2]]
CM_RF4_TN <- CM_RF4[[2]][1]/nrow(testing_set4)
CM_RF4_FP <- CM_RF4[[2]][2]/nrow(testing_set4)
CM_RF4_FN <- CM_RF4[[2]][3]/nrow(testing_set4)
CM_RF4_TP <- CM_RF4[[2]][4]/nrow(testing_set4)
Expected_value_RF4 <-
  CM_RF4_TN*0+CM_RF4_FP*Cost_SatisProgram+CM_RF4_FN*Oppo_cost+CM_RF4_TP*Profit
Expected_value_RF4
```

## Step 4:XGBOOST
```{r}
pre_xgb1 = predict(xgb_spam1,newdata = dtest1))
CM1<- table(test1_3,pre_xgb1,dnn=c("true","pre"))
CM1_TN <- CM1[[2]][1]/nrow(dtest1)
CM1_FP <- CM1[[2]][2]/nrow(dtest1)
CM1_FN <- CM1[[2]][3]/nrow(dtest1)
CM1_TP <- CM1[[2]][4]/nrow(dtest1)
Expected_value_CM1 <-
  CM1_TN*0+CM1_FP*Cost_SatisProgram+CM1_FN*Oppo_cost+CM1_TP*Profit
Expected_value_CM1
```
```{r}
pre_xgb2 = predict(xgb_spam2,newdata = dtest2))
CM2<- table(test2_3,pre_xgb2,dnn=c("true","pre"))
CM2_TN <- CM2[[2]][1]/nrow(dtest2)
CM2_FP <- CM2[[2]][2]/nrow(dtest2)
CM2_FN <- CM2[[2]][3]/nrow(dtest2)
CM2_TP <- CM2[[2]][4]/nrow(dtest2)
Expected_value_CM2 <-
  CM2_TN*0+CM2_FP*Cost_SatisProgram+CM2_FN*Oppo_cost+CM2_TP*Profit
Expected_value_CM2
```
```{r}
pre_xgb3 = predict(xgb_spam3,newdata = dtest3))
CM3<- table(test3_3,pre_xgb1,dnn=c("true","pre"))
CM3_TN <- CM3[[2]][1]/nrow(dtest3)
CM3_FP <- CM3[[2]][2]/nrow(dtest3)
CM3_FN <- CM3[[2]][3]/nrow(dtest3)
CM3_TP <- CM3[[2]][4]/nrow(dtest3)
Expected_value_CM3<-
  CM3_TN*0+CM3_FP*Cost_SatisProgram+CM3_FN*Oppo_cost+CM3_TP*Profit
Expected_value_CM3
```
```{r}
pre_xgb4 = predict(xgb_spam4,newdata = dtest4))
CM4<- table(test4_3,pre_xgb1,dnn=c("true","pre"))
CM4_TN <- CM4[[2]][1]/nrow(dtest4)
CM4_FP <- CM4[[2]][2]/nrow(dtest4)
CM4_FN <- CM4[[2]][3]/nrow(dtest4)
CM4_TP <- CM4[[2]][4]/nrow(dtest4)
Expected_value_CM4 <-
  CM4_TN*0+CM4_FP*Cost_SatisProgram+CM4_FN*Oppo_cost+CM4_TP*Profit
Expected_value_CM4
```




