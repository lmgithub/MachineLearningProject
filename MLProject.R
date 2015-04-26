# Init
rm(list=ls())
setwd("/home/lm/Projects/MachineLearning/Project")

# loadind all packages
library("caret")

# Getting data
urlTrain <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
urlTest <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
fileTrain <- "Data/train.csv"
fileTest <- "Data/test.csv"
if (!file.exists(fileTrain)) {
    download.file(urlTrain, fileTrain, method="curl")
}
if (!file.exists(fileTest)) {
    download.file(urlTest, fileTest, method="curl")
}
dfTrainSource <- read.csv(fileTrain, na.strings=c("", "NA", "#DIV/0!"))

# Preparing data
# Splitting data for cross validation (60% for training and 40% for testing)
set.seed(1245)
inTrain <- createDataPartition(dfTrainSource$classe, p=0.6, list=FALSE)
dfTrain <- dfTrainSource[inTrain, ]
dfTest <- dfTrainSource[-inTrain, ]

# Removing Near Zero Variables
colsNZV <- nearZeroVar(dfTrain)
# names(dfTrain)[nzv]
dfTrain2 <- subset(dfTrain, select=-colsNZV)

# Removing columns with NA > 50%
sumNACols <- sapply(dfTrain2, function(x){sum(is.na(x))})
colsNNA <- sumNACols < nrow(dfTrain2)/2
dfTrain2 <-  subset(dfTrain2, select=colsNNA)

# Select columns with highest correlaton
# Transforming classe to integer
iclasse <- 
    ifelse(dfTrain2$classe=="A", 1, 
    ifelse(dfTrain2$classe=="B", 2,  
    ifelse(dfTrain2$classe=="C", 3, 
    ifelse(dfTrain2$classe=="D", 4,
    ifelse(dfTrain2$classe=="E", 5, NA
    )))))
# Making dataset with numeric data only
classCols <- sapply(dfTrain2, class)
colsNumeric <-  classCols=="numeric"|classCols=="integer"
dfTrainNum <- subset(dfTrain2, select=colsNumeric)
# Calculating correlation
corClasse <- cor(iclasse, dfTrainNum)
names(corClasse) <- names(dfTrainNum)
corClasse <- corClasse[1, ]
corClasseSoted <- sort(abs(corClasse))
summary(corClasseSoted)
# Removing columns with low and NA correlation
# (keeping all non-numeric columns and numeric columns with correlation > mean)
colsRemove <- c(
    names(corClasseSoted[corClasseSoted<=mean(corClasseSoted)]), # columns with correlation < 0.1
    names(corClasse[is.na(corClasse)]) # columns with NA correlation
)
dfTrainPrepared <- dfTrain2[, !names(dfTrain2) %in% colsRemove]
dfTrainPrepared <- dfTrain2

# Preparing 
dfTestPrepared <- dfTest[, names(dfTrainPrepared)]
dfTestPrepared <- subset(dfTestPrepared, select=-classe)

# Building models
# cm$overall

modF <- rpart(classe ~ ., data=dfTrainPrepared, method="class")
predF <- predict(modF, newdata=dfTestPrepared, type="class")
cm <- confusionMatrix(predF, dfTest$classe)
cm$table
cm$overall[1]

# Fitting a model
modRF <- randomForest(classe ~ ., data=dfTrainPrepared) 
# Predicting 
predRF <- predict(modRF, newdata=dfTestPrepared, type="class")
# Comparing predicted and actual values
cmRF <- confusionMatrix(predRF, dfTest$classe)

mod1 <- train(classe ~ ., method="rpart", data=dfTrainPrepared, type="class")
pred1 <- predict(mod1, newdata=dfTestPrepared)
cm1 <- confusionMatrix(pred1, dfTest$classe)
cm1

mod2 <- train(classe ~ ., method="rpart2", data=dfTrainPrepared)
pred2 <- predict(mod2, newdata=dfTestPrepared)
cm2 <- confusionMatrix(pred2, dfTest$classe)
cm2

mod3 <- train(classe ~ ., method="glm", data=dfTrainPrepared)
pred3 <- predict(mod3, newdata=dfTestPrepared)
cm3 <- confusionMatrix(pred3, dfTest$classe)
cm3

predRPart <- processModel("rpart")
predRPart2 <- processModel("rpart2")
predRF <- processModel("rf")




processModel <- function(pMethod){
    pModel <- train(classe ~ ., method=pMethod, data=dfTrainPrepared)
    pPred <- predict(pModel, newdata=dfTestPrepared)
    cm <- confusionMatrix(pred1, dfTest$classe)
    print(cm)
    pPred
}

processModel1 <- function(lmethod){
    mod1 <- train(classe ~ ., method=lmethod, data=dfTrainPrepared)
    pred1 <- predict(mod1, newdata=dfTestPrepared)
    cm <- confusionMatrix(pred1, dfTest$classe)
    cm$overall
}
processModel2 <- function(index){
    mod1 <- train(classe ~ ., method=dfPred$Method[index], data=dfTrainPrepared)
    pred1 <- predict(mod1, newdata=dfTestPrepared)
    cm <- confusionMatrix(pred1, dfTest$classe)
    cm$overall
}

processModel("rpart")

dfPred <- data.frame(Method="rpart", Accuracy=0, Kappa=0, AccuracyLower=0, AccuracyUpper=0, AccuracyNull=0, AccuracyPValue=0, McnemarPValue=0)
dfPred[, 2:8] <- processModel(dfPred$Method)
dfPred$Mod <- mod1
#=======================
#dfTrain3 <- dfTrain2[, -c(2, 5, 131)]

