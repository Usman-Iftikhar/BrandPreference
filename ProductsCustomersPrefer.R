# Title: Classification: Predict which Brand of Products Customers Prefer

# Last update: 2019.01.06

# File/project name: ComputerBrand/ProductsCustomersPrefer.R
# RStudio Project name: Customer Brand Preference

###############
# PROJECT NOTES
###############

# Summarize project: Predict which of the two brands of computers customers prefer.
# Use the CompleteResponse.csv to train and SurveyIncomplete.csv to test and predict

# Summarize top model and/or filtered dataset

# x <- 5
# Assignment "<-" short-cut: 
#   OSX [Alt]+[-] (next to "+" sign)
#   Win [Alt]+[-] 


###############
# HOUSEKEEPING
###############

# CLEAR OBJECTS IF NECESSARY
rm(list = ls())

# GET WORKING DIRECTORY
getwd()

# SET WORKING DIRECTORY
setwd("C:/Users/muift/Documents/R_Projects/ComputerBrand")
dir()


################
# LOAD PACKAGES
################

#install.packages("caret")
#install.packages("corrplot")
#install.packages("readr")
#install.packages("doParallel")
#install.packages("C50")
#install.packages("inum")
#install.packages("knitr")
#install.packages("plyr")
#require("devtools")
library(caret)
library(corrplot)
library(C50)
#library(doMC) # FOR osx
library(doParallel)
library(mlbench)
library(readr)
library(parallel)
library(plyr)
library(knitr)

#####################
# PARALLEL PROCESSING
#####################

# NOTE: Be sure to use the correct package for your operating system.

#--- for OSX ---#
#install.packages("doMC")  # install in 'Load packages' section above 
#library(doMC)
#detectCores()   # detect number of cores
#registerDoMC(cores = 2)  # set number of cores (don't use all available)

#--- for Win ---#
#install.packages("doParallel") # install in 'Load packages' section above
#library(doParallel)  # load in the 'Load Packages' section above
detectCores()  # detect number of cores
cl <- makeCluster(2)  # select number of cores
registerDoParallel(cl) # register cluster
getDoParWorkers()  # confirm number of cores being used by RStudio
# Stop Cluster. After performing your tasks, make sure to stop your cluster. 
stopCluster(cl)


###############
# IMPORT DATA
##############

#--- LOAD RAW DATASETS ---#

## Load Train/Existing data (Dataset 1)
completeResponses <- read.csv("completeResponses.csv", stringsAsFactors = FALSE)
class(completeResponses)  # "data.frame"

## Load Predict/New data (Dataset 2) ---#
surveyIncomplete <- read.csv("surveyIncomplete.csv", stringsAsFactors = FALSE)
class(surveyIncomplete)  # "data.frame"


#--- Load preprocessed datasets ---#
# No preprocessed data to load

#ds_name <- read.csv("dataset_name.csv", stringsAsFactors = FALSE) 


################
# EVALUTE DATA
################

#--- DATASET 1 ---#
str(completeResponses)  
#'data.frame':	9898 obs. of  7 variables:
#$ salary : num  119807 106880 78021 63690 50874 ...
#$ age    : int  45 63 23 51 20 56 24 62 29 41 ...
#$ elevel : int  0 1 0 3 3 3 4 3 4 1 ...
#$ car    : int  14 11 15 6 14 14 8 3 17 5 ...
#$ zipcode: int  4 6 2 5 4 3 5 0 0 4 ...
#$ credit : num  442038 45007 48795 40889 352951 ...
#$ brand  : int  0 1 0 1 0 1 1 1 0 1 ...

names(completeResponses)
#[1] "salary"  "age"     "elevel"  "car"     "zipcode" "credit"  "brand"  

summary(completeResponses)
#         salary            age           elevel           car           zipcode          credit           brand       
#Min.   : 20000   Min.   :20.00   Min.   :0.000   Min.   : 1.00   Min.   :0.000   Min.   :     0   Min.   :0.0000  
#1st Qu.: 52082   1st Qu.:35.00   1st Qu.:1.000   1st Qu.: 6.00   1st Qu.:2.000   1st Qu.:120807   1st Qu.:0.0000  
#Median : 84950   Median :50.00   Median :2.000   Median :11.00   Median :4.000   Median :250607   Median :1.0000  
#Mean   : 84871   Mean   :49.78   Mean   :1.983   Mean   :10.52   Mean   :4.041   Mean   :249176   Mean   :0.6217  
#3rd Qu.:117162   3rd Qu.:65.00   3rd Qu.:3.000   3rd Qu.:15.75   3rd Qu.:6.000   3rd Qu.:374640   3rd Qu.:1.0000  
#Max.   :150000   Max.   :80.00   Max.   :4.000   Max.   :20.00   Max.   :8.000   Max.   :500000   Max.   :1.0000

head(completeResponses)
#     salary age elevel car   zipcode   credit brand
#1 119806.54  45      0  14       4 442037.71     0
#2 106880.48  63      1  11       6  45007.18     1
#3  78020.75  23      0  15       2  48795.32     0
#4  63689.94  51      3   6       5  40888.88     1
#5  50873.62  20      3  14       4 352951.50     0
#6 130812.74  56      3  14       3 135943.02     1

tail(completeResponses)
#        salary age elevel car zipcode   credit brand
#9893  28751.26  60      2  10       0      0.0     1
#9894  87580.91  75      1  18       8 282511.9     1
#9895 129181.38  75      2   7       4 384871.4     1
#9896  97828.09  66      2  15       0 399446.7     1
#9897  20000.00  24      1  14       1 223204.6     1
#9898  96430.16  34      1   2       7 224029.8     0

# PLOTS
#hist(WholeYear$SolarRad)
#plot(WholeYear$TimeofDay, WholeYear$SolarRad)
#qqnorm(WholeYear$SolarRad)

# CHECK FOR MISSING VALUES
anyNA(completeResponses)
#[1] FALSE

is.na(completeResponses)


#--- DATASET 2 ---#

# If there is a dataset with unseen data to make predictions on, then preprocess here
# to make sure that it is preprossed the same as the training dataset.

str(surveyIncomplete)  
#'data.frame':	5000 obs. of  7 variables:
#$ salary : num  150000 82524 115647 141443 149211 ...
#$ age    : int  76 51 34 22 56 26 64 50 26 46 ...
#$ elevel : int  1 1 0 3 0 4 3 3 2 3 ...
#$ car    : int  3 8 10 18 5 12 1 9 3 18 ...
#$ zipcode: int  3 3 2 2 3 1 2 0 4 6 ...
#$ credit : num  377980 141658 360980 282736 215667 ...
#$ brand  : int  1 0 1 1 1 1 1 1 1 0 ...

names(surveyIncomplete)
#[1] "salary"  "age"     "elevel"  "car"     "zipcode" "credit"  "brand"  

summary(surveyIncomplete)
#     salary            age            elevel           car          zipcode          credit           brand       
#Min.   : 20000   Min.   :20.00   Min.   :0.000   Min.   : 1.0   Min.   :0.000   Min.   :     0   Min.   :0.0000  
#1st Qu.: 52590   1st Qu.:35.00   1st Qu.:1.000   1st Qu.: 6.0   1st Qu.:2.000   1st Qu.:122311   1st Qu.:0.0000  
#Median : 86221   Median :50.00   Median :2.000   Median :11.0   Median :4.000   Median :250974   Median :0.0000  
#Mean   : 85794   Mean   :49.94   Mean   :2.009   Mean   :10.6   Mean   :4.038   Mean   :249546   Mean   :0.0126  
#3rd Qu.:118535   3rd Qu.:65.00   3rd Qu.:3.000   3rd Qu.:16.0   3rd Qu.:6.000   3rd Qu.:375653   3rd Qu.:0.0000  
#Max.   :150000   Max.   :80.00   Max.   :4.000   Max.   :20.0   Max.   :8.000   Max.   :500000   Max.   :1.0000 

head(surveyIncomplete)
#     salary age elevel car zipcode   credit brand
#1 150000.00  76      1   3       3 377980.1     1
#2  82523.84  51      1   8       3 141657.6     0
#3 115646.64  34      0  10       2 360980.4     1
#4 141443.39  22      3  18       2 282736.3     1
#5 149211.27  56      0   5       3 215667.3     1
#6  46202.25  26      4  12       1 150419.4     1

tail(surveyIncomplete)
#     salary    age elevel car zipcode    credit brand
#4995  29945.49  75      2   9       1 170179.21     0
#4996  83891.56  52      2  14       5  28685.23     0
#4997 125979.29  71      0  12       7 276614.83     0
#4998  74064.71  24      2   2       2 202279.58     0
#4999 106485.57  46      3  16       0 381242.09     0
#5000  50333.58  70      1   5       5 224871.17     0

# PLOTS
#hist(WholeYear$SolarRad)
#plot(WholeYear$TimeofDay, WholeYear$SolarRad)
#qqnorm(WholeYear$SolarRad)

# CHECK FOR MISSING VALUES
anyNA(surveyIncomplete)
#[1] FALSE

is.na(surveyIncomplete)


#############
# PROPROCESS
#############

#--- DATASET 1 ---#

# RENAME A COLUMN
#names(DatasetName)<-c("ColumnName","ColumnName","ColumnName") 

# HANDLE MISSING VALUES 
na.omit(DatasetName$ColumnName)
na.exclude(DatasetName$ColumnName)        
DatasetName$ColumnName[is.na(DatasetName$ColumnName)] <- mean(DatasetName$ColumnName,na.rm = TRUE)

# DISCRETIZE (if applicable)

# RECODE VALUES
completeResponses$brand[completeResponses$brand == 0] <- "Acer"
completeResponses$brand[completeResponses$brand == 1] <- "Sony"

# CHANGE DATA TYPES 
completeResponses$brand <- as.factor(completeResponses$brand)

#--- Dataset 2 ---#

# RECODE VALUES
surveyIncomplete$brand[surveyIncomplete$brand == 0] <- "Acer"
surveyIncomplete$brand[surveyIncomplete$brand == 1] <- "Sony"

# CHANGE DATA TYPES 
surveyIncomplete$brand <- as.factor(surveyIncomplete$brand)


#################
# FEATURE REMOVAL
#################

#--- Dataset 1 ---#

# REMOVE ID AND OBVIOUS FEATURES
# create 7v ds 
#WholeYear7v <- WholeYear
#WholeYear7v$X <- NULL   # remove ID
#str(WholeYear7v)


# SAVE DATASET
#write.csv(WholeYear7v, file = "wholeYear7v.csv")

# OPEN DATASET
#read.csv("wholeYear7v.csv")

#--- DATASET 2 ---#



################
# SAMPLING
################

# ---- SAMPLING ---- #

# Note: For this task, use the 1000 sample, and not the 20%

# 1K SAMPLE
#WholeYear7v1k <- WholeYear7v[sample(1:nrow(WholeYear7v), 1000, replace=FALSE),]

#head(WholeYear7v1k) # ensure randomness

#nrow(WholeYear7v1k) # ensure number of obs

# CREATE 10% SAMPLE FOR 7V DS
#set.seed(998) # set random seed
#WholeYear7v10p <- WholeYear7v[sample(1:nrow(WholeYear7v), round(nrow(WholeYear)*.1), replace=FALSE),]

#nrow(WholeYear7v10p)


#head(WholeYear7v10p) # ensure randomness


##################
# TRAIN / TEST SETS
##################

# CREATE THE TRAINING PARTITION THAT IS 75% OF TOTAL OBS
set.seed(123) # set random seed
inTraining <- createDataPartition(completeResponses$brand, 
                                  p = 0.75, 
                                  list = FALSE)
str(inTraining)
#int [1:7424, 1] 1 3 5 6 7 8 9 10 11 12 ...
#- attr(*, "dimnames")=List of 2
#..$ : NULL
#..$ : chr "Resample1"

# CREATE TRAINING/TESTING DATASET
trainSet <- completeResponses[inTraining,]  
testSet <- completeResponses[-inTraining,]  

# VERIFY NUMBER OF OBSERVATIONS
str(trainSet) #'data.frame':	7424 obs. of  7 variables:
str(testSet) #'data.frame':	2474 obs. of  7 variables:


################
# TRAIN CONTROL
################

# ---- AUTOMATIC GRID C5.0 ---- #

# SET 10 FOLD CROSS VALIDATION
fitControl <- trainControl(method = "repeatedcv", 
                           number = 10, 
                           repeats = 1)

# ---- MANUAL GRID RANDOM FOREST  - 5 different mtry values---- #

rfGrid <- expand.grid(mtry=c(1,2,3,4,5))

# ---- RANDOM SEARCH RANDOM FOREST ---- #
# ---- EXAMPLES ---- #

##############
# TRAIN MODEL
##############

## ------- C5.0 AUTOMATIC GRID ------- ##

# C5.0 TRAIN/FIT

# AUTOMATIC GRID - EXAMPLE 1
set.seed(123)
system.time(c50Fit1 <- train(brand~., 
                            data=trainSet, 
                            method = "C5.0", 
                            trControl = fitControl,
                            tuneLength = 2))
c50Fit1
#C5.0 

#7424 samples
#6 predictor
#2 classes: 'Acer', 'Sony' 

#No pre-processing
#Resampling: Cross-Validated (10 fold, repeated 1 times) 
#Summary of sample sizes: 6681, 6682, 6683, 6681, 6681, 6681, ... 
#Resampling results across tuning parameters:
  
#  model  winnow  trials  Accuracy   Kappa    
#rules  FALSE    1      0.8476448  0.6941408
#rules  FALSE   10      0.9206645  0.8311749
#rules   TRUE    1      0.8460257  0.6862461
#rules   TRUE   10      0.9244352  0.8388092
#tree   FALSE    1      0.8450841  0.6848557
#tree   FALSE   10      0.9218744  0.8341763
#tree    TRUE    1      0.8460257  0.6862465
#tree    TRUE   10      0.9255147  0.8418161

#Accuracy was used to select the optimal model using the largest value.
#The final values used for the model were trials = 10, model = tree and winnow = TRUE.

varImp(c50Fit1)
#C5.0 variable importance

#Overall
#age      100.00
#salary   100.00
#elevel     1.56
#zipcode    0.00
#car        0.00
#credit     0.00

## ------- RF MANUAL GRID------- ##

# RF TRAIN/FIT

set.seed(123)
system.time(rfFitm5 <- train(brand~., 
                             data=trainSet, 
                             method = "rf", 
                             trControl = fitControl,
                             tuneGrid = rfGrid))
rfFitm5
#Random Forest 

#7424 samples
#6 predictor
#2 classes: 'Acer', 'Sony' 

#No pre-processing
#Resampling: Cross-Validated (10 fold, repeated 1 times) 
#Summary of sample sizes: 6681, 6682, 6683, 6681, 6681, 6681, ... 
#Resampling results across tuning parameters:
  
#  mtry  Accuracy   Kappa    
#1     0.8662503  0.7070784
#2     0.9218771  0.8343207
#3     0.9221450  0.8346689
#4     0.9198548  0.8297298
#5     0.9197204  0.8294954

#Accuracy was used to select the optimal model using the largest value.
#The final value used for the model was mtry = 3.

varImp(rfFitm5)
#rf variable importance

#Overall
#salary  100.000
#age      62.450
#credit    8.556
#car       3.037
#zipcode   1.323
#elevel    0.000

#################
# EVALUATE MODELS
#################

##--- COMPARE MODELS ---##

# USE RESAMPLES TO COMPARE MODEL PERFORMANCE
ModelFitResults <- resamples(list(rf = rfFitm5, 
                                  C50DT = c50Fit1))

# OUTPUT SUMMARY METRICS FOR TUNED MODELS 
summary(ModelFitResults)
# ds (completeResponses) Make a note of the dataset that the performance metrics belong to.
# Note performance metrics. Add the summary output as a comment.

#Call:
#summary.resamples(object = ModelFitResults)

#Models: rf, C50DT 
#Number of resamples: 10 

#Accuracy 
#           Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#rf    0.9057873 0.9160464 0.9231806 0.9221450 0.9272972 0.9367429    0
#C50DT 0.9044415 0.9218881 0.9272237 0.9255147 0.9316278 0.9448183    0

#Kappa 
#           Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#rf    0.8016233 0.8221632 0.8363644 0.8346689 0.8453766 0.8657850    0
#C50DT 0.7980924 0.8328466 0.8451143 0.8418161 0.8552859 0.8829188    0


##--- CONCLUSION ---##
# Note which model is top model, and why.
# Best model: Both are about the same, pick c50Fit1



##--- SAVE TOP PERFORMING MODEL ---##

# SAVE MODEL 
saveRDS(c50Fit1, file = "completeResponsesBestModel.rds")  
# Q: What type of object does saveRDS create?
# function to write a single R object to a file, and to restore it

# load and name model to make predictions with new data
c50Fit1 <- readRDS("completeResponsesBestModel.rds") # Q: What type of object does readRDS create?



########################
# PREDICT WITH TOP MODEL
########################

# MAKE PREDICTIONS
c50Pred1 <- predict(c50Fit1, testSet)
c50Pred2 <- predict(c50Fit1, surveyIncomplete)

# PERFORMANCE MEASUREMENT

# FOR TEST TEST
postResample(c50Pred1, testSet$brand)

#Accuracy     Kappa 
#0.9183508 0.8259877

#plot predicted verses actual
plot(c50Pred1,testSet$brand)

# ----- CONFUSION MATRIX ----- #

table(c50Pred1,testSet$brand)
#c50Pred1 Acer Sony
#Acer  829   95
#Sony  107 1443

table(c50Pred2,surveyIncomplete$brand)
#c50Pred2 Acer Sony
#Acer 1866    4
#Sony 3071   59

# ----- PRINT PREDICTIONS ----- #
c50Pred1
c50Pred2

# ----- COUNT PREDICTED BRAND PREFERENCES ----- #

# TEST SET COUNT
count(c50Pred1)
#     x freq
#1 Acer  924
#2 Sony 1550

# SURVEYINCOMPLETE SET COUNT
count(c50Pred2)
#     x freq
#1 Acer 1870
#2 Sony 3130

count(completeResponses, 'brand')
#  brand freq
#1  Acer 3744
#2  Sony 6154