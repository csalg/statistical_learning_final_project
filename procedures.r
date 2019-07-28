library(RSSL)
library(functional)
library(plyr)
library(dplyr)
library(caret)

source("config.r")

# ---------   ALLOW PARALLELIZATION TO SPAWN WORKERS --------- #

#install.packages("doParallel")
library("doParallel")

cl <- makePSOCKcluster(NUM_CORES)
registerDoParallel(cl)



# ---------   HELPER FUNCTIONS    --------- #

getData = function(file_location) {
  ptm <- proc.time()
  
  print(sprintf("Loading: %s", file_location))
  
  dat <- read.csv(file_location)
  dat[,colSums(abs(dat)) != 0]  -> dat
  
  print(sprintf("Loaded. Took %s", proc.time() - ptm))
  
  return(dat)
}

getTest = function() {
  return(getData(TEST_SET)  %>% select(-id))
}

getTrain = function() {
  dat = getData(TRAINING_SET)  %>% select(-id)
  dat$categories = as.factor(dat$categories)
  # Numeric values are a problem unfortunately
  levels(dat$categories) <- c("a", "b", "c", "d", 
                              "e", "f", "g", "h",
                              "i", "j", "k", "l")
  return(dat)
}


getAllData = function() {
  return(rbind(getTrain() %>% select(-categories), getTest()))
}


loadModel = function(model_name) {
  return(readRDS(sprintf("%s/output/models/%s_fit.rds",HOMEDIR,model_name)))
}

writeToDisk = function(predictions, filename) {
  id <- seq(0, length(predictions)-1)
  x_name <- "id"
  y_name <- "categories"
  
  df <- data.frame(id, predictions)
  names(df) <- c(x_name,y_name)
  
  write.csv(df, file = filename, row.names=F)
}

savePredictions = function(predictions, model_name) {
  
  OUTPUT_PREDICTIONS <- sprintf("%s/output/predictions/%s_%s_pred.csv",HOMEDIR,as.numeric(Sys.time()),model_name)
  levels(predictions) <- c(0,1,2,3,4,5,6,7,8,9,10,11)
  writeToDisk(predictions, OUTPUT_PREDICTIONS)
}

saveModel = function(model, model_name) {
  OUTPUT_MODEL <- sprintf("%s/output/models/%s_fit.rds",HOMEDIR,model_name)
  saveRDS(model, OUTPUT_MODEL)
}
# ---------   PCA HELPER FUNCTIONS  --------- #


# ---------   LOAD AND CLEAN DATASET #OBSOLETE  --------- #

# Loading this thing takes up 1.5G
# dat = getData()

# ---------   HELPERS - PCA  --------- #

plotPCAPairs = function(pca){
  
  return(pairs(pca$x[,1:8], col=training_set_y$categories, upper.panel = NULL, pch = 16, cex = 0.5))
}

plotPCA3DScatter = function(training_pca){
  
  return(scatterplot3d(x = training_pca$PC1, 
                       y = training_pca$PC2, 
                       z = training_pca$PC3, 
                       xlab = "Principal Component 1", 
                       ylab = "Principal Component 2",
                       zlab = "Principal Component 3",
                       color = training_pca$categories, 
                       angle=20,
                       pch = 16,))
}

plotPCAVariance = function(pca) {
  # variance
  pr_var = ( pca$sdev )^2 
  
  # % of variance
  prop_varex = pr_var / sum( pr_var )
  
  # Plot
  plot( prop_varex, xlab = "Principal Component", 
        ylab = "Proportion of Variance Explained", type = "b" )
  
  
  plot( cumsum( prop_varex ), xlab = "Principal Component", 
        ylab = "Cumulative Proportion of Variance Explained", type = "b" )
  
}

make_pca = function() {
  ptm <- proc.time()
  pca = prcomp( getAllData(), scale=T, center=T) # We can use all the data for PCA Analysis
  print(sprintf("Took: %s", as.numeric(proc.time() - ptm)))
  saveRDS(pca, "pca.rds")
  
  gc()
  # Now let me project the training set onto PC space and save that
  training_pca = predict(pca, getTrain() %>% select(-categories)) %>% as.data.frame()
  training_pca$categories <- categories$categories
  categories <-  getTrain() %>% select(categories)
  gc()
  saveRDS(training_pca, TRAINING_SET_PCA)
  
  plotPCA(pca)
  plotPCAVariance(pca)
  
  test_pca = predict(pca, getTest()) %>% as.data.frame()
  saveRDS(test_pca, TEST_SET_PCA)
}

getPCADat = function(fileLocation,ncomp) {
  print(fileLocation)
  dat = readRDS(file=fileLocation)
  return(dat[,1:ncomp])
}

getTestPCA = Curry(getPCADat, fileLocation=TEST_SET_PCA)

getTrainingPCA = function(ncomp) {
  dat = getPCADat(TRAINING_SET_PCA, ncomp)
  categories <- readRDS(file=TRAINING_SET_PCA) %>% select(categories)
  dat$categories <- categories$categories
  head(dat$categories)
  return(dat)
}

#trainset = getTrainingPCA(10)
#head(trainset)

#test_pca = predict(pca, getTest()) %>% as.data.frame()
#test_pca = getTestPCA(10)

# ---------    HELPERS - MODELS  --------- #

plotAccuracy = function(model) {
  df <- data.frame( x =1:20
                    , Accuracy = model[["results"]][["Accuracy"]]*100
                    , L = model[["results"]][["Accuracy"]]*100 - 1.96*model[["results"]][["AccuracySD"]]*100
                    , U = model[["results"]][["Accuracy"]]*100 + 1.96*model[["results"]][["AccuracySD"]]*100
  )
  require(ggplot2)
  ggplot(df, aes(x = x, y = Accuracy)) +
    geom_point(size = 4) +
    geom_errorbar(aes(ymax = U, ymin = L))
}

makeModel = function(train_set, get_test_set, model_name, tune_length=10, repeats=5, number=2) {
  
  myfolds <- createMultiFolds(train_set$categories, k = 5, times = 2)

  
  control <- trainControl("repeatedcv"
                          , number =number
                          , repeats = repeats
                          #, number =5
                          #, repeats = 2
                          , index = myfolds
                          , selectionFunction = "oneSE"
                          , verboseIter=T)
  
  model <- 0
  
    if (tune_length == 0) {
      train(categories ~ ., data = train_set,
            method = model_name,
            metric = "Accuracy",
            trControl = control,
            preProc = c("zv","center","scale")) -> model
    } else {
      train(categories ~ ., data = train_set,
            method = model_name,
            metric = "Accuracy",
            trControl = control,
            tunelength=tune_length,
            verbose = T,
            preProc = c("zv","center","scale")) -> model  
  }
  
  print(summary(model))
  
  print("Saving model")
  print(OUTPUT_MODEL)
  saveModel(model, model_name)
  
  #plotAccuracy(model)
  
  # I moved the whole test data loading here so that it doesn't stick around in memory.
  
  print("Loading test data")
  test_set = get_test_set()
  print("Predicting")
  predict(model, test_set) -> predictions

  print("Saving predictions")
  savePredictions(predictions, model_name)
  
  return(model)
}

makePCAModel = function(ncomp){
  return(Curry(makeModel, train_set=getTrainingPCA(ncomp), function() {return(test_set=getTestPCA(ncomp))}))
}

makeSelfLearningModelLeastSquaresClassifier = function(ncomp,lambda=0) {
  print(sprintf(" ncomp: %s, lambda: %s", ncomp, lambda))
  
  X = getTrainingPCA(ncomp) %>% select(-categories)
  y = getTrainingPCA(10) %>% select(categories)
  y = y$categories %>% as.factor()
  X_u = getTestPCA(ncomp)
  
  model <- SelfLearning(X,y,X_u, method=LeastSquaresClassifier, lambda=lambda)
  model_name = sprintf("rssl_selflearn_LeastSquaresClassifier_lambda%s_%spc", lambda,ncomp)
  
  print("Saving model")
  saveModel(model, model_name)
  
  print("Predicting")
  predictions <- predict(model, X_u)
  
  print("Saving predictions")
  savePredictions(predictions, model_name)
  
  return(model)
}

plsdaSelfLearning = function(X,y,X_u){
  library(MASS)
  ModelVariables<-PreProcessing(X=X,y=y,X_u=X_u,scale=FALSE,intercept=FALSE,x_center=FALSE)
  X<-ModelVariables$X
  X_u<-ModelVariables$X_u
  y<-ModelVariables$y
  scaling<-ModelVariables$scaling
  modelform<-ModelVariables$modelform
  classnames<-ModelVariables$classnames
}
