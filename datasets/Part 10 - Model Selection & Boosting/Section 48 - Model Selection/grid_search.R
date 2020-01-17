# Grid Search

# =======================================================================================================
#
# Partiendo de que cualquier algoritmo de Machine Learning tiene 2 tipos de parametros:
# El primer tipo de parametros,  se aprenden a travez del propio algoritmo (por ejemplo, los coeficientes
# en una regresion lineal o los pesos en una Red Neronal); y el otro tipo de parametros (hiperparametros)
# los elige la persona que esta detras del algoritmo (por ejemplo, la eleccion del kernel en una SVM,
# parametros de penalizacion o tasas de aprendizaje en REdes Neuronales)
#
# "Grid Search" es una tecnica que intenta mejorar el rendimiento de los modelos, al encontrar los valores optimos para los
# hiperparametros del modelo. Ademas nos dira si es mejor escoger un modelo lineal o NO lineal,
# como en el caso de las SVM, donde se puede escoger un kernel lineal o no.
#
# =======================================================================================================

################################################
###          IMPORTAR EL DATA SET            ###
################################################

setwd("~/Documentos/Udemy/machinelearning-az/datasets/Part 10 - Model Selection & Boosting/Section 48 - Model Selection")
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[, 3:5]

dataset$Purchased = factor(dataset$Purchased)

#################################################################################
### Dividir el data set en conjunto de entrenamiento y conjunto de testing    ###
#################################################################################

# install.packages("caTools")
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)

################################################
#            Escalado de variables             #
################################################

training_set[,1:2] = scale(training_set[,1:2])
testing_set[,1:2] = scale(testing_set[,1:2])

#########################################################################
#  Ajustar el modelo de Clasificacion con el dataset de Entrenamiento   #
#########################################################################

#install.packages("e1071")
library(e1071)
classifier = svm(formula = Purchased ~ .,
                 data = training_set, 
                 type = "C-classification",
                 kernel = "radial")

################################################
#                PREDICCION                    #
################################################

# Predicción de los resultados con el conjunto de testing
y_pred = predict(classifier, newdata = testing_set[,-3])

################################################
#        EVALUACION DEL RENDIMIENTO            #
################################################

## Crear la matriz de confusión
cm = table(testing_set[, 3], y_pred)

# INTERPRETACION: Las columnas representan el dato real, mientras que las fila la prediccion.
# La diagonal de izq. a der. representa aquellos casos en los que lagoritmo acerto, mientras
# que la diagonal que va de der. a izq. representa aquellos caso en los que el algoritmo fallo.
# VN  FN
# FP  VP

## Aplicar algoritmo de k-fold cross validation 

# install.packages("caret")
library(caret)
folds = createFolds(training_set$Purchased, k = 10)
cv = lapply(folds, function(x) { 
  training_fold = training_set[-x, ]
  test_fold = training_set[x, ]
  classifier = svm(formula = Purchased ~ .,
                   data = training_fold, 
                   type = "C-classification",
                   kernel = "radial")
  y_pred = predict(classifier, newdata = test_fold[,-3])
  cm = table(test_fold[, 3], y_pred)
  accuracy = (cm[1,1]+cm[2,2])/(cm[1,1]+cm[1,2]+cm[2,1]+cm[2,2])
  return(accuracy)  
})
accuracy = mean(as.numeric(cv))
accuracy_sd = sd(as.numeric(cv))

#############################################################################
# Aplicar la mejora de Grid Search para otimizar el modelo y sus parámetros #
#############################################################################

#install.packages("caret")
library(caret)
classifier = train(form = Purchased ~ .,
                   data = training_set, method = 'svmRadial')
classifier
classifier$bestTune

################################################
#        VISUALIZACION DE RESULTADOS           #
################################################

# Visualización del conjunto de entrenamiento
#install.packages("ElemStatLearn")
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'SVM Kernel (Conjunto de Entrenamiento)',
     xlab = 'Edad', ylab = 'Sueldo Estimado',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))


# Visualización del conjunto de testing
set = testing_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'SVM Kernel (Conjunto de Testing)',
     xlab = 'Edad', ylab = 'Sueldo Estimado',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))
