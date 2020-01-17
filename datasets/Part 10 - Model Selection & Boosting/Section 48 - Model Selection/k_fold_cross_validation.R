# k-fold cross validation

# =======================================================================================================
#
# Normalmente se divide el dataset en un conjunto de Entrenamiento y en otro de Testing y esta no es la
# mejor manera, pues se pueden tener problemas de varianza en los datos. Es decir que si se vuelve a 
# ejecutar el modelo con conjunto de diferentes, la precision en la prediccion puede variar. Por lo que
# juzgar el rendimiento de un modelo con 1 solo conjunto de pruea, NO ES EL MEJOR ENFOQUE POSIBLE.
#
# En el "k-fold Cross Validation" se reemuestrea varias veces el dataset de Entrenamiento.
# El dataset se divide en "k" partes, y en cada iteracion se utilizaran "k-1" partes para entrenar y
# 1 parte para evaluar, de modo que en cada iteracion se use una parte diferente para realizar la evaluacion.
#   Todos los datos habran sido utilizados para entrenar y evaluar, evitando los sesgos generados cuando
# un conjunto de datos de evaluacion tiene mucha varianza, pues en las demas ocasiones no se utilizara
# para evaluar.
#
# El error final sera la desviacion estandar observada en cada una de las iteraciones, ponderada por 
# una k-esimauna parte de cada una de ellas.
#
# Sesgo Bajo: Cuando el modelo elabora predicciones cercanas a los datos reales.
# Sesgo Alto: Cuando el modelo elabora predicciones alejadas de los datos reales.
# Varianza Baja: Cuando ejecutamos el modelo varias veces y las predicciones no varian demasiado.
# Varianza Alta: Cuando ejecutamos el modelo varias veces y las predicciones varian demasiado.
#
# Habra que ver en que caso se encuentra:
# Sesgo Bajo Varianza Baja
# Sesgo Bajo Varianza Alta
# Sesgo Alto Varianza Baja
# Sesgo Alto Varianza Alta
#
# =======================================================================================================

################################################
###          IMPORTAR EL DATA SET            ###
################################################

setwd("~/Documentos/Udemy/machinelearning-az/datasets/Part 10 - Model Selection & Boosting/Section 48 - Model Selection")
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[, 3:5]

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
.
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

# Crear la matriz de confusión
cm = table(testing_set[, 3], y_pred)

# INTERPRETACION: Las columnas representan el dato real, mientras que las fila la prediccion.
# La diagonal de izq. a der. representa aquellos casos en los que lagoritmo acerto, mientras
# que la diagonal que va de der. a izq. representa aquellos caso en los que el algoritmo fallo.
# VN  FN
# FP  VP

## Aplicar algoritmo de k-fold cross validation

# install.packages("caret")
library(caret)
# Primero se crean las "k" separaciones del dataset
# Se selecciona la variable dependiente del Dataset de Entrenamiento y el numero "k" en el que se dividira este mismo.
folds = createFolds(training_set$Purchased, k = 10)
# Se aplica la CV con ayuda de la funcion lapply. De tal manera que se entrena el modelo, se hace la prediccion y se calcula 
# la matriz de confusion y se obtiene el accuracy "k" veces.
# El resultado es una lista (CV) con todas las accuracy.
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
# Se saca la media y la desviacion estandard del accuracy
accuracy = mean(as.numeric(cv))
accuracy_sd = sd(as.numeric(cv))

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
