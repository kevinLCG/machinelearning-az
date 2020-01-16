# PCA

# =======================================================================================================
#
# Resources:
# http://setosa.io/ev/principal-component-analysis/
#
# IDEA
# De las "m" variables independientes del dataset, se estrane las "p" <= m, nuevas variables independientes
# que explican la mayor parte de la varianza del dataset, sin importar el valor de la variable dependiente.
# Como no se hace uso de la variable dependiente, es un algoritmo no supervisado.
#
# Despues de aplicar el PCA, al graficar las observaciones, estas se encontrarian distribuidas en el
# espacio en 2 componentes, de modo que los 2 componentes principales, serian en donde se observa la mayor
# varianza entre los datos.
#
# PASOS:
# 1.- Aplicar el escalado de variables a la matriz de caracteristicas "X", formada por "m" variables
#     independientes.
# 2.- Calcular la matriz de covarianzas de las "m" variables independientes de "x".
# 3.- Calcular los valores y vectores propios de la matriz de covarianzas.
# 4.- Elegir un porcentaje "P" de varianza explicada y elegir los p <= m valores porpios mas grandes,
#     tales que:  (Σ^p λj) / (Σ^m λi) > P
# 5.- Los "p" vectores propios asociados a estos "p" valores propios mas grandes, son los componentes principales.
#     El espacio m-dimensional del dataset original se proyectara al nuevo espacio p-dimensional de
#     caracteristicas, aplicando la matriz de proyecciones (que tiene los "p" vectores propios por columnas).
#
# Dataset:
# https://archive.ics.uci.edu/ml/datasets/Wine
#
# =======================================================================================================


################################################
###          IMPORTAR EL DATA SET            ###
################################################

setwd("~/Documentos/Udemy/machinelearning-az/datasets/Part 9 - Dimensionality Reduction/Section 43 - Principal Component Analysis (PCA)")
dataset = read.csv('Wine.csv')

#################################################################################
### Dividir el data set en conjunto de entrenamiento y conjunto de testing    ###
#################################################################################

# install.packages("caTools")
library(caTools)
set.seed(123)
split = sample.split(dataset$Customer_Segment, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)

################################################
#            Escalado de variables             #
################################################

# este paso es importante para centrar las variables en cero y con desviacion estandar 1.
training_set[,-14] = scale(training_set[,-14])
testing_set[,-14] = scale(testing_set[,-14])

################################################
#   Reducir la dimensión del dataset con PCA   #
################################################

# Proyección de las componentes principales
# install.packages("caret")
library(caret)
# install.packages("e1071")
library(e1071)
# Desde la funcion de "preProcess", se puede hacer el escalado de variables
# y tambien se puede elejir la proporcion minima de la varianza a explicar,
# en vez de la cantidad de Componentes Principales.
# Se pueden elegir otros metodos de reduccion de dimension.
pca = preProcess(x = training_set[, -14], method = "pca", pcaComp = 2)

# Aplicamos la transformacion generada en el paso anteroir al dataset de Entrenamiento y de Testing
# Y cambiamos el orden de las columnas.
# Los datasets de Training y de Testing, ahora tendran tantas columnas como componentes principales se hayan elegido, mas la columna de la variable dependiente.
training_set = predict(pca, training_set)
training_set = training_set[, c(2, 3, 1)]
testing_set = predict(pca, testing_set)
testing_set = testing_set[, c(2, 3, 1)]

#########################################################
#    Ajustar el modelo con el dataset de Entrenamiento  #
#########################################################

# En esta parte se agrega el modelo de clasificacion que se decida.

classifier = svm(formula = Customer_Segment ~ ., 
                 data = training_set,
                 type = "C-classification",
                 kernel = "linear")

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

# INTERPRETACION: Las filas representan el dato real, mientras que las columnas la prediccion.
# La dimension de esta matriz dependera de la cantidad de clases que haya, es decir que si hay 4 clases
# la matriz dera de 4*4, por ejemplo.
# La diagonal de izq. a der. representa aquellos casos en los que lagoritmo acerto, mientras
# que las demas casillas representan aquellos caso en los que el algoritmo fallo.

################################################
#        VISUALIZACION DE RESULTADOS           #
################################################

# Visualización del conjunto de entrenamiento
#install.packages("ElemStatLearn")
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.025)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.025)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('PC1', 'PC2')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'SVM (Conjunto de Entrenamiento)',
     xlab = 'CP1', ylab = 'CP2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid==2, 'deepskyblue', 
                                         ifelse(y_grid == 1, 'springgreen3', 'tomato')))
points(set, pch = 21, bg = ifelse(set[, 3]==2, 'blue3', 
                                  ifelse(set[, 3] == 1, 'green4', 'red3')))


# Visualización del conjunto de testing
set = testing_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.02)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.02)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('PC1', 'PC2')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'SVM (Conjunto de Testing)',
     xlab = 'CP1', ylab = 'CP2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid==2, 'deepskyblue', 
                                         ifelse(y_grid == 1, 'springgreen3', 'tomato')))
points(set, pch = 21, bg = ifelse(set[, 3]==2, 'blue3', 
                                  ifelse(set[, 3] == 1, 'green4', 'red3')))
