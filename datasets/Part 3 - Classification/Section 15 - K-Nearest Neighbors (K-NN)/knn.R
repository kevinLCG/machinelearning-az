# K - Nearest Neighbors (K-NN)

# =======================================================================================================
# IDEA
#
# El algoritmo parte de tener una serie de datos categorizados. Al aparecer un nuevo dato, el algoritmo
# a que conjunto pertenece.
#
# PASOS:
# 1.- Elegir el numero "k" de vecinos del nuevo dato se tomaran en cuenta para la clasificacion. Se recomienda
#     un numero impar, para evitar empates. K = 5 es muy usado.
# 2.- Se toman los "k" veninos mas cercanos del nuevo dato; utilizando distancias euclidianas la mayoria
#     de las ocasiones, aunque tambien se puede utilizar la distancia Manhattan del valor absoluto, la
#     distancia Minkowski, metrica del infinito, etc.
# 3.- Ver a que categoria pertenecen los "k" vecinos y contar cuantos pertenecen a cada categpria
# 4.- Asignar el nuevo dato a la categoria con mas vecinos.
#
# =======================================================================================================
# 
# La frontera entre las clases practicamente nunca es lineal.
#
# =======================================================================================================

################################################
###          IMPORTAR EL DATA SET            ###
################################################

setwd("~/Documentos/Udemy/machinelearning-az/datasets/Part 3 - Classification/Section 15 - K-Nearest Neighbors (K-NN)")
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

#############################################################################################################
##   Ajustar el modelo de KNN con todo el dataset de Entrenamiento y hacer la REDICCION con el de Testing   #
#############################################################################################################

# Cargamos la libreria "class" que contiene un conjunto de funciones para procesos de clasificacion.
library(class)
# La funcion devuelve la prediccion, recibe como parametros a lo conjuntod de Entrenamiento
# y Testing. ¡OJO! "train" hace referencia a la matriz de caracteristicas de Entrenamiento (variables independientes)
# mientras que "cl", hace referencia a la clasificaciones verdaderas del conjunto de Entrenamiento.

# El parametro "k", hace referencia al numero de "k" vecinos para clasificar
# a los nuevos datos.
y_pred = knn(train = training_set[,-3],
             test = testing_set[,-3],
             cl = training_set[,3],
             k = 5)

# La prediccion es un vector con con cada una de las clases elegidas (0 o 1), ya no es la probabilidad.

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
y_grid = knn(train = training_set[,-3],
             test = grid_set,
             cl = training_set[,3],
             k = 5)
plot(set[, -3],
     main = 'K-NN (Conjunto de Entrenamiento)',
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
y_grid = knn(train = training_set[,-3],
             test = grid_set,
             cl = training_set[,3],
             k = 5)
plot(set[, -3],
     main = 'K-NN (Conjunto de Testing)',
     xlab = 'Edad', ylab = 'Sueldo Estimado',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

