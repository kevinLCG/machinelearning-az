# Random Forest

# =======================================================================================================
# IDEA
#
# Es una version mejorada de la Clasificacion por Arbol de Decision, pues ahora son muchos de estos. De esta
# manera se reduce el error en la prediccion. Al dato nuevo se le asigna la clase que hayan eligido la
# mayoria de los arboles.
#
# PASOS:
# 1.- Se selecciona un numero "k" de puntos aleatorios del Conjunto de entrenamiento.
# 2.- Construir un arbol de decision asociado a esos "k" puntos de datos.
# 3.- Elegir un numero "n" de arboles a construir y repetir los pasos 1 y 2.
# 4.- Para clasificar un nuevo punto, los "n" arboles realizan una prediccion sobre la categoria a la que
# pertenece este dato. Y asignar le al dato la categoria con mas votos.
#
# =======================================================================================================
#
# CUIDADO
# ¡ESTE ALGORITMO TIENDE A GENERAR OVERFITTING!
#
# =======================================================================================================


################################################
###          IMPORTAR EL DATA SET            ###
################################################

setwd("~/Documentos/Udemy/machinelearning-az/datasets/Part 3 - Classification/Section 20 - Random Forest Classification")
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[, 3:5]

# Codificar la variable de clasificación como factor
dataset$Purchased = factor(dataset$Purchased, levels = c(0,1))

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

# No es encesario escalar las variables, porque el algoritmos no esta basado en
# distancias Euclidianas. 
# Escalar las variables si se quiere que el grafico conserve la proporcion y quede mas fino.
# Pero si se quiere conservar a las variables con la misma escala, comentra las lineas.

training_set[,1:2] = scale(training_set[,1:2])
testing_set[,1:2] = scale(testing_set[,1:2])

#########################################################
##   Ajustar el modelo con el dataset de Entrenamiento  #
#########################################################

#install.packages("randomForest")
library(randomForest)
# El parametro "ntree" hace referencia al numero de arboles de clasificacion a utilizar.
classifier = randomForest(x = training_set[,-3],
                          y = training_set$Purchased,
                          ntree = 50)

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
     main = 'Random Forest (Conjunto de Entrenamiento)',
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
     main = 'Random Forest (Conjunto de Testing)',
     xlab = 'Edad', ylab = 'Sueldo Estimado',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

