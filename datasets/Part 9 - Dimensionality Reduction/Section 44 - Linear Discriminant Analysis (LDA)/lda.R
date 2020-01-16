# LDA

# =======================================================================================================
#
# Se basa de proyectar el espacio de caracteristicas a un espacio de dimension inferior, manteniendo la
# informacion discriminatoria lo maximo posible de cada una de las clases. Ademas de encontrar los
# ejes de las componentes de proyeccion, se tiene informacion sobre que ejes maximizan la separacion entre
# multiples clases. De las "n" variables independientes del dataset, se extraen las "p" <= n  nuevas 
# variables independientes que separen la mayoria de las clases de la variable dependiente.
#
# Como en el data set tiene que haber una columna con la clase a la que pertenecen las observaciones, es
# decir que se hace uso de la variable dependiente, este es un algoritmo supervisado.
#
# El objetivo es obtener aquellos componentes/ejes/discriminantes lineales que hagan que las clases queden lo mas separado posible
#
# PASOS:
# 1.- Aplicar escalado de variables a la matriz de caracteristicas, compuesta por "n" variables independientes
# 2.- Sea "C" el numero de clases; calcular "C" vectores m-dimensionales, de modo que cada uno contenga
#     las medias de las caracteristicas de las observaciones para cada clase. Obteniendo asi un vector con
#     las medias de todas las columnas de cada clase.
# 3.- Calcular la matrix de productos cruzados centrados en la media para cada clase, que mide la varianza
#     para cada clase.
# 4.- Se calcula la covarianza normalizada de todas las matrices anteriores, W
# 5.- Calcular la matriz de covarianza global entre clases, B
# 6.- Calculas los valores y vectores propios de la matriz. Es decir: W^-1*B
# 7.- Elegir los "p" valores propios mas grandes como el numero de dimensiones reducidas.
# 8.- Los "p" vectores propios asociados a los "p" valores propios mas grandes, son los discriminantes  
#     lineales. El espacio m-dimensional del dataset original, se proyecta al nuevo sub-espacio p-dimensional
#     de caracteristicas, aplicando la matriz de proyecciones (que tiene los p vectores propios por columnas).
#
# Siempre hay un discriminante lineal menos que el numero de clases.
#
# =======================================================================================================

################################################
###          IMPORTAR EL DATA SET            ###
################################################

setwd("~/Documentos/Udemy/machinelearning-az/datasets/Part 9 - Dimensionality Reduction/Section 44 - Linear Discriminant Analysis (LDA)")
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

training_set[,-14] = scale(training_set[,-14])
testing_set[,-14] = scale(testing_set[,-14])

################################################
#   Reducir la dimensión del dataset con LDA   #
################################################

library(MASS)
# Indicamos cual es la variable dependiente y del dataset de donde se obtienen las observaciones.
# Se puede especificar el numero de discriminantes lineales a tomar en cuenta, pero como en este caso tenemos 3
# clases y siempre el numero de discriminantes es igual al no. de clases-1, no hace falta.
lda = lda(formula = Customer_Segment ~ ., data = training_set)

# Aplicamos la transformacion generada en el paso anteroir al dataset de Entrenamiento y de Testing
# Una vez aplicada la transformacion, las primeras columnas son derivadas de las ecuaciones del modelo de LDA.
# Luego conservamos las columnas utiles y ordenamos el orden de las columnas.
# Los datasets de Training y de Testing, ahora tendran tantas columnas como componentes principales se hayan elegido, mas la columna de la variable dependiente.
training_set = as.data.frame(predict(lda, training_set)) # tenemos que convertirlo a un datadrame
training_set = training_set[, c(5, 6, 1)]
testing_set = as.data.frame(predict(lda, testing_set))
testing_set = testing_set[, c(5, 6, 1)]

#########################################################
#    Ajustar el modelo con el dataset de Entrenamiento  #
#########################################################

# En esta parte se agrega el modelo de clasificacion que se decida.

library(e1071)
classifier = svm(formula = class ~ ., 
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
colnames(grid_set) = c('x.LD1', 'x.LD2')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'SVM (Conjunto de Entrenamiento)',
     xlab = 'DL1', ylab = 'DL2',
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
colnames(grid_set) = c('x.LD1', 'x.LD2')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'SVM (Conjunto de Testing)',
     xlab = 'DL1', ylab = 'DL2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid==2, 'deepskyblue', 
                                         ifelse(y_grid == 1, 'springgreen3', 'tomato')))
points(set, pch = 21, bg = ifelse(set[, 3]==2, 'blue3', 
                                  ifelse(set[, 3] == 1, 'green4', 'red3')))
