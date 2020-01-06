# SVM

# =======================================================================================================
# IDEA
# 
# Se busca busca obtener la mejor linea de decision que ayude a separar el espacio en las 2 clases, la
# linea que se busca ása por lo que se conoce como "margen maximo". Se crea un pasillo y la SVM detecta
# los 2 puntos mas cercanos a esta linea y a eso le llama "margen maximo", por lo que esta linea es
# equidistante a los puntos que le quedan mas cerca. 
# Los datos deben ser linealmente separables, para el caso en que el kernel sea lineal.
#
# OBJETIVO
# Se busca que la distancia entre estos puntos equidistantes sea maxima para crear el pasillo mas grande
# posible. Y a estos 2 puntos se le llaman "vectores de soporte".
# 
# Generalizando...
# Si hubiera mas variables y por lo tanto mas dimensiones al margen maximo se le llama "hiperplano de 
# marge maximo" con dimension 1 menos que el espacio donde se este trabajando.
#
# De los 2 hiperplanos paralelos, uno es el hiperplano positivo"" y el otro el "hiperplano negativo"
# (asignados de manera arbitraria).
# 
## CLASIFICACION
# Ajustan el major pasillo posible entre 2 clases, ajustando la anchura de este.
# En el caso de clasificacion los vectores "X" se utilizan para definir un hiperplano que separe las 2
# categorias de la solucion. Estos vectores se utilizan para llevar a cabo la regresion lineal.
# Los vectores que quedan mas cercanos al conjunto de Testing, son los llamados "VECTORES DE SOPORTE".
# CUIDADO: Podemos evaluar nuestra funcion en cualquier lugar, por lo que cualquier vector podria estar
# mas cerca en el conjunto de evaluacion.
#
# =======================================================================================================
#
# Son utiles cuando los datos de una categoria son parecidos a la de la otra, pues, se toma como borde
# a aquellos datos extremos que se parecen a la otra categoria.
#
# =======================================================================================================

################################################
###          IMPORTAR EL DATA SET            ###
################################################

setwd("~/Documentos/Udemy/machinelearning-az/datasets/Part 3 - Classification/Section 16 - Support Vector Machine (SVM)")
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

######################################################
##   Ajustar el modelo de SVM con todo el dataset    #
######################################################

#install.packages("e1071")
library(e1071)
# El parametro "type" hace referencia al tipo de SVM a utilizar, y el valor "C-classification"
# indica que es para un problema de clasificacion.
# Se eligira un kernel lineal, aunque tambien se puede el usar uno gaussiano, polinomial, etc.
classifier = svm(formula = Purchased ~ ., 
                 data = training_set,
                 type = "C-classification",
                 kernel = "linear")

################################################
#                PREDICCION                    #
################################################

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
     main = 'SVM (Conjunto de Entrenamiento)',
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
     main = 'SVM (Conjunto de Testing)',
     xlab = 'Edad', ylab = 'Sueldo Estimado',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

