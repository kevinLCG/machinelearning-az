# Naïve Bayes 

# =======================================================================================================
# IDEA
# 
# Se tienen 2 clases "A" y "B" y entra un nuevo dato. Se busca la pobabilidad de que provenga de "A" o de 
# "B". La probabilidad se obtiene a travez de informacion obtenida del propio contexto.
# 
# Teorema de Bayes
# P(clase|X) = ( P(X|clase)*P(clase) ) / ( P(X) )
# Donde: "X" son las caracteristicas variables del nuevo dato.
#        P(clase) - Es la Pobabilidad a Priori
#        P(X) - Es la Probabilidad Marginal con respecto a las caracteristicas
#        P(X|clase) - Es la Pobabilidad Condicionada
#        P(clase|X) - Es la Pobabilidad Posterior
#
# Para obtener la probabilidad marginal se genera un circulo con radio arbitrario para obtener
# observaciones similares y se cuenta cuantas observaciones caen dentro de ese circulo. La probabilidad
# marginal seria l numero de observaciones similares entre el total de observaciones.
# 
# Para obtener la probabilidad condicionada "P(X|clase)", se obtiene considerando unicamente a los individuos
# de la clase en cuestion. De estos individuos se cuenta cuantos de estos caen dentro del circulo de
# observaciones similares al nuevo dato. Este nuemro se divide entre el numero de observaciones totales
# de esa clsae. 
# 
# En el caso de "Bayesianos Ingenous", lo que se hace es aplicar el Teorema de Bayes tantas veces como
# clases haya y obtener las probabilidades de que el nuevo dato pertenezca a cada una de las clases.
# Se comparan las probabilidades y al nuevo dato se le asigna la clase que sea mas probable.
# 
# =======================================================================================================
# 
# PORQUE BAYESIANOS "INGENUOS"
# Porque supone una independencia entre los datos que aparecen dentro de las probabilidades, es decir que 
# las variables sean independientes, que muchas veces no es cierta y por eso se hace una suposicion 
# "ingenua".
# 
# P(X)
# Numero de observaciones similares, entre observaciones totales. Como este valor siempre es el mismo para
# el calculo de la Probabilidad Posterior de todas las clases, se puede omitir este valor y no afectara el
# resultado de la comparacion. NOTA: El resultado obtenido ya no sera la probabilidad de que el dato
# pertenezca a una clase determinada.
#
# MULTIPLES CLASES
# Haecr el calculo de todas las probabilidades y compararlas de forma habitual.
# 
# =======================================================================================================


################################################
###          IMPORTAR EL DATA SET            ###
################################################

setwd("~/Documentos/Udemy/machinelearning-az/datasets/Part 3 - Classification/Section 18 - Naive Bayes")
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[, 3:5]

# Codificar la variable de clasificación como factor
# Esto porque la funcion "naiveBayes", necesita que el vector de la variable
# dependiente sea un factor.
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

training_set[,1:2] = scale(training_set[,1:2])
testing_set[,1:2] = scale(testing_set[,1:2])

#########################################################
##   Ajustar el modelo con el dataset de Entrenamiento  #
#########################################################

#install.packages("e1071")
library(e1071)
classifier = naiveBayes(x = training_set[,-3], 
                        y = training_set$Purchased)

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
     main = 'Naïve Bayes (Conjunto de Entrenamiento)',
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
     main = 'Naïve Bayes (Conjunto de Testing)',
     xlab = 'Edad', ylab = 'Sueldo Estimado',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))
