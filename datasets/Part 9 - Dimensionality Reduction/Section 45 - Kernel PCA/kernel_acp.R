# Kernel PCA

# =======================================================================================================
#
# Permite reducir la dimension cuando se trata con un problema de Clasificacion que NO es linealmente
# separable. Es una version de PCA adaptada con un kernel que transforma los datos, primero a una dimension
# superior donde si sean linealmente separables y a partir de ahi exptraer componentes principales para
# solucionar el problema como si fuera linelamente separable.
#
# Como primero se aplico un kernel y luego se hizo un PCA a un conjunto de datos que no era linealmente
# searable, ahora las observaciones se localizan en zonas del plano diferentes a las que se tenian en un
# principio.
# Los datos se separan de una mejor manera por una recta que sin aplicar el kernel.
#
# =======================================================================================================


################################################
###          IMPORTAR EL DATA SET            ###
################################################

setwd("~/Documentos/Udemy/machinelearning-az/datasets/Part 9 - Dimensionality Reduction/Section 45 - Kernel PCA")
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

#######################################################
#   Reducir la dimensión del dataset con Kernel PCA   #
#######################################################

#install.packages("kernlab")
library(kernlab)
# Se crea el kernel que eumentara la dimension de los datos para hacerlos linealmente separables.
# La formula se puede escribir como "~." para indicar que todas las variables son independientes,
# por lo que del trining set se quita la columna de la variable dependiente.
# Se puede aplicar un kernel gaussiano (rbf), polinoial, sigmoide, etc. dependiendo de la naturaleza de los datos.
# con "features", seleccionamos el no. de componentes principales a tomar en cuenta.
kpca = kpca(~., data = training_set[, -3], 
            kernel = "rbfdot", features = 2)
# Aplicamos la transformacion generada en el paso anteroir al dataset de Entrenamiento y de Testing
# Y cambiamos el orden de las columnas.
# Los datasets de Training y de Testing, ahora tendran tantas columnas como componentes principales se hayan elegido.
training_set_pca = as.data.frame(predict(kpca, training_set)) # se necesita hacer un casting a DataFrame
# Se le añade al training set la columna de la variable dependiente
training_set_pca$Purchased = training_set$Purchased
testing_set_pca = as.data.frame(predict(kpca, testing_set)) # se necesita hacer un casting a DataFrame
# Se le añade al testing set la columna de la variable dependiente
testing_set_pca$Purchased = testing_set$Purchased

# training_set - Tiene al dataset con las variables escaladas.
# training_set_pca - Tiene al dataset con las variables escaladas y con la transformacion.

#########################################################
#    Ajustar el modelo con el dataset de Entrenamiento  #
#########################################################

# En esta parte se agrega el modelo de clasificacion que se decida.

classifier = glm(formula = Purchased ~ .,
                 data = training_set_pca, 
                 family = binomial)

################################################
#                PREDICCION                    #
################################################

# Predicción de los resultados con el conjunto de testing
prob_pred = predict(classifier, type = "response",
                    newdata = testing_set_pca[,-3])

y_pred = ifelse(prob_pred> 0.5, 1, 0)

################################################
#        EVALUACION DEL RENDIMIENTO            #
################################################

# Crear la matriz de confusión
cm = table(testing_set_pca[, 3], y_pred)

################################################
#        VISUALIZACION DE RESULTADOS           #
################################################

# Visualización del conjunto de entrenamiento
#install.packages("ElemStatLearn")
library(ElemStatLearn)
set = training_set_pca
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('V1', 'V2')
prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)
plot(set[, -3],
     main = 'Clasificación (Conjunto de Entrenamiento)',
     xlab = 'CP1', ylab = 'CP2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))


# Visualización del conjunto de testing
set = testing_set_pca
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('V1', 'V2')
prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)
plot(set[, -3],
     main = 'Clasificación (Conjunto de Testing)',
     xlab = 'CP1', ylab = 'CP2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

