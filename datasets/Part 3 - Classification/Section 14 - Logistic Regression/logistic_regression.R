# Regresión Logística

# =======================================================================================================
# IDEA
# 
# Proviene una regresion lineal (y = b0 + b1*x), pero se aplica una funcion sigmoide (p = 1/(1 + e^-y))
# para transformar el valor "y". A la prediccion se le aplica una funcion sigmoide, para transformar el 
# valor final de la regresion en una probabilidad.
# 
# Si se toma un ejemplo de clasificacion binaria (Si/No); en la funion sigmoide, "p" representa la 
# probabilidad de un Si.
# 
# REGRESION LOGISTICA
# ln(p/1-p) = b0 + b1*x 	
# Lo que antes era una recta, se convierte ahora en una funcion Logistica, una funcion sigmoide.
# Es una funcion que se interpreta del mismo modo que la pendiente de una regresion lineal, solo que
# un poco curva y modela la tendencia a ambos grupos.
# 
# Se busca la mejor linea sigmoidea que mejor se ajuste a los datos.
# El eje y se transforma e una probabilidad.
# 
# 
# El resultado es la probabilidad de que un suceso ocurra. Se proyecta el valor de la variable del eje x
# en la curva logistica obtenida, la probabilidades se obtienen al proyectar estos valores, ahora sobre
# el eje y.
# 
# Lo que se hace despues es definir una probabilidad (se suele utilizar .50); todas aquellas probabilidades
# inferiores a esta, se proyectan a que es mas probable un "No". Mientras que todas los valores por encima
# de esta probabilidad, se proyectaran a un "Si".
# 
# =======================================================================================================
# 
# Dado que la regresion logistica es lineal, la frontera entre las clases sea una linea recta, sino seria
# un plano o un hiperplano.
# El clasificadoe se basa en una regresion lineal simple por lo que la recta obtenida sera la mejor dado
# umbral, pues se hace use de los minimos cuadrados.
#
# =======================================================================================================


################################################
###          IMPORTAR EL DATA SET            ###
################################################

setwd("~/Documentos/Udemy/machinelearning-az/datasets/Part 3 - Classification/Section 14 - Logistic Regression")
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

######################################################################
##   Ajustar el modelo de regresión logistica con todo el dataset    #
######################################################################

# Utilizamos la funcion generalizada de lm (generalized linear model).
# El parametro familiy hace referencia al tipo de regresion generalizada a llevar a cabo.
# El valor binomial sirve para crear una regresion logistina, por default es Gsussiana
classifier = glm(formula = Purchased ~ .,
                 data = training_set, 
                 family = binomial)

################################################
#                PREDICCION                    #
################################################

# Predicción de los resultados con el conjunto de testing
# Obtenemos la probabilidad de obtener un "Si"/1.
# El parametro "type", hace referencia al tipo de dato que deseamos de vuelta,
# "response" hace que las probabilidades se listen en un vector.
prob_pred = predict(classifier, type = "response",
                    newdata = testing_set[,-3])

# Transformamos las probabilides a los valores  0 o 1, seleccionando el umbral
# con el que se tomara esta decision.s
y_pred = ifelse(prob_pred> 0.5, 1, 0)

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

# El color de los puntos representan la clase verdadera, mientras que el color
# de fondo representa lo que el modedlo predijo (zonas de prediccion).

# Recordar que el resultado obtenido es la mejor linea recta que se separa a los puntos.
# Se puede variar el umbral de decision para comparar como cambiael rendimiento del modelo.

# Visualización del conjunto de entrenamiento
#install.packages("ElemStatLearn")
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)
plot(set[, -3],
     main = 'Clasificación (Conjunto de Entrenamiento)',
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
prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)
plot(set[, -3],
     main = 'Clasificación (Conjunto de Testing)',
     xlab = 'Edad', ylab = 'Sueldo Estimado',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))






