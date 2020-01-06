# Random Forest Regression

# =======================================================================================================
#
# Es una version mejorada de la regresion por Arbol de Decision, pues ahora son muchos de estos. De esta
# manera se reduce el error y evita que haya cambios bruscos en la prediccion.
#
# PASOS:
# 1.- Elegir un numero aleatorio "k" de puntos de datos del Conjunto de entrenamiento 
# 2.- Se construye un arbol aleatorio de regresion asociado a esos "k" puntos.
# 3.- Elegir el numero de arboles a construir y repetir los pasos 1 y 2.
#     NOTA: Cada arbol tendra una vision parcial del conjunto global de entrenamiento.
# 4.- Para un nuevo nuevo dato. Cada uno de los arboles  hara una prediccion del valor "y" para el punto en
#     cuestion. La prediccion final sera un promedio de todas las predicciones de los arboles.
# 
# En ciertas ocasiones en lugar de la media sse utiliza la mediana, para evitar la distorcion por outliers
# o la media recortada, donde se quita el 5% de valores mas grandes y el 5% de valores mas chicos.
# 
# =======================================================================================================


################################################
###          IMPORTAR EL DATA SET            ###
################################################

dataset = read.csv('Position_Salaries.csv')
dataset = dataset[, 2:3]

#################################################################################
### Dividir el data set en conjunto de entrenamiento y conjunto de testing    ###
#################################################################################

# En este caso no se hara por la escases de datos

# library(caTools)
# set.seed(123)
# split = sample.split(dataset$Purchased, SplitRatio = 0.8)
# training_set = subset(dataset, split == TRUE)
# testing_set = subset(dataset, split == FALSE)


################################################
#            Escalado de variables             #
################################################

# training_set[,2:3] = scale(training_set[,2:3])
# testing_set[,2:3] = scale(testing_set[,2:3])

############################################################
##  Ajustar la regresión polinómica con todo el dataset    #
############################################################

# install.packages("randomForest")
library(randomForest)
set.seed(1234)
# En el valor del parametro "x" de toman los valores de la columna 1 por su indice
#  y no como dataset$<columna>. Esto porquela funcion recibe un dataframe y al acceder
# por indice, se genera un sub dataframe, mientras que si se toma la columna con "$",
# se genera un vector.
# Por otro lado, el parametro "y" si recibe un vector.
# El parametro "ntree", hace referencia al numero de arboles de regresion a utilizar.
regression = randomForest(x = dataset[1],
                          y = dataset$Salary,
                          ntree = 500)

# Recomendacion: Variar el numero de arboles para ver como cambian los resultados. Las grafica no se
# vuelve mas suave al aumentar el numero de arboles, pero el resultado vaya que si aumenta.

################################################
#                PREDICCION                    #
################################################

y_pred = predict(regression, newdata = data.frame(Level = 6.5))

################################################
#        VISUALIZACION DE RESULTADOS           #
################################################

# Visualización del modelo de Random Forest
# install.packages("ggplot2")
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)
ggplot() +
  geom_point(aes(x = dataset$Level , y = dataset$Salary),
             color = "red") +
  geom_line(aes(x = x_grid, y = predict(regression, 
                                        newdata = data.frame(Level = x_grid))),
            color = "blue") +
  ggtitle("Predicción (Random Forest)") +
  xlab("Nivel del empleado") +
  ylab("Sueldo (en $)")
