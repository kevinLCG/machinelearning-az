# Árbol de Decisión para Regresión

################################################
###          IMPORTAR EL DATA SET            ###
################################################

setwd("~/Documentos/Udemy/machinelearning-az/datasets/Part 2 - Regression/Section 8 - Decision Tree Regression")
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[, 2:3]

#################################################################################
### Dividir el data set en conjunto de entrenamiento y conjunto de testing    ###
#################################################################################

# En este caso no se hara debido a la escases de datos.

# library(caTools)
# set.seed(123)
# split = sample.split(dataset$Purchased, SplitRatio = 0.8)
# training_set = subset(dataset, split == TRUE)
# testing_set = subset(dataset, split == FALSE)


################################################
#            Escalado de variables             #
################################################

# Para los arboles de decision no suele hacerse escalado de variables, porque
# el algoritmo no utiliza distancias euclidianas.

# training_set[,2:3] = scale(training_set[,2:3])
# testing_set[,2:3] = scale(testing_set[,2:3])


################################################
#     Ajustar la regresión con el dataset      #
################################################

# install.packages("rpart")
library(rpart)
# Hacemos que el minimo numero de elementos de cada nodo hoja sea 1.
regression = rpart(formula = Salary ~ .,
                   data = dataset,
                   control = rpart.control(minsplit = 1))

################################################
#                PREDICCION                    #
################################################

y_pred = predict(regression, newdata = data.frame(Level = 6.5))

################################################
#        VISUALIZACION DE RESULTADOS           #
################################################

# Podria ser que obtengamos una linea horizontal (donde todos tuvieran el mismo sueldo),
# lo que podria estar pasando es que el arbol tuviera restricciones muy estrictas a la hora de dividir
# una rama en nodos hoja; como el tener un numero minimo de candidatos para cada nodo.
# Tambien el no. de divisiones, pues al arbol evita el overfitting, evitando nodos muy pequeños
# CHECAR PARAMETROS DE rpart()

library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level , y = dataset$Salary),
             color = "red") +
  geom_line(aes(x = dataset$Level, y = predict(regression, 
                                        newdata = data.frame(Level = dataset$Level))),
            color = "blue") +
  ggtitle("Predicción con Árbol de Decisión (Modelo de Regresión)") +
  xlab("Nivel del empleado") +
  ylab("Sueldo (en $)")

# En este plot podemos ver realmente lo que pasa en el algoritmo, al asignar la media como valor
# de la variable independiente. Ademas esto es lo que pasa cuando cada nodo contiene a 1 solo individuo.

x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Level , y = dataset$Salary),
             color = "red") +
  geom_line(aes(x = x_grid, y = predict(regression, 
                                        newdata = data.frame(Level = x_grid))),
            color = "blue") +
  ggtitle("Predicción con Árbol de Decisión (Modelo de Regresión)") +
  xlab("Nivel del empleado") +
  ylab("Sueldo (en $)")