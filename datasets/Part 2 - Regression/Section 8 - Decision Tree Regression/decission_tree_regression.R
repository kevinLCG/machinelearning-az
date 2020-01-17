# Árbol de Decisión para Regresión

# CART (Classification and Regression Trees)
# =======================================================================================================
# 
#  Si graficamos en 2D datos de 2 variables independientes (X1,X2) que permitiran predecir  una variable
# dependiente, el algoritmo dividira los conjuntos de puntos con lineas rectas. El algoritmo mira la entropia,
# que tan juntos o dispersos pueden estar esos puntos o que similitudes tienen entre si. 
# Cada uno de estos conjuntos corresponde a un nodo hoja. Se continua haciendo divisiones hasta cierto
# punto, p.e. Cuando un nodo hoja se quede por lo menos al 5% de datos original.
# 
# La idea del algoritmo es que la cantidad de informacion aumenta (es mas acertada) cuando dividimos los 
# puntos en conjuntos o alguna otra regla, para hacer que el algoritmo converja y no haya Overffiting.
# 
# Al mismo tiempo que se divide el conjunto, se "genera" un arbol. si la primera division fuera que X1 < 20
# se generarian entonces 2 ramas. Luego si la sig. div. es que X2 < 170, pero solo en quellos con X1 > 20, 
# se generan 2 nuevas ramas dentro de esa rama, y asi sucesivamente.
# 
# La 3ra Dimension, dada por la variable dependiente, es la que nos sirve para la prediccion. Se saca la
# media  de la variable dependiente de cada conjunto. Una vez que ingrese un nuevo dato, se enontrara
# el nodo hoja al que pertenece haciendo uso de la informacion de las variables independientes, y se le
# asignara como variable dependiente, la media de ese conjunto para esa variable.
# 
# =======================================================================================================

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


########################################################
#  Ajustar el modelo con el dataset de Entrenamiento   #
########################################################

# install.packages("rpart")
# Crgamos la libreria "rpart" (Recursive Partitioning and Regression Tree)
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