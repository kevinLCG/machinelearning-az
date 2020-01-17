# SVR

# =======================================================================================================
#
# Las SVM (Support Vector Machines) sirven para resolver problemas de REGRESION y de CLASIFICACION, en
# este caso de trabaja con regresion, por lo que se les llama SVR (Support Vector Regresion).
#
# =======================================================================================================
#
# IDEA DE SVR
#
# Se hace uso de SVM, que sirven tanto para Regresiones Lineales y NO Lineales, el Kernel determina el
# tipo de regresion que se llevara a cabo.
# La idea es ajustar el pasillo (o calle) intentando mantener la mayor cantidad de observaciones
# posibles del conjunto de datos dentro del pasillo en torno a la recta. Ajustando y limitando la anchura
# del pasillo, conocido como "Margen Maximo"; Esta se controla mediante el hiperparameto epsilon, entre
# mayor sea el valor de este, mayor sera la anchura de pasillo.
# Cada uno de las resctas que representan los bordes/limites del pasillo, representan los potenciales
# vectores de soporte.
#
# El algoritmo hara una regresion lineal en un espacio vectorial de dimension superior a la dimension de
# los datos; cada punto del conjunto de entrenamiento representa su propia dimension. Por lo que al
# evaluar el kernel entre un punto de test y uno de entrenamiento, el resultado sera una coordenada
# trasladada a esa dimension superior.
#
# Al aplicarle la evaluacion de la SVR, a un punto de testing, este se trasladara al espacio de dimension
# superior, ese nuevo vector se representara como "k". Teniendo este vector en esta dimension, ahora si
# se lleva a cabo la regresion.
#
# El resultado es la transformacion de la recta en el espacio de dimension superior, al proyectarla en
# la dimension del conjunto de datos.
#
# # La SVM aproximara la funcion f, tal que aplicada al dominio "X", me de como resultados los puntos "y".
#
# =======================================================================================================
#
# PASOS
# 1.- Tener un conjunto de Entrenamiento. Se necesita que este cubra todo el dominio de interes y
#     vaya acompañado de las soluciones en dicho dominio. Este conjunto estara formado por la matriz
#     de caracteristixa "X" y la variable "y" a predecir
#
# 2.- Elegir un nucleo (una funcion) y sus parametros. Llevar a cabo cuanlquier regularizacion necesaria
#     (como eliminar el ruido del conjunto de entrenamiento).
#     Nucleos:
#            * Lineal (x,y)
#            * No Lineal (φ(x),φ(y)) = matriz K(x,y)
#               - Gaussiano
#     Regularizacion:
#            * Ruido
#
# 3.- Crear la matriz de Correlaciones "K".
#
# 4.- Se resuelve el sistema de ecuaciones Kα = y; Donde y = vector de valores del conjunto de entrenamiento,
#     K = matriz de correlacion, α = conjunto de incognitas para las que se resuelve el sist. de ec.
#     Esto de resuelve como: α = K(^-1)*y - PASO DE OPTIMIZACION
#     Se resuelve de forma EXACTA si se invierte la matriz o de forma APROXIMADA si se utiliza algun
#     metodo numerico, para obtener los coeficientes de contraccion "α".
#
# 5.- Utilizar los coeficientes anteriores y el kernel para crear un estimador "y*" que sea capaz de dar
#     la prediccion, tal que  y* = f(X,α,x*).
#     Calculando primero le vector de correlaciones "k^->" y luego se obtiene la prediccion "y*" a partir
#     de un conjunto de Testing "x*", donde: y* = α(^->)*k(^->)
#
# =======================================================================================================
#
# DIFERENCIAS CON LA REGRESION LINEAL SIMPLE
# La SVR tiene como objetivo que los errores no superen el umbral establecido, mientrs que en la regresion
# lineal simple se intenta minimizar el error entre la prediccion y los datos.
#
# =======================================================================================================
#
# RECURSOS
# http://alex.smola.org/papers/2004/SmoSch04.pdf
# https://www.mathworks.com/help/stats/understanding-support-vector-machine-regression.html
# https:/stats.stackexchange.com/questions/82044/how-does-support-vector-regression-work-intuitively
#
# =======================================================================================================


################################################
###          IMPORTAR EL DATA SET            ###
################################################

setwd("~/Documentos/Udemy/machinelearning-az/datasets/Part 2 - Regression/Section 7 - Support Vector Regression (SVR)")
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[, 2:3]

#################################################################################
### Dividir el data set en conjunto de entrenamiento y conjunto de testing    ###
#################################################################################

# En este caso no se hara debido a la escases de datos.

# install.packages("caTools")
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

########################################################
#  Ajustar el modelo con el dataset de Entrenamiento   #
########################################################

#install.packages("e1071")
library(e1071)
# Type indica que usaremos la SVM para hacer una regresion, por default esta clasificacion.
# Utilizamos un kernel radial/gaussiano, lineal, etc.; por defecto esta el radial.
regression = svm(formula = Salary ~ ., 
                 data = dataset, 
                 type = "eps-regression", 
                 kernel = "radial")
 
################################################
#                PREDICCION                    #
################################################

# Predicción de nuevos resultados con SVR 
y_pred = predict(regression, newdata = data.frame(Level = 6.5))

################################################
#        VISUALIZACION DE RESULTADOS           #
################################################

# Visualización del modelo de SVR
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Level , y = dataset$Salary),
             color = "red") +
  geom_line(aes(x = dataset$Level, y = predict(regression, 
                                        newdata = data.frame(Level = dataset$Level))),
            color = "blue") +
  ggtitle("Predicción (SVR)") +
  xlab("Nivel del empleado") +
  ylab("Sueldo (en $)")

# La SVR menosprecia aquellos valores atipicos o outlayers, por lo que no se ajusta
# muy bien con el sueldo del CEO