'''
-*- coding: utf-8 -*-
 Autor: Kevin Meza
 Fecha: 10.12.2019
 OBJETIVO: Procesado de Datos: Machine Learning
 '''

#=========================================================================================
#                           MAIN PROGRAMM
#=========================================================================================

###########################################################
#                       Input Dataset                     #
###########################################################

setwd("/home/kevinml/Documentos/Udemy/machinelearning-az/datasets/Part 1 - Data Preprocessing/Section 2 -------------------- Part 1 - Data Preprocessing --------------------/my_version_kevinml")
dataset = read.csv('Data.csv') 


###########################################################
#                  Tratamiento de los NAs                 #
###########################################################

# Sustituimos cada un o de los valores del Dataframe con el valor promedio de la columna
# Lo hacemos tanto para la columna de edades como para la de salarios
dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Age)
dataset$Salary = ifelse(is.na(dataset$Salary),
                        ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                        dataset$Salary)

###########################################################
#            Codificacion de Datos Categoricos            #
###########################################################

# Hacemos que la columna de Country y de Purchased sea un factor, indicamos los diferentes niveles del factor
# y con labels elegimos el valor con el que queremos sustituirlos
dataset$Country = factor(dataset$Country,
                         levels = c("France", "Spain", "Germany"),
                         labels = c(1, 2, 3))
dataset$Purchased = factor(dataset$Purchased,
                           levels = c("No", "Yes"),
                           labels = c(0,1))

###########################################################
#             Training & Testing Splitting                #
###########################################################

# Cargamos una libreria y definimos semilla
library(caTools)
set.seed(123)

# Con esta funcion seleccionamos que datos usaremos para entrenar, indicando que queremos el 80% de estos
split = sample.split(dataset$Purchased, SplitRatio = 0.8)

# Obtenemos del Dataset origunal, dos subsets cn los datos de entrenamiento y de testing
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)


###########################################################
#                 Escalado de Variables                   #
###########################################################

# Esto se hace devido a que el rango dinamico de cada una de las variables diferentes
# y al momento de operar con ellas, como al momento de sacar distancias euclidianas
# el valor de las variables de mayor rango, puede opacar el de aquellas cuyo rango sea menor.
# Obtenderemos variables entre -1 y 1.

# Escalamos unicamente las columnas 2 (edad) y 3 (sueldo).
training_set[,2:3] = scale(training_set[,2:3])
testing_set[,2:3] = scale(testing_set[,2:3])