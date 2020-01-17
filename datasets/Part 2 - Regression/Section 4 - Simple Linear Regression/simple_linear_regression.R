# Autor: juangabriel & Kevin Meza
# Version: 13-12-2019

# Regresión Lineal Simple

################################################
###          IMPORTAR EL DATA SET            ###
################################################

setwd("/home/kevinml/Documentos/Udemy/machinelearning-az/datasets/Part 2 - Regression/Section 4 - Simple Linear Regression/")
dataset = read.csv('Salary_Data.csv')
#dataset = dataset[, 2:3]

#################################################################################
### Dividir el data set en conjunto de entrenamiento y conjunto de testing    ###
#################################################################################

library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)

################################################
#            Escalado de variables             #
################################################

# training_set[,2:3] = scale(training_set[,2:3])
# testing_set[,2:3] = scale(testing_set[,2:3])

#################################################################
#  Ajustar la regresión lineal con el dataset de Entrenamiento   #
#################################################################

# La primera variable es la dependiente y la segunda la independiene.
regressor = lm(formula = Salary ~ YearsExperience,
               data = training_set)
# Si queremos obtener informacion sobre la regresion podemos hacer: summary(regressor)
# El valorde R-squared Multiple/Ajustado nos indica que tan buena es la regresion lineal, estan los residuos y los p-valores para contrastar la hipótesis de regresión lineal.
# Los coeficientes son IMPORTANTES: 
# (INTERCEPT) - ORDENADA DEL ORIGEN; suel do con cero años de experiencis
# PENDIENTE; que tanto aumenta el sueldo por cada año de experiencia

################################################
#                PREDICCION                    #
################################################

# OJO: Las columnas del newdata debenllamarse igual a las que se utilizaron ara crear al modelo lineal.
y_pred = predict(regressor, newdata = testing_set)

################################################
#        VISUALIZACION DE RESULTADOS           #
################################################

# Visualización de los resultados en el conjunto de entrenamiento
#install.packages("ggplot2")
library(ggplot2)
ggplot() + 
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             colour = "red") +
  geom_line(aes(x = training_set$YearsExperience, 
                y = predict(regressor, newdata = training_set)),
            colour = "blue") +
  ggtitle("Sueldo vs Años de Experiencia (Conjunto de Entrenamiento)") +
  xlab("Años de Experiencia") +
  ylab("Sueldo (en $)")

# Visualización de los resultados en el conjunto de testing
ggplot() + 
  geom_point(aes(x = testing_set$YearsExperience, y = testing_set$Salary),
             colour = "red") +
  geom_line(aes(x = training_set$YearsExperience, 
                y = predict(regressor, newdata = training_set)),
            colour = "blue") +
  ggtitle("Sueldo vs Años de Experiencia (Conjunto de Testing)") +
  xlab("Años de Experiencia") +
  ylab("Sueldo (en $)")


