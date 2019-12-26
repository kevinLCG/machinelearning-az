# Regresión Polinómica

####################################################################################################
#
# En una regresion poinomial se tiene una variable, pero esta esta elevada a deferentes potencias.
# y = b + b1x1 + b2x1^1 + b3x1^2 + b4x1^3 + ... + bnx1^n
# Este tipo de regresion es util cuando al graficar los datos, una curva los podria describir de mejor 
# manera que una recta, como en el caso de una exponencial.
#
####################################################################################################


################################################
###          IMPORTAR EL DATA SET            ###
################################################

setwd("~/Documentos/Udemy/machinelearning-az/datasets/Part 2 - Regression/Section 6 - Polynomial Regression")
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[, 2:3]

#################################################################################
### Dividir el data set en conjunto de entrenamiento y conjunto de testing    ###
#################################################################################

# En este caso no vamos a dvidir nuestro conjunto de datos, ya que contamos con 
# un escaso numero de observaciones. No hay la suficiente informacion para
# entrenar un modelo. Ademas en este cas buscamos hacer una interpolacion
# para predecir el sueldo de un empleado entre estos niveles.
# Sin embargo este paso es escencial.

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

##################################################
#      REFUTACION LA Ho (REGRESION LINEAL)       #
##################################################

# Generamos una regresion lineal para ver lo que pasa cuando se intenta 
# ajustar datos que no son lineales. veremos despues como regresa la cosa
# con una regresion lineal polinomica.

# Ajustar Modelo de Regresión Lineal con el Conjunto de Datos
lin_reg = lm(formula = Salary ~ ., 
             data = dataset)
summary(lin_reg)

# Con la parte de Estimate (coef.) nos podemos dar cuenta de que el modelo no es el 
# mejor. Intercept es en -195,333, ¿lo que significaria que para entrar a la empresa
# abria que pagar?. Tambien nos dice esto que al aumentar cada nivel se te
# pagan $80,879 mas, lo cual no coincide con los datos.

############################################################
##  Ajustar la regresión polinómica con todo el dataset    #
############################################################

# Ajustar Modelo de Regresión Polinómica con el Conjunto de Datos
# Le damos tratamiento a las variables independientes, construimos los terminos
# del polinomio de forma explicita, para que haya tantas columnas de la variable
# independiente como grados del polinomio.
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
dataset$Level5 = dataset$Level^5
dataset$Level6 = dataset$Level^6
poly_reg = lm(formula = Salary ~ .,
              data = dataset)
summary(poly_reg)

# NOTA: Hay que ir subiendo de grado el polinomio para ir generando cada
# vez un mejor ajuste hasta quedarnos con un modelo deseable.
# Cuidado con poner grados de mas.

################################################
#                PREDICCION                    # 
################################################

# Predicción de nuevos resultados con Regresión Lineal
# Generamos un dataframe para que lo acepte la funcion, cuya unica fila es "Level"
# (mismo nombre de la columna del dataset con el que se creo el modelo).
# Si en level colucamos un vector, podemos hacer varias predicciones a la vez.
y_pred = predict(lin_reg, newdata = data.frame(Level = c(6.5,5)))

# Predicción de nuevos resultados con Regresión Polinómica
# Hacemos un dataframe con las columnas correspondiantes a los coeficientes de
# la regresion lineal polinomica.
y_pred_poly = predict(poly_reg, newdata = data.frame(Level = 6.5,
                                                     Level2 = 6.5^2,
                                                     Level3 = 6.5^3,
                                                     Level4 = 6.5^4))

################################################
#        VISUALIZACION DE RESULTADOS           #
################################################

# Visualización del modelo lineal
# install.packages("ggplot2")
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level , y = dataset$Salary),
             color = "red") +
  # Al momento de hacer la prediccion, en "newdata", con pasarle todo el dataset
  # es  suficiente, automaticamente toma la variable independiente de definimos
  # al generar nuestro modelo de regresion lineal, unicamente la columna tiene 
  # que tener el mismo nombre que el dataset usado para crear el modelo.
  geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset)),
            color = "blue") +
  ggtitle("Predicción lineal del sueldo en función del nivel del empleado") +
  xlab("Nivel del empleado") +
  ylab("Sueldo (en $)")


# Visualización del modelo polinómico

# Para evitar que la funcion continua que predice el salario ,se vea 
# como trocitos de recta, se creea una secuencia de valores que esten
# en ptos. intermedios entre el 1y el 10, que vayan de 0.1 en 0.1.
# Utilizando esta secuencia de numeros, generamos un nuevo dataframe 
# con tantas columnas como grados del polinomio. Nuevamente visualizamos el resultado.
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Level , y = dataset$Salary),
             color = "red") +
  geom_line(aes(x = x_grid, y = predict(poly_reg, 
                                        newdata = data.frame(Level = x_grid,
                                                             Level2 = x_grid^2,
                                                             Level3 = x_grid^3,
                                                             Level4 = x_grid^4,
                                                             Level5 = x_grid^5,
                                                             Level6 = x_grid^6))),
            color = "blue") +
  ggtitle("Predicción polinómica del sueldo en función del nivel del empleado") +
  xlab("Nivel del empleado") +
  ylab("Sueldo (en $)")
