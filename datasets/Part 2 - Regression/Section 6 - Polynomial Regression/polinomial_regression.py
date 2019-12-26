#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 12:45:44 2019

@author: juangabriel and Kevin Meza
"""

####################################################################################################
#
# En una regresion poinomial se tiene una variable, pero esta esta elevada a deferentes potencias.
# y = b + b1x1 + b2x1^1 + b3x1^2 + b4x1^3 + ... + bnx1^n
# Este tipo de regresion es util cuando al graficar los datos, una curva los podria describir de mejor
# manera que una recta, como en el caso de una exponencial.
#
####################################################################################################


# REGRESIÓN POLINOMIAL

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

################################################
###          IMPORTAR EL DATA SET            ###
################################################

dataset = pd.read_csv('Position_Salaries.csv')
# En este caso no timaremos en cuenta la primer columna, pues provee la
# misma informacion que la segunda columna.

# X = dataset.iloc[:, 1].values
# Para los algoritmos de Machine Learning, se necesita una MATRIZ de caracteristicas.
# La sintaxis de arriba nos arroja un vector, mientras que al filtrar la informacion
# poniendo "1:2" en vez de "1", obtenemos como resultado una matriz.
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#################################################################################
### Dividir el data set en conjunto de entrenamiento y conjunto de testing    ###
#################################################################################

# En este caso no vamos a dvidir nuestro conjunto de datos, ya que contamos con
# un escaso numero de observaciones. No hay la suficiente informacion para
# entrenar un modelo. Ademas en este cas buscamos hacer una interpolacion
# para predecir el sueldo de un empleado entre estos niveles.
# Sin embargo este paso es escencial.

"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""

################################################
#            Escalado de variables             #
################################################

# No se hara. El modelo de RLP busca entender las relaciones no lineales.
# Si se escalan los datos, esta relacion NO LINEAL podria perderse.
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

##################################################
#      REFUTACION LA Ho (REGRESION LINEAL)       #
##################################################

# Generamos una regresion lineal para ver lo que pasa cuando se intenta
# ajustar datos que no son lineales. veremos despues como regresa la cosa
# con una regresion lineal polinomica.

# Ajustar la regresión lineal con todo el dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

############################################################
##  Ajustar la regresión polinómica con todo el dataset    #
############################################################

from sklearn.preprocessing import PolynomialFeatures
# Seleccionamos el grado hasta el que queremos nuestra regresion pilinomica
# Por defecto es de grado 2.
poly_reg = PolynomialFeatures(degree=6)
# Transformamos nuestra matriz de caracteristicas, para generar tantas
# nuevas columnas como grados del polinomio que hayamos elegido.
# Estas nuevas columnas tendran los valores de la columna original pero
# elevados a tantas potencias, como grados tenga el polinomio.

# Transformamos nuestra matriz. La funcion añade una colunmna de 1's hasta
# el principio que servira para la obtencion del termino independiente
# en la ecuacion polinomial.
X_poly = poly_reg.fit_transform(X)
# Ahora hacemos el ajuste polinomial y creamos nustros objeto que hace
# nuestra regresion lineal. Es la misma funcion que para la multiple.
lin_reg_2 = LinearRegression()
# Ajustamos nuestra regresion a los datos.
lin_reg_2.fit(X_poly, y)

# NOTA: Hay que ir subiendo de grado el polinomio para ir generando cada
# vez un mejor ajuste hasta quedarnos con un modelo deseable.
# Cuidado con poner grados de mas.

################################################
#                PREDICCION                    #
################################################

# Finalmente averiguaremos el sueldo que se espera de un empleado de nivel
# Predicción de nuestros modelos
# PARA EL MODELO DE RLM
lin_reg.predict([[6.5]])  # Creamos un nd adarray
# PARA EL MODELO DE RP
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))


################################################
#        VISUALIZACION DE RESULTADOS           #
################################################

# Visualización de los resultados del MODELO LINEAL
plt.scatter(X, y, color="red")
plt.plot(X, lin_reg.predict(X), color="blue")
plt.title("Modelo de Regresión Lineal")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()

# Pdemos ver que existe un mal ajuste de los datos.

# Visualización de los resultados del MODELO POLINÓMICO

# Para evitar que la funcion continua que predice el salario ,se vea
# como trocitos de recta, se creea una secuencia de valores que esten
# en ptos. intermedios entre el 1y el 10, que vayan de 0.1 en 0.1.
X_grid = np.arange(min(X), max(X), 0.1)
# Convertimos el vector a una matriz de 1 columna.
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color="red")
# Con esta parte hacemos una transfoemcion de la matriz de la X_grid
# que creeamos previamente y hacemos la prediccion de los valores.
plt.plot(X_grid, lin_reg_2.predict(
    poly_reg.fit_transform(X_grid)), color="blue")
plt.title("Modelo de Regresión Polinómica")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()
