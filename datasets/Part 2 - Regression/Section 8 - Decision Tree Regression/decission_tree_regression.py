#!/home/kevinml/anaconda3/bin/python3.7
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 19:04:40 2019

@author: juangabriel and Kevin Meza
"""

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


# Regresión con Árboles de Decisión

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

################################################
###          IMPORTAR EL DATA SET            ###
################################################

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


#################################################################################
### Dividir el data set en conjunto de entrenamiento y conjunto de testing    ###
#################################################################################

# En este caso no se hara por la escases de datos

"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""

################################################
#            Escalado de variables             #
################################################

# Para los arboles de decision no suele hacerse escalado de variables, porque
# el algoritmo no utiliza distancias euclidianas.
# Recomendado: Ver como sale el modelo sin escalar variables y luego escalar.

"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

########################################################
#  Ajustar el modelo con el dataset de Entrenamiento   #
########################################################

#importamos la libreria para hcaer nuestro regresor
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
from sklearn.tree import DecisionTreeRegressor
# Es recomendable checar los  parametros, como donde hacer cortes en el arbol.
# se suele utilizar la medida del error cuadrado medio, para minimizar la 
# diferencia al cuadrado entre la prediccion y el resultado. De esta forma
# se busca la forma de cortar minimiza los cuadrados de los errores.
# Otro parametro importante es el tipo de lineas que se trazan,
# verticales y horizontales o incluyendo diagonales, dependiento si se toman
# en cuenta 1 o 2 caracteristicas ara hacer las divisiones de los conjuntos.
# Mas parametros = no. max. de nodos hoja, no. max. de elementos por nodo
regression = DecisionTreeRegressor(random_state=0)
regression.fit(X, y)

################################################
#                PREDICCION                    #
################################################

y_pred = regression.predict([[6.5]])

################################################
#        VISUALIZACION DE RESULTADOS           #
################################################

plt.scatter(X, y, color="red")
plt.plot(X, regression.predict(X), color="blue")
plt.title("Modelo de Regresión")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()

# En este plot podemos ver realmente lo que pasa en el algoritmo, al asignar la media como valor
# de la variable independiente. Ademas esto es lo que pasa cuando cada nodo contiene a 1 solo individuo.
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color="red")
plt.plot(X, regression.predict(X_grid), color="blue")
plt.title("Modelo de Regresión")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()