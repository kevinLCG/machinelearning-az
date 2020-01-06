#!/home/kevinml/anaconda3/bin/python3.7
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 19:53:04 2019

@author: juangabriel and Kevin Meza
"""

# Regresión Bosques Aleatorios

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

"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

############################################################
##  Ajustar la regresión polinómica con todo el dataset    #
############################################################

# Cargamos la libreria para hacer nuestro regresor
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
from sklearn.ensemble import RandomForestRegressor
# El parametro "n_estimators", hace referencia al numero de arboles de regresion a utilizar.
# El parametro "criteriom", hace referencia al criterio con el que se dividira una rama en subramas
# o en nodos hoja, por default es "mse" error minimo cuadrado.
# El parametro "max_features", hace referencia al numero de caracteristicas que seran tomadas en
# al momento de hacer las ramificaciones, por defecto son todas.
regression = RandomForestRegressor(n_estimators=300, random_state=0)
regression.fit(X, y)

# Recomendacion: Variar el numero de arboles para ver como cambian los resultados. Las grafica no se
# vuelve mas suave al aumentar el numero de arboles, pero el resultado vaya que si aumenta.

################################################
#                PREDICCION                    #
################################################

y_pred = regression.predict([[6.5]])

################################################
#        VISUALIZACION DE RESULTADOS           #
################################################

# Visualización de los resultados del Random Forest
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color="red")
plt.plot(X_grid, regression.predict(X_grid), color="blue")
plt.title("Modelo de Regresión con Random Forest")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()
