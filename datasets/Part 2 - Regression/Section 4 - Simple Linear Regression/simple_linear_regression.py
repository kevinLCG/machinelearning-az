#!/home/kevinml/anaconda3/bin/python3.7
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 12:07:43 2019

@author: juangabriel and Kevin Meza
"""

# Regresión Lineal Simple

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


# Dividimos el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1 / 3, random_state=0)


# Escalado de variables
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train
X_test = sc_X.transform(X_test)
"""

# IMPORTANTE
# NO ES NECESARIO ESCALAR LAS VARIABLES. LA LIBRERIA SKLEARN YA LO HACE.

# Crear modelo de Regresión Lienal Simple con el conjunto de entrenamiento

# Cargamos una libreria
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
from sklearn.linear_model import LinearRegression

# Creamos un (objeto) modelo de regresion lineal.
regression = LinearRegression()

# Ajustamos el modelo de regresion anuestros datos de "Entrenamiento".
# (debe haber el mismo numero de filas, en la consola de iPython al momento de correrlo te da info. de los parametros con los que se corrio el comando)
regression.fit(X_train, y_train)

# Utilizamos el conjunto de variables independientes para predecir la variable independiente (salario en este caso).
# El resultado se guarda en una variable.
y_pred = regression.predict(X_test)

# Visualizar los resultados de entrenamiento
plt.scatter(X_train, y_train, color="red")
plt.plot(X_train, regression.predict(X_train), color="blue")
plt.title("Sueldo vs Años de Experiencia (Conjunto de Entrenamiento)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo (en $)")
plt.show()

# Visualizar los resultados de test
# Graficamos los valores de entrenamiento
plt.scatter(X_test, y_test, color="red")
# Graficamos la recta que predice el modelo
plt.plot(X_train, regression.predict(X_train), color="blue")
plt.title("Sueldo vs Años de Experiencia (Conjunto de Testing)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo (en $)")
plt.show()
