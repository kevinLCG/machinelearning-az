#!/home/kevinml/anaconda3/bin/python3.7
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 10:38:56 2019

@author: juangabriel and Kevin Meza
"""

# Redes Neuronales Artificales

# Instalar Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Instalar Tensorflow y Keras
# conda install -c conda-forge keras

# ========================================================================
# Parte 1 - Pre procesado de datos

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

################################################
###          IMPORTAR EL DATA SET            ###
################################################

dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

################################################
#     Codificarcion de  datos categóricos      #
################################################

# Cambiamos las variables categoricas de pais y genero en variables dummy.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
# Quitamos una de las filas que corresponden a los paises para no caer en la trampa de las variables dummy
# y asi evitar la milticolinealidad.
X = X[:, 1:]

#################################################################################
### Dividir el data set en conjunto de entrenamiento y conjunto de testing    ###
#################################################################################

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

################################################
#            Escalado de variables             #
################################################

# Este paso es muy importante porque los valores de las variables se multiplicaran por los pesos
# y no queremos sesgar al modelo a que unas variables pesen mas que otras.

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# ========================================================================
# Parte 2 - Construir la RNA

# Importar Keras y librerías adicionales
import keras  # importa Tensorflow
# Se necesitan 2 sublibrerias adicionales
# Sirve para inicializar parametros de red neuronal
from keras.models import Sequential
from keras.layers import Dense  # Sirve para cerar capas ocultas de la red neuronal

# ----------- Inicializar la RNA -----------

# Existen 2 formas de inicializar la RNA: 1. Definir la secuencia de capas.
# 2. Definir como se van a relacionar las capas.
# El modelo neuronal hara el papel de clasificador.
# En los parametros puedes agregar las capas, pero en este caso las haremos de forma separada.
classifier = Sequential()

# ----------- Añadir las capas de entrada y primera capa oculta -----------

# El parametro "units", hace referencia a la salida, numero de neuronas que tiene la primera capa oculta,
# se suele poner la media entre los nodos de la capa de entrada y los nodos de salida.
# El parametro "kernel_initializer", hace referencia a como inicializar los pesos "w", se pueden iniciar con respecto
# a una funcion uniforme, una constante.
# El parametro "input_dim", indica la entrada, el numero de nodos de la capa de  entrada.
# Definimos al Rectificador Lineal Unitario como funcion de activacion.
classifier.add(Dense(units=6, kernel_initializer="uniform",
                     activation="relu", input_dim=11))

# ----------- Añadir la segunda capa oculta -----------

# El parametro de entrada ya no hace falta porque la red sabe que recibira 6 entradas, de la capa anterior.
classifier.add(
    Dense(units=6, kernel_initializer="uniform",  activation="relu"))

"""
Se pueden crean mas de una capa oculta con el siguiente codigo:
no_capas_profundas = 2
for i in range(1:no_capas_profundas+1):
    classifier.add(Dense(units=128, activation="relu"))
"""

# ----------- Añadir la capa de salida -----------

# Definimos el valor del parametro de salida "units" en 1, porque la salida sera 1 nodo.
# Cambiamos la funcion de activacion a la "funcion sigmoide".
# En el caso de tener varias categorias, el valor de nodos de salida sera igual al numero de categorias
# Con varias categorias a la funcion sigmoide se le debe aplicar un "Soft Max" para que todas las probabilidades sumen 1
# o usar otras funciones de activacion como relu o la funcion escalon.
classifier.add(
    Dense(units=1, kernel_initializer="uniform",  activation="sigmoid"))

# ----------- Compilar la RNA -----------

# El parametro "loss", al algoritmo que se utiliza para encontrar el conjunto optimo de pesos.
# se puede elegir: "gradient descent" o "esthocastic gradient descent"
# El parametro "loss", hace referencia a la funcion de coste a utilizar.
# En "metrics", le damos una lista para evaluar el modelo y buscar que aumenenten estas metricas durante el entrenamiento.
classifier.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# ----------- Ajustamos la RNA al Conjunto de Entrenamiento -----------

# El parametro "batch_size", hace referencia a cada cuantas observaciones se actualizaran los pesos.
classifier.fit(X_train, y_train,  batch_size=10, epochs=100)

# ========================================================================
# Parte 3 - Evaluar el modelo y calcular predicciones finales

################################################
#                PREDICCION                    #
################################################

# Predicción de los resultados con el Conjunto de Testing
# Las predicciones es este caso son probabilidades (de que deje el banco en este caso).
y_pred = classifier.predict(X_test)
# Convertimos las probabilidades a categoerias definiendo un umbral de decision.
# Ahora se tiene un vector booleano con True y False.
y_pred = (y_pred > 0.5)

################################################
#        EVALUACION DEL RENDIMIENTO            #
################################################

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# INTERPRETACION: Las filas representan el dato real, mientras que las columnas la prediccion.
# La diagonal de izq. a der. representa aquellos casos en los que lagoritmo acerto, mientras
# que la diagonal que va de der. a izq. representa aquellos caso en los que el algoritmo fallo.
# VN  FP
# FN  VP
