#!/home/kevinml/anaconda3/bin/python3.7
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 12:06:32 2019

@author: juangabriel and Kevin Meza
"""

# PCA

# =======================================================================================================
#
# Resources:
# http://setosa.io/ev/principal-component-analysis/
#
# IDEA
# De las "m" variables independientes del dataset, se estrane las "p" <= m, nuevas variables independientes
# que explican la mayor parte de la varianza del dataset, sin importar el valor de la variable dependiente.
# Como no se hace uso de la variable dependiente, es un algoritmo no supervisado.
#
# Despues de aplicar el PCA, al graficar las observaciones, estas se encontrarian distribuidas en el
# espacio en 2 componentes, de modo que los 2 componentes principales, serian en donde se observa la mayor
# varianza entre los datos.
#
# PASOS:
# 1.- Aplicar el escalado de variables a la matriz de caracteristicas "X", formada por "m" variables
#     independientes.
# 2.- Calcular la matriz de covarianzas de las "m" variables independientes de "x".
# 3.- Calcular los valores y vectores propios de la matriz de covarianzas.
# 4.- Elegir un porcentaje "P" de varianza explicada y elegir los p <= m valores porpios mas grandes,
#     tales que:  (Σ^p λj) / (Σ^m λi) > P
# 5.- Los "p" vectores propios asociados a estos "p" valores propios mas grandes, son los componentes principales.
#     El espacio m-dimensional del dataset original se proyectara al nuevo espacio p-dimensional de
#     caracteristicas, aplicando la matriz de proyecciones (que tiene los "p" vectores propios por columnas).
#
# Dataset:
# https://archive.ics.uci.edu/ml/datasets/Wine
#
# =======================================================================================================

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

################################################
###          IMPORTAR EL DATA SET            ###
################################################

dataset = pd.read_csv('Wine.csv')

X = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13].values


#################################################################################
### Dividir el data set en conjunto de entrenamiento y conjunto de testing    ###
#################################################################################

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


################################################
#            Escalado de variables             #
################################################

# Este paso es muy importante. Hay que centrar la variables en 0 y con desviacion estandar 1.
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

################################################
#   Reducir la dimensión del dataset con PCA   #
################################################

# Se aplica la transformacion al conjunto de Entrenamiento y al de Testing.

from sklearn.decomposition import PCA
# Con el parametro "n_components", seleccionamos el no. de componentes principales,
# si el valor de este parametro es "None", se calcula un analisis con todas las componentes principales y posteriormente se puede
# visualizar que porcentaje de la varianza explica cada componente.
pca = PCA(n_components = 2)
# Los datasets de Training y de Testing, ahora tendran tantas columnas como componentes principales se hayan elegido.
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
# En esta parte se crea una variabe que contendra cada uno de los componentes principales y la varianza que explica cada uno de estos.
explained_variance = pca.explained_variance_ratio_

#########################################################
#    Ajustar el modelo con el dataset de Entrenamiento  #
#########################################################

# En esta parte se agrega el modelo de clasificacion que se decida.

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

################################################
#                PREDICCION                    #
################################################

# Predicción de los resultados con el Conjunto de Testing
y_pred  = classifier.predict(X_test)

################################################
#        EVALUACION DEL RENDIMIENTO            #
################################################

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# INTERPRETACION: Las columnas representan el dato real, mientras que las filas la prediccion.
# La dimension de esta matriz dependera de la cantidad de clases que haya, es decir que si hay 4 clases
# la matriz dera de 4*4, por ejemplo.
# La diagonal de izq. a der. representa aquellos casos en los que lagoritmo acerto, mientras
# que las demas casillas representan aquellos caso en los que el algoritmo fallo.

################################################
#        VISUALIZACION DE RESULTADOS           #
################################################

# Representación gráfica de los resultados del algoritmo en el Conjunto de Entrenamiento
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.8, cmap = ListedColormap(('red', 'green', 'blue'))) # añadir tantos colores como categorias
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Clasificador (Conjunto de Entrenamiento)')
plt.xlabel('CP1')
plt.ylabel('CP2')
plt.legend()
plt.show()


# Representación gráfica de los resultados del algoritmo en el Conjunto de Testing
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue'))) # añadir tantos colores como categorias
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Clasificador (Conjunto de Test)')
plt.xlabel('CP1')
plt.ylabel('CP2')
plt.legend()
plt.show()
