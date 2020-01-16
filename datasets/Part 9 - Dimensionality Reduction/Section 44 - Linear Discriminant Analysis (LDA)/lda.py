#!/home/kevinml/anaconda3/bin/python3.7
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 13:01:48 2019

@author: juangabriel and Kevin Meza
"""

# LDA

# =======================================================================================================
#
# Se basa de proyectar el espacio de caracteristicas a un espacio de dimension inferior, manteniendo la
# informacion discriminatoria lo maximo posible de cada una de las clases. Ademas de encontrar los
# ejes de las componentes de proyeccion, se tiene informacion sobre que ejes maximizan la separacion entre
# multiples clases. De las "n" variables independientes del dataset, se extraen las "p" <= n  nuevas 
# variables independientes que separen la mayoria de las clases de la variable dependiente.
#
# Como en el data set tiene que haber una columna con la clase a la que pertenecen las observaciones, es
# decir que se hace uso de la variable dependiente, este es un algoritmo supervisado.
#
# El objetivo es obtener aquellos componentes/ejes/discriminantes lineales que hagan que las clases queden lo mas separado posible
#
# PASOS:
# 1.- Aplicar escalado de variables a la matriz de caracteristicas, compuesta por "n" variables independientes
# 2.- Sea "C" el numero de clases; calcular "C" vectores m-dimensionales, de modo que cada uno contenga
#     las medias de las caracteristicas de las observaciones para cada clase. Obteniendo asi un vector con
#     las medias de todas las columnas de cada clase.
# 3.- Calcular la matrix de productos cruzados centrados en la media para cada clase, que mide la varianza
#     para cada clase.
# 4.- Se calcula la covarianza normalizada de todas las matrices anteriores, W
# 5.- Calcular la matriz de covarianza global entre clases, B
# 6.- Calculas los valores y vectores propios de la matriz. Es decir: W^-1*B
# 7.- Elegir los "p" valores propios mas grandes como el numero de dimensiones reducidas.
# 8.- Los "p" vectores propios asociados a los "p" valores propios mas grandes, son los discriminantes 	
#     lineales. El espacio m-dimensional del dataset original, se proyecta al nuevo sub-espacio p-dimensional
#     de caracteristicas, aplicando la matriz de proyecciones (que tiene los p vectores propios por columnas).
#
# Siempre hay un discriminante lineal menos que el numero de clases.
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

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


################################################
#   Reducir la dimensión del dataset con LDA   #
################################################

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# Con el parametro "n_components", seleccionamos el no. de DIRECCIONES DE SEPARACION MAXIMA(componentes).
lda = LDA(n_components = 2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

# Los datasets de Training y de Testing, ahora tendran tantas columnas como componentes principales se hayan elegido.


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
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Clasificador (Conjunto de Entrenamiento)')
plt.xlabel('DL1')
plt.ylabel('DL2')
plt.legend()
plt.show()


# Representación gráfica de los resultados del algoritmo en el Conjunto de Testing
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Clasificador (Conjunto de Test)')
plt.xlabel('DL1')
plt.ylabel('DL2')
plt.legend()
plt.show()