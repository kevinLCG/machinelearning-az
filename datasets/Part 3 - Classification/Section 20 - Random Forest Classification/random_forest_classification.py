#!/home/kevinml/anaconda3/bin/python3.7
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 11:55:27 2019

@author: juangabriel and Kevin Meza
"""

# Random Forest Classification

# =======================================================================================================
# IDEA
#
# Es una version mejorada de la Clasificacion por Arbol de Decision, pues ahora son muchos de estos. De esta
# manera se reduce el error en la prediccion. Al dato nuevo se le asigna la clase que hayan eligido la
# mayoria de los arboles.
#
# PASOS:
# 1.- Se selecciona un numero "k" de puntos aleatorios del Conjunto de entrenamiento.
# 2.- Construir un arbol de decision asociado a esos "k" puntos de datos.
# 3.- Elegir un numero "n" de arboles a construir y repetir los pasos 1 y 2.
# 4.- Para clasificar un nuevo punto, los "n" arboles realizan una prediccion sobre la categoria a la que
# pertenece este dato. Y asignar le al dato la categoria con mas votos.
#
# =======================================================================================================
#
# CUIDADO
# ¡ESTE ALGORITMO TIENDE A GENERAR OVERFITTING!
#
# =======================================================================================================

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

################################################
###          IMPORTAR EL DATA SET            ###
################################################

dataset = pd.read_csv('Social_Network_Ads.csv')

X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values


#################################################################################
### Dividir el data set en conjunto de entrenamiento y conjunto de testing    ###
#################################################################################

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)


################################################
#            Escalado de variables             #
################################################

# No es encesario escalar las variables, porque el algoritmos no esta basado en
# distancias Euclidianas.
# Escalar las variables si se quiere que el grafico conserve la proporcion y quede mas fino.
# Pero si se quiere conservar a las variables con la misma escala, comentra las lineas.
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


#########################################################
##   Ajustar el modelo con el dataset de Entrenamiento  #
#########################################################

from sklearn.ensemble import RandomForestClassifier
# El parametro "n_estimators", hace referencia al numero de arboles de clasificacion a utilizar.
# El parametro "max_features", hace referencia al numero de caracteristicas que seran tomadas en
# al momento de hacer las ramificaciones, por defecto son todas.
#
# El parametro "criterion" hace referencia al criterio por el cual se divide una
# rama en dos ramas, por default el valor es "gini". La mayoria de los Clasificadores
# usan el criterio que minimiza la entropia "entropy", porque es facil de interpretar.
# Pues es una medida de la dispersion de la informacion; mide la calidad de las
# divisiones para ver cual es la mejor, para que los nodos hoja sean homogeneos y no haya
# nodos hoja con observaciones con distinta clase y de esta manera se reduce la entropia del
# nodo padre al hijo.
# Entropia en un noso es igual a cero (0), el grupo es completamente homogeneo y esta puede
# clasificar con un cierto de efectividad a las observaciones en la clase correcta.
classifier = RandomForestClassifier(
    n_estimators=10, criterion="entropy", random_state=0)
classifier.fit(X_train, y_train)

################################################
#                PREDICCION                    #
################################################

# Predicción de los resultados con el Conjunto de Testing
y_pred = classifier.predict(X_test)

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

################################################
#        VISUALIZACION DE RESULTADOS           #
################################################

# Representación gráfica de los resultados del algoritmo en el Conjunto de Entrenamiento
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('Random Forest  (Conjunto de Entrenamiento)')
plt.xlabel('Edad')
plt.ylabel('Sueldo Estimado')
plt.legend()
plt.show()


# Representación gráfica de los resultados del algoritmo en el Conjunto de Testing
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('Random Forest (Conjunto de Test)')
plt.xlabel('Edad')
plt.ylabel('Sueldo Estimado')
plt.legend()
plt.show()
