#!/home/kevinml/anaconda3/bin/python3.7
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 10:24:47 2019

@author: juangabriel and Kevin Meza
"""

# Clasificación con árboles de Decisión

# =======================================================================================================
# IDEA
# 
# Se va cortando en segmentos los datos con lineas rectas.
# Al mismo tiempo que se divide el conjunto, se "genera" un arbol. si la primera division fuera que X1 < 20
# se generarian entonces 2 ramas. Luego si la sig. div. es que X2 < 170, pero solo en quellos con X1 > 20, 
# se generan 2 nuevas ramas dentro de esa rama, y asi sucesivamente. Hasta que se dividan los datos de tal
# manera que a partir de cuestionar el valor de las variables se pueda determinar la clase de las nuevas
# observaciones.
# 
# Pueede darse el caso en donde ya no se puedan seguir haciendo divisiones puesto que ya existen demasiados
# niveles de decision que resulta computacionalmente muy costoso y se hacen simplificaciones, pues si es
# mucho mas probable encontrar una clase que otras, se asigna esa clase y ya no se haen mas ramificaciones;
# sobretodo cuando se trata con muchas variables.
# 
# =======================================================================================================
# 
# Mejoras de este Algoritmo:
#    - Random Forest
#    - Gradient Boosting
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

# Como el algoritmao de "Arboles de Decision" no hace uso de Distancias Euclidianas, no es necesario
# escalar las variables.

#########################################################
##   Ajustar el modelo con el dataset de Entrenamiento  #
#########################################################

# Importamos la libreria para crear el Clasificador
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
from sklearn.tree import DecisionTreeClassifier
# El parametro "criterion" hace referencia al criterio por el cual se divide una
# rama en dos ramas, por default el valor es "gini". La mayoria de los Clasificadores
# usan el criterio que minimiza la entropia "entropy", porque es facil de interpretar.
# Pues es una medida de la dispersion de la informacion; mide la calidad de las 
# divisiones para ver cual es la mejor, para que los nodos hoja sean homogeneos y no haya
# nodos hoja con observaciones con distinta clase y de esta manera se reduce la entropia del
# nodo padre al hijo.
# Entropia en un noso es igual a cero (0), el grupo es completamente homogeneo y esta puede
# clasificar con un cierto de efectividad a las observaciones en la clase correcta.

classifier = DecisionTreeClassifier(criterion="entropy", random_state=0)
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
# En la grid generamos una secuencia con una mayor diferencia entre los numeros, pues sino serian muchisimos
# numeros y # seria computacionalmente costoso crear el grafico.
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=1),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=500))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('Árbol de Decisión (Conjunto de Entrenamiento)')
plt.xlabel('Edad')
plt.ylabel('Sueldo Estimado')
plt.legend()
plt.show()


# Representación gráfica de los resultados del algoritmo en el Conjunto de Testing
X_set, y_set = X_test, y_test
# En la grid generamos una secuencia con una mayor diferencia entre los numeros, pues sino serian muchisimos
# numeros y # seria computacionalmente costoso crear el grafico.
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=1),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=500))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('Árbol de Decisión (Conjunto de Test)')
plt.xlabel('Edad')
plt.ylabel('Sueldo Estimado')
plt.legend()
plt.show()
