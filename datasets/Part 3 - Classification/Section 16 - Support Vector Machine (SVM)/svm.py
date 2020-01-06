#!/home/kevinml/anaconda3/bin/python3.7
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 19:32:40 2019

@author: juangabriel and Kevin Meza
"""

# SVM

# =======================================================================================================
# IDEA
# 
# Se busca busca obtener la mejor linea de decision que ayude a separar el espacio en las 2 clases, la
# linea que se busca ása por lo que se conoce como "margen maximo". Se crea un pasillo y la SVM detecta
# los 2 puntos mas cercanos a esta linea y a eso le llama "margen maximo", por lo que esta linea es
# equidistante a los puntos que le quedan mas cerca. 
# Los datos deben ser linealmente separables, para el caso en que el kernel sea lineal.
#
# OBJETIVO
# Se busca que la distancia entre estos puntos equidistantes sea maxima para crear el pasillo mas grande
# posible. Y a estos 2 puntos se le llaman "vectores de soporte".
# 
# Generalizando...
# Si hubiera mas variables y por lo tanto mas dimensiones al margen maximo se le llama "hiperplano de 
# marge maximo" con dimension 1 menos que el espacio donde se este trabajando.
#
# De los 2 hiperplanos paralelos, uno es el hiperplano positivo"" y el otro el "hiperplano negativo"
# (asignados de manera arbitraria).
# 
## CLASIFICACION
# Ajustan el major pasillo posible entre 2 clases, ajustando la anchura de este.
# En el caso de clasificacion los vectores "X" se utilizan para definir un hiperplano que separe las 2
# categorias de la solucion. Estos vectores se utilizan para llevar a cabo la regresion lineal.
# Los vectores que quedan mas cercanos al conjunto de Testing, son los llamados "VECTORES DE SOPORTE".
# CUIDADO: Podemos evaluar nuestra funcion en cualquier lugar, por lo que cualquier vector podria estar
# mas cerca en el conjunto de evaluacion.
#
# =======================================================================================================
#
# Son utiles cuando los datos de una categoria son parecidos a la de la otra, pues, se toma como borde
# a aquellos datos extremos que se parecen a la otra categoria.
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

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


######################################################
##   Ajustar el modelo de SVM con todo el dataset    #
######################################################

# Cargamos la libreria para importar el SVC (Support Vector Classifier)
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
from sklearn.svm import SVC
# Se elige un kernel lineal para que la separacion sea lineal, por default
# el valor es "rbf" (radial base function), es decir un kernel Gaussiano.
# El parametro c, es un factor de penalizacion que puede mejorar el modelo.
classifier = SVC(kernel="linear", random_state=0)
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
plt.title('SVM (Conjunto de Entrenamiento)')
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
plt.title('SVM (Conjunto de Test)')
plt.xlabel('Edad')
plt.ylabel('Sueldo Estimado')
plt.legend()
plt.show()
