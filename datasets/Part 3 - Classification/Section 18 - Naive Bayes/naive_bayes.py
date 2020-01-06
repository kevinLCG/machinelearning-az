#!/home/kevinml/anaconda3/bin/python3.7
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 18:53:21 2019

@author: juangabriel and Kevin Meza
"""

# Naïve Bayes

# =======================================================================================================
# IDEA
# 
# Se tienen 2 clases "A" y "B" y entra un nuevo dato. Se busca la pobabilidad de que provenga de "A" o de 
# "B". La probabilidad se obtiene a travez de informacion obtenida del propio contexto.
# 
# Teorema de Bayes
# P(clase|X) = ( P(X|clase)*P(clase) ) / ( P(X) )
# Donde: "X" son las caracteristicas variables del nuevo dato.
#        P(clase) - Es la Pobabilidad a Priori
#        P(X) - Es la Probabilidad Marginal con respecto a las caracteristicas
#        P(X|clase) - Es la Pobabilidad Condicionada
#        P(clase|X) - Es la Pobabilidad Posterior
#
# Para obtener la probabilidad marginal se genera un circulo con radio arbitrario para obtener
# observaciones similares y se cuenta cuantas observaciones caen dentro de ese circulo. La probabilidad
# marginal seria l numero de observaciones similares entre el total de observaciones.
# 
# Para obtener la probabilidad condicionada "P(X|clase)", se obtiene considerando unicamente a los individuos
# de la clase en cuestion. De estos individuos se cuenta cuantos de estos caen dentro del circulo de
# observaciones similares al nuevo dato. Este nuemro se divide entre el numero de observaciones totales
# de esa clsae. 
# 
# En el caso de "Bayesianos Ingenous", lo que se hace es aplicar el Teorema de Bayes tantas veces como
# clases haya y obtener las probabilidades de que el nuevo dato pertenezca a cada una de las clases.
# Se comparan las probabilidades y al nuevo dato se le asigna la clase que sea mas probable.
# 
# =======================================================================================================
# 
# PORQUE BAYESIANOS "INGENUOS"
# Porque supone una independencia entre los datos que aparecen dentro de las probabilidades, es decir que 
# las variables sean independientes, que muchas veces no es cierta y por eso se hace una suposicion 
# "ingenua".
# 
# P(X)
# Numero de observaciones similares, entre observaciones totales. Como este valor siempre es el mismo para
# el calculo de la Probabilidad Posterior de todas las clases, se puede omitir este valor y no afectara el
# resultado de la comparacion. NOTA: El resultado obtenido ya no sera la probabilidad de que el dato
# pertenezca a una clase determinada.
#
# MULTIPLES CLASES
# Haecr el calculo de todas las probabilidades y compararlas de forma habitual.
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

X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values


#################################################################################
### Dividir el data set en conjunto de entrenamiento y conjunto de testing    ###
#################################################################################

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


################################################
#            Escalado de variables             #
################################################

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#########################################################
##   Ajustar el modelo con el dataset de Entrenamiento  #
#########################################################

# Importamos la libreria para generar nuestro clasificador
# https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
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
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naïve Bayes (Conjunto de Entrenamiento)')
plt.xlabel('Edad')
plt.ylabel('Sueldo Estimado')
plt.legend()
plt.show()


# Representación gráfica de los resultados del algoritmo en el Conjunto de Testing
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naïve Bayes (Conjunto de Test)')
plt.xlabel('Edad')
plt.ylabel('Sueldo Estimado')
plt.legend()
plt.show()
