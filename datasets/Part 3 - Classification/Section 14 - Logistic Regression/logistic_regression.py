#!/home/kevinml/anaconda3/bin/python3.7
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:50:03 2019

@author: juangabriel
"""

# Regresión Logística

# =======================================================================================================
# IDEA
#
# Proviene una regresion lineal (y = b0 + b1*x), pero se aplica una funcion sigmoide (p = 1/(1 + e^-y))
# para transformar el valor "y". A la prediccion se le aplica una funcion sigmoide, para transformar el
# valor final de la regresion en una probabilidad.
#
# Si se toma un ejemplo de clasificacion binaria (Si/No); en la funion sigmoide, "p" representa la
# probabilidad de un Si.
#
# REGRESION LOGISTICA
# ln(p/1-p) = b0 + b1*x
# Lo que antes era una recta, se convierte ahora en una funcion Logistica, una funcion sigmoide.
# Es una funcion que se interpreta del mismo modo que la pendiente de una regresion lineal, solo que
# un poco curva y modela la tendencia a ambos grupos.
#
# Se busca la mejor linea sigmoidea que mejor se ajuste a los datos.
# El eje y se transforma e una probabilidad.
#
#
# El resultado es la probabilidad de que un suceso ocurra. Se proyecta el valor de la variable del eje x
# en la curva logistica obtenida, la probabilidades se obtienen al proyectar estos valores, ahora sobre
# el eje y.
#
# Lo que se hace despues es definir una probabilidad (se suele utilizar .50); todas aquellas probabilidades
# inferiores a esta, se proyectan a que es mas probable un "No". Mientras que todas los valores por encima
# de esta probabilidad, se proyectaran a un "Si".
#
# =======================================================================================================
#
# Dado que la regresion logistica es lineal, la frontera entre las clases sea una linea recta, sino seria
# un plano o un hiperplano.
# El clasificadoe se basa en una regresion lineal simple por lo que la recta obtenida sera la mejor dado
# umbral, pues se hace use de los minimos cuadrados.
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
# Escalamos las variables independientes
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# En este caso no hace falta escalar la variable dependiente porque por si misma
# ya es una categoria (0 o 1).

######################################################################
##   Ajustar el modelo de regresión logistica con todo el dataset    #
######################################################################

# Cargamos la libreria para generar nuestro modelo de clasificacion
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0, solver="liblinear")
classifier.fit(X_train, y_train)

################################################
#                PREDICCION                    #
################################################

# La prediccion es un vector con con cada una de las clases elegidas (0 o 1), ya no es la probabilidad.

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

# El color de los puntos representan la clase verdadera, mientras que el color
# de fondo representa lo que el modedlo predijo (zonas de prediccion).

# Recordar que el resultado obtenido es la mejor linea recta que se separa a los puntos.

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
plt.title('Clasificador (Conjunto de Entrenamiento)')
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
plt.title('Clasificador (Conjunto de Test)')
plt.xlabel('Edad')
plt.ylabel('Sueldo Estimado')
plt.legend()
plt.show()
