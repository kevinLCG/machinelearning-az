#!/home/kevinml/anaconda3/bin/python3.7
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 16:47:17 2019

@author: juangabriel and Kevin Meza
"""

# k - Fold Cross Validation

# =======================================================================================================
#
# Normalmente se divide el dataset en un conjunto de Entrenamiento y en otro de Testing y esta no es la
# mejor manera, pues se pueden tener problemas de varianza en los datos. Es decir que si se vuelve a 
# ejecutar el modelo con conjunto de diferentes, la precision en la prediccion puede variar. Por lo que
# juzgar el rendimiento de un modelo con 1 solo conjunto de pruea, NO ES EL MEJOR ENFOQUE POSIBLE.
#
# En el "k-fold Cross Validation" se reemuestrea varias veces el dataset de Entrenamiento.
# El dataset se divide en "k" partes, y en cada iteracion se utilizaran "k-1" partes para entrenar y
# 1 parte para evaluar, de modo que en cada iteracion se use una parte diferente para realizar la evaluacion.
#   Todos los datos habran sido utilizados para entrenar y evaluar, evitando los sesgos generados cuando
# un conjunto de datos de evaluacion tiene mucha varianza, pues en las demas ocasiones no se utilizara
# para evaluar.
#
# El error final sera la desviacion estandar observada en cada una de las iteraciones, ponderada por 
# una k-esimauna parte de cada una de ellas.
#
# Sesgo Bajo: Cuando el modelo elabora predicciones cercanas a los datos reales.
# Sesgo Alto: Cuando el modelo elabora predicciones alejadas de los datos reales.
# Varianza Baja: Cuando ejecutamos el modelo varias veces y las predicciones no varian demasiado.
# Varianza Alta: Cuando ejecutamos el modelo varias veces y las predicciones varian demasiado.
#
# Habra que ver en que caso se encuentra:
# Sesgo Bajo Varianza Baja
# Sesgo Bajo Varianza Alta
# Sesgo Alto Varianza Baja
# Sesgo Alto Varianza Alta
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


#########################################################################
#  Ajustar el modelo de Clasificacion con el dataset de Entrenamiento   #
#########################################################################

from sklearn.svm import SVC
classifier = SVC(kernel="rbf", random_state=0)
classifier.fit(X_train, y_train)

################################################
#                PREDICCION                    #
################################################

# Predicción de los resultados con el Conjunto de Testing
y_pred = classifier.predict(X_test)

################################################
#        EVALUACION DEL RENDIMIENTO            #
################################################

## Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# INTERPRETACION: Las filas representan el dato real, mientras que las columnas la prediccion.
# La diagonal de izq. a der. representa aquellos casos en los que lagoritmo acerto, mientras
# que la diagonal que va de der. a izq. representa aquellos caso en los que el algoritmo fallo.
# VN  FP
# FN  VP


## Aplicar algoritmo de k-fold cross validation
from sklearn.model_selection import cross_val_score
# Los parametros, son "estimator", que se refiere al clasificadoe, "X", qque se refiere a la matriz de cracteristicas, 
# "y", que se refiere a la variable dependiente y "cv", que se refiere al numero de partes "k", en las que se dividira el dataset. de Entrenamiento
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
accuracies.mean()
accuracies.std()

# Se tendra como resultado un vector con "k" medidas de accuracy.

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
plt.title('SVM Kernel (Conjunto de Entrenamiento)')
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
