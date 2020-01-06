#!/home/kevinml/anaconda3/bin/python3.7
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 19:45:15 2019

@author: juangabriel and Kevin meza
"""

# Kernel SVM

# =======================================================================================================
# IDEA
#
# Puede ser que los datos no sean linealmente separables. Puede ser que se necesite hacer un circulo, una
# esfera o una elipse, como el el caso de que una clase esta rodeada por otra.
# La SVM tiene una hipotesis, y es que el limite de decision se tiene que decidir a prioro o por lo menos
# la forma que tiene el limie optimo.
#
# Se pueden llevar los datos a dimensiones superiores para que estos ya sean linealmente separables.
#
# =======================================================================================================
# TRANSFORMACION A ESPACIOS DE DIMENSION SUPERIOR
#
# Lo que hay que hacer es conseuir una funcion que aumente de dimension los datos y los haga linealmente
# separables, cuando sea posible.
# Ej. 1D -> 2D
# Si se tienen 9 datos en 1 dimension;3 pertenecen a una clase "A" y 6 a otra clae "B", los 3 puntos de la
# clase "A" se encuentran en medio de los de la clase "B", con 3 de cada lado. Se puede llevar a los puntos
# de la clase "A" cerca del cero al restarle un valos a todos los datos y loego elevar todos los datos
# y formar una parabola, en donde los datos de la clase "A" quedaran mas abajo que los de la clase "B".
# Pudiendose asi separa linealmente.
#
# Ej. 1D -> 2D
# Se tienen 2 clases "A" y "B", en donde la clae "A" se encuentra rodeada por la clase "B".
# Se le aplica a los datos una "Funcion de Tansformacion" (ahi salen el kernel radial, kernel polinomico,
# etc), para aumentarle una dimension a los datos y que queden en 3D , para que los datos puedan ser
# separados por un "hiperplano separador" (un plano en este caso).
# Se proyecta este "hiperplano separador" en la dimension original y resultaria un circulo.
#
# =======================================================================================================
# CONTRAS
#
# Mapear los datos a una dimension superior es my costoso computacionalmente, en el caso de que haya
# muchos datos, el algoritmo puede tardar muchisimo en converger.
#
# =======================================================================================================
# TRUCO DEL KERNEL
#
# En lugar de crear espacios de dimension superior, se puede permanecer en la dimension original de los
# datos y realizar la separacion, ya no con sepradores lineales, sino con otros kernels.
#
# KERNEL GAUSSIANO O RBF (Radial Base Function)
#
# Util cuando los datos tienen como limite de separacion un circulo.
#
# Se le aplica a la observacion	(x) y al "landmark" (l) (pto. de referencia con el que se va acontrastar
# donde cae la observacion) una transformacion. Se transforma c/punto en un numero que va del cero al uno,
# calculado por: k(x,l) = e^( |x-l|^2 / 2σ^2 )
# Sigma (σ) modifica la amplitud de la campana de Gauss y de este depende que los datos de una clase
# resulten levantados y los datos de la otra clase terminen practicamente planos. Para valores grandes de
# sigma, la amplitud es mayor, mientras que para valores pequeños, la amplitud es menor. Es un valor que
# se define a priori.
#
#	La funcion gaussiana se mira como una campana tridimensional, en donde todo esta concentrado a un
# punto y entre mas alejado seeste respecto de ese punto, todo queda aplanado. Este punto cuspide es el
# "landmark" o punto central, y todos los puntos se comparan contra este; entre mas alejado esten, menor
# es la exponencial y se obtiene un numero mas bajo y viceversa.
# 	A cualquir punto se le puede calcular su kernel. Primero se calcula la distancia al "landmark", esta
# distancia se eleva al cuandrado y se divide entre 2 veces el valor de "σ^2".
#
#  OJO
# Los kernels se pueden sumar k(x,l) + k(x,l). Como por ejemplo cuando el vorde de una categoria y otra
# tenga forma de "binocular". En este caso se usan 2 kernels; de igual forma, entre mas alejado est un
# punto de los kernels tendera a ser plano, mientras que entre mas cerca este, estara mas elevado.
#
# KERNEL SIGMOIDE
#
# Se elige un punto de referencia, con ayuda de una serie de parametros y dependiendo de que tan lejos
# un dato esta del punto de referencia, se obtiene un numero menor o mayor a 0.5.
# Sirve para clasificar de forma binaria hacia un lado u otro de la distribucion. A partir de un punto
# hacia la izq. todo sea de una clase y hacia la derecha sea de otra clase.
#
# KERNEL POLINOMICO
#
# Se puede elegir el grado.
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


#########################################################
##   Ajustar el modelo con el dataset de Entrenamiento  #
#########################################################

# Cargamos la libreria para importar el SVC (Support Vector Classifier)
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
from sklearn.svm import SVC
# El parametro c, es un factor de penalizacion que puede mejorar el modelo.
classifier = SVC(kernel="rbf", random_state=0)
classifier.fit(X_train, y_train)

'''
Ejemplo de Clasificador con kernel Polinomial:
classifier = SVC(kernel="poly", degree=3, random_state=0)

Ejemplo de Clasificador con kernel Sigmoide:
classifier = SVC(kernel="sigmoid", random_state=0)
'''

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
