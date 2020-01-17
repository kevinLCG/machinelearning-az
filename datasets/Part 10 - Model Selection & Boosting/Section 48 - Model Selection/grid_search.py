#!/home/kevinml/anaconda3/bin/python3.7
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 21:09:58 2019

@author: juangabriel adn Kevin Meza
"""

# Grid Search

# =======================================================================================================
#
# Partiendo de que cualquier algoritmo de Machine Learning tiene 2 tipos de parametros:
# El primer tipo de parametros,  se aprenden a travez del propio algoritmo (por ejemplo, los coeficientes
# en una regresion lineal o los pesos en una Red Neronal); y el otro tipo de parametros (hiperparametros)
# los elige la persona que esta detras del algoritmo (por ejemplo, la eleccion del kernel en una SVM,
# parametros de penalizacion o tasas de aprendizaje en REdes Neuronales)
#
# "Grid Search" es una tecnica que intenta mejorar el rendimiento de los modelos, al encontrar los valores optimos para los
# hiperparametros del modelo. Ademas nos dira si es mejor escoger un modelo lineal o NO lineal,
# como en el caso de las SVM, donde se puede escoger un kernel lineal o no.
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
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
accuracies.mean()
accuracies.std()

#############################################################################
# Aplicar la mejora de Grid Search para otimizar el modelo y sus parámetros #
#############################################################################

from sklearn.model_selection import GridSearchCV
# Se crea un diccionario donde las llaves son los hyperparametros que se utilizan dentro del algoritmo y los valores del diccionario
# seran los valores que se quieren evaluar dentro del modelo de busqueda.
# El algoritmo construira todas las combinaciones de parametros posibles y sacara la mejor combinacion de todas.
# El hyperparametro "C" es un factor de penalizacion para evitar el Overfitting.
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [
                  0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
              ]
# Creamos el modelo que generara las combinaciones de hyperparametros
grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters, # Seleccionamos la combinacion de hypermarametros que se hicieron anteriormente
                           scoring='accuracy', # indica que medida se usara para definir que combinacion de parametros es mejor
                           cv=10, # Se indica el numero de k-cross validation. El resultado de la medida elegida anteriormente, sera la media de los valores obtenidos en las "k" veces.
                           n_jobs=-1) # nodos

# Ajustamos el Grid Search a conjunto de Entrenamiento.
grid_search = grid_search.fit(X_train, y_train)
# Guardamos el mejor valor para a medida de eficacia seleccionada anteriormente
best_accuracy = grid_search.best_score_
# Guardamos la mejor combinacion de hiperparametros
best_parameters = grid_search.best_params_

'''
Si se quiere obtener todavia mejores resultados, se puede generar otra Grid, con parametros similares
a los parametros seleccionados anteriormente como mejores.
'''

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
