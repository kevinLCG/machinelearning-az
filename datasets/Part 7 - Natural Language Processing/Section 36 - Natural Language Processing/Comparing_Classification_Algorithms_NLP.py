#!/home/kevinml/anaconda3/bin/python3.7
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 17:13:51 2019

@author: Kevin Meza
"""

# Natural Language Processing

# Importar librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

################################################
###          IMPORTAR EL DATA SET            ###
################################################

# El parametro "quoting" en el valor 3 hace que se ignoren las commillas dobles dentro del texto y asi no se
# generen errores al momento de leer el archivo.
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter="\t", quoting=3)

######################################
###       Limpieza de texto        ###
######################################

# Se elimina la puntuacion y los numeros
# Las palabras que son declinaciones de una raiz se agrupan en un mismo conjunto, para hacer una bolsa de palabras no tan grande

import re
import nltk  # natural languaje toolkit
nltk.download('stopwords')  # descargamos las palabras irrelevantes
from nltk.corpus import stopwords
# Libreria para eliminar declinaciones
from nltk.stem.porter import PorterStemmer
corpus = []  # lista con todas las oraciones limpias

# Hacemos el preprocesado del texto para cada review, dejando solo letras, apsando todo a minusculas
for i in range(0, 1000):
    # Quitamos todo lo que no sea letras, sustituyendo todo lo demas por espacios
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    # pasamos todo a minusculas
    review = review.lower()
    # tokenizamos
    review = review.split()
    ps = PorterStemmer()  # creamos nuestro objeto para quitar declinaciones
    # Para cada palabra del review checamos que no esten dentro de las stopwords para conservarlas
    # y luego elimino las declinaciones de las palabras, quedandomos con las palabras raiz "stemwords".
    review = [ps.stem(word) for word in review if not word in set(
        stopwords.words('english'))]
    # Generamos oraciones de las listas de palabras, uniendo las palabras de las listas por una espacio en blanco
    review = ' '.join(review)
    corpus.append(review)

######################################
###     Crear el Bag of Words      ###
######################################

# El objetivo hay que hacer una matriz dispersa llena de 1's y 0's, donde las columnas son palabras
# y las filas las reseñas.

from sklearn.feature_extraction.text import CountVectorizer
# El parametro "stop_words", te permite quitar stopwords, tambien se puede tokenizar, pasar a minuscula, etc
# El parametro "max_features" indica cual es el numero de columnas/palabras maximo que habra en la matriz dispersa,
# conservando aquellas con mas frecuencia, para no tomar en cuenta aquellas poco frecuentes que no proporcionan mucha informacio.
# word2vec. Transformara a las palabras en vectores de frecuencias.
cv = CountVectorizer(max_features=1500)
# volvemos un vector al objeto resultante un vector
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Se transforman las palabras en vectores de frecuencias


#################################################################################
### Dividir el data set en conjunto de entrenamiento y conjunto de testing    ###
#################################################################################

# Cada una de las palabras/columnas sera una variable independiente. Lo que se quiere es estudiar la correlacion
# que existe entre las palabras. Para que a partir de la presencia  o ausencia de palabras, se deduzca si el review es positivo o negativo.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0)


# =======================================================================================================
#                               COMPARACION DE ALGORITMOS DE CLASIFICACION                              #
# =======================================================================================================

# ==============================================
#            BAYESIANOS INGENUOS
# ==============================================

####################################################
#     AJUSTAR EL  AL DATASET DE ENTRENAMIENTO      #
####################################################

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

################################################
#                PREDICCION                    #
################################################

# Predicción de los resultados con el Conjunto de Testing
y_pred_NB = classifier.predict(X_test)

################################################
#        EVALUACION DEL RENDIMIENTO            #
################################################

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm_NB = confusion_matrix(y_test, y_pred_NB)

# INTERPRETACION: Las filas representan el dato real, mientras que las columnas la prediccion.
# La diagonal de izq. a der. representa aquellos casos en los que lagoritmo acerto, mientras
# que la diagonal que va de der. a izq. representa aquellos caso en los que el algoritmo fallo.
# VN  FP
# FN  VP

# Accuracy
accuracy_NB = (cm_NB[0,0] + cm_NB[1,1])/(cm_NB[0,1] + cm_NB[1,0] + cm_NB[0,0] + cm_NB[1,1])
# Precission
precission_NB = (cm_NB[1,1])/(cm_NB[0,1] + cm_NB[1,1])
# Recall
recall_NB = (cm_NB[1,1])/(cm_NB[1,0] + cm_NB[1,1])
# F1-Score
F1_NB = 2 * precission_NB * recall_NB / (precission_NB + recall_NB)



# ==============================================
#             REGRESION LOGISTICA
# ==============================================



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
y_pred_LR = classifier.predict(X_test)

################################################
#        EVALUACION DEL RENDIMIENTO            #
################################################

# Elaborar una matriz de confusión
cm_LR = confusion_matrix(y_test, y_pred_LR)

# INTERPRETACION: Las filas representan el dato real, mientras que las columnas la prediccion.
# La diagonal de izq. a der. representa aquellos casos en los que lagoritmo acerto, mientras
# que la diagonal que va de der. a izq. representa aquellos caso en los que el algoritmo fallo.
# VN  FP
# FN  VP

# Accuracy
accuracy_LR = (cm_LR[0,0] + cm_LR[1,1])/(cm_LR[0,1] + cm_LR[1,0] + cm_LR[0,0] + cm_LR[1,1])
# Precission
precission_LR = (cm_LR[1,1])/(cm_LR[0,1] + cm_LR[1,1])
# Recall
recall_LR = (cm_LR[1,1])/(cm_LR[1,0] + cm_LR[1,1])
# F1-Score
F1_LR = 2 * precission_LR * recall_LR / (precission_LR + recall_LR)



# ==============================================
#                     KNN
# ==============================================



######################################################
##   Ajustar el modelo de KNN con todo el dataset    #
######################################################

# Importamos la libreria para eobtener el modelo de KNN
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
from sklearn.neighbors import KNeighborsClassifier
# El parametro "n_neighbors", hace referencia al numero de "k" vecinos para clasificar
# a los nuevos datos.
# En el parametro "metric", el valor "minkowski" es una familia de distancias;
# cuando p = 1 sale la distancia Manhattan y con p = 2 sale la distancia L" o euclidiana.
classifier = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
classifier.fit(X_train, y_train)

################################################
#                PREDICCION                    #
################################################

# Predicción de los resultados con el Conjunto de Testing
y_pred_KNN = classifier.predict(X_test)

################################################
#        EVALUACION DEL RENDIMIENTO            #
################################################

# Elaborar una matriz de confusión
cm_KNN = confusion_matrix(y_test, y_pred_KNN)

# INTERPRETACION: Las filas representan el dato real, mientras que las columnas la prediccion.
# La diagonal de izq. a der. representa aquellos casos en los que lagoritmo acerto, mientras
# que la diagonal que va de der. a izq. representa aquellos caso en los que el algoritmo fallo.
# VN  FP
# FN  VP

# Accuracy
accuracy_KNN = (cm_KNN[0,0] + cm_KNN[1,1])/(cm_KNN[0,1] + cm_KNN[1,0] + cm_KNN[0,0] + cm_KNN[1,1])
# Precission
precission_KNN = (cm_KNN[1,1])/(cm_KNN[0,1] + cm_KNN[1,1])
# Recall
recall_KNN = (cm_KNN[1,1])/(cm_KNN[1,0] + cm_KNN[1,1])
# F1-Score
F1_KNN = 2 * precission_KNN * recall_KNN / (precission_KNN + recall_KNN)



# ==============================================
#                     SVM
# ==============================================



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
y_pred_SVM = classifier.predict(X_test)

################################################
#        EVALUACION DEL RENDIMIENTO            #
################################################

# Elaborar una matriz de confusión
cm_SVM = confusion_matrix(y_test, y_pred_SVM)

# INTERPRETACION: Las filas representan el dato real, mientras que las columnas la prediccion.
# La diagonal de izq. a der. representa aquellos casos en los que lagoritmo acerto, mientras
# que la diagonal que va de der. a izq. representa aquellos caso en los que el algoritmo fallo.
# VN  FP
# FN  VP

# Accuracy
accuracy_SVM = (cm_SVM[0,0] + cm_SVM[1,1])/(cm_SVM[0,1] + cm_SVM[1,0] + cm_SVM[0,0] + cm_SVM[1,1])
# Precission
precission_SVM = (cm_SVM[1,1])/(cm_SVM[0,1] + cm_SVM[1,1])
# Recall
recall_SVM = (cm_SVM[1,1])/(cm_SVM[1,0] + cm_SVM[1,1])
# F1-Score
F1_SVM = 2 * precission_SVM * recall_SVM / (precission_SVM + recall_SVM)



# ==============================================
#                 DECISION TREE
# ==============================================



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
y_pred_DT = classifier.predict(X_test)

################################################
#        EVALUACION DEL RENDIMIENTO            #
################################################

# Elaborar una matriz de confusión
cm_DT = confusion_matrix(y_test, y_pred_DT)

# INTERPRETACION: Las filas representan el dato real, mientras que las columnas la prediccion.
# La diagonal de izq. a der. representa aquellos casos en los que lagoritmo acerto, mientras
# que la diagonal que va de der. a izq. representa aquellos caso en los que el algoritmo fallo.
# VN  FP
# FN  VP

# Accuracy
accuracy_DT = (cm_DT[0,0] + cm_DT[1,1])/(cm_DT[0,1] + cm_DT[1,0] + cm_DT[0,0] + cm_DT[1,1])
# Precission
precission_DT = (cm_DT[1,1])/(cm_DT[0,1] + cm_DT[1,1])
# Recall
recall_DT = (cm_DT[1,1])/(cm_DT[1,0] + cm_DT[1,1])
# F1-Score
F1_DT = 2 * precission_DT * recall_DT / (precission_DT + recall_DT)



# ==============================================
#               RANDOM FOREST
# ==============================================



########################################################
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
y_pred_RF = classifier.predict(X_test)

################################################
#        EVALUACION DEL RENDIMIENTO            #
################################################

# Elaborar una matriz de confusión
cm_RF = confusion_matrix(y_test, y_pred_RF)

# INTERPRETACION: Las filas representan el dato real, mientras que las columnas la prediccion.
# La diagonal de izq. a der. representa aquellos casos en los que lagoritmo acerto, mientras
# que la diagonal que va de der. a izq. representa aquellos caso en los que el algoritmo fallo.
# VN  FP
# FN  VP

# Accuracy
accuracy_RF = (cm_RF[0,0] + cm_RF[1,1])/(cm_RF[0,1] + cm_RF[1,0] + cm_RF[0,0] + cm_RF[1,1])
# Precission
precission_RF = (cm_RF[1,1])/(cm_RF[0,1] + cm_RF[1,1])
# Recall
recall_RF = (cm_RF[1,1])/(cm_RF[1,0] + cm_RF[1,1])
# F1-Score
F1_RF = 2 * precission_RF * recall_RF / (precission_RF + recall_RF)



# ===================================================================================================
#                  RESULTADOS DE LA COMPARACION DE MODELOS DE CLASIFICACION
# ===================================================================================================

# Bayesianos Ingenuos
print("Bayesianos Ingenuos")
print("cm = " + str(cm_NB))
print("accuracy = " + str(accuracy_NB))
print("precission = " + str(precission_NB))
print("recall = " + str(recall_NB))
print("F1-Score = " + str(F1_NB))

# Regresion Logistica
print("Regresion Logistica")
print("cm = " + str(cm_LR))
print("accuracy = " + str(accuracy_LR))
print("precission = " + str(precission_LR))
print("recall = " + str(recall_LR))
print("F1-Score = " + str(F1_LR))

# KNN
print("KNN")
print("cm = " + str(cm_KNN))
print("accuracy = " + str(accuracy_KNN))
print("precission = " + str(precission_KNN))
print("recall = " + str(recall_KNN))
print("F1-Score = " + str(F1_KNN))

# SVM
print("SVM")
print("cm = " + str(cm_SVM))
print("accuracy = " + str(accuracy_SVM))
print("precission = " + str(precission_SVM))
print("recall = " + str(recall_SVM))
print("F1-Score = " + str(F1_SVM))

# Arbol de Decision
print("Arbol de Decision")
print("cm = " + str(cm_DT))
print("accuracy = " + str(accuracy_DT))
print("precission = " + str(precission_DT))
print("recall = " + str(recall_DT))
print("F1-Score = " + str(F1_DT))

# Random Forest
print("Random Forest")
print("cm = " + str(cm_RF))
print("accuracy = " + str(accuracy_RF))
print("precission = " + str(precission_RF))
print("recall = " + str(recall_RF))
print("F1-Score = " + str(F1_RF))


accuracy_DT
accuracy_RF
accuracy_SVM
accuracy_KNN
accuracy_NB
accuracy_KNN