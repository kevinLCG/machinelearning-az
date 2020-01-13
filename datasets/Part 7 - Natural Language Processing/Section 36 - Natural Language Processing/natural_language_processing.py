#!/home/kevinml/anaconda3/bin/python3.7
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 17:13:51 2019

@author: juangabriel and Kevin Meza
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
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter = "\t", quoting = 3)

######################################
###       Limpieza de texto        ###
######################################

# Se elimina la puntuacion y los numeros
# Las palabras que son declinaciones de una raiz se agrupan en un mismo conjunto, para hacer una bolsa de palabras no tan grande

import re
import nltk # natural languaje toolkit
nltk.download('stopwords') # descargamos las palabras irrelevantes
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer # Libreria para eliminar declinaciones
corpus = [] # lista con todas las oraciones limpias

# Hacemos el preprocesado del texto para cada review, dejando solo letras, apsando todo a minusculas
for i in range(0, 1000):
    # Quitamos todo lo que no sea letras, sustituyendo todo lo demas por espacios
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    # pasamos todo a minusculas
    review = review.lower()
    # tokenizamos
    review = review.split()
    ps = PorterStemmer() # creamos nuestro objeto para quitar declinaciones
    # Para cada palabra del review checamos que no esten dentro de las stopwords para conservarlas
    # y luego elimino las declinaciones de las palabras, quedandomos con las palabras raiz "stemwords".
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
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
cv = CountVectorizer(max_features = 1500) # word2vec. Transformara a las palabras en vectores de frecuencias.
X = cv.fit_transform(corpus).toarray() # volvemos un vector al objeto resultante un vector
y = dataset.iloc[:, 1].values

# Se transforman las palabras en vectores de frecuencias


#################################################################################
### Dividir el data set en conjunto de entrenamiento y conjunto de testing    ###
#################################################################################

# Cada una de las palabras/columnas sera una variable independiente. Lo que se quiere es estudiar la correlacion  
# que existe entre las palabras. Para que a partir de la presencia  o ausencia de palabras, se deduzca si el review es positivo o negativo.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


####################################################
#     AJUSTAR EL  AL DATASET DE ENTRENAMIENTO      #
####################################################

# Colocar aqui el algoritmo de clasificacion de Preferencia
# Para NLP se suelen usar: Bayesianos Ingenuos (sencillo), SVM y Arboles de Decision (para cosas mas avanzadas)

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

# Obtenemos la eficacia del Algoritmo
# (VP+VN)/TOtal de Observaciones
# Para este caso: 
(55+91)/200