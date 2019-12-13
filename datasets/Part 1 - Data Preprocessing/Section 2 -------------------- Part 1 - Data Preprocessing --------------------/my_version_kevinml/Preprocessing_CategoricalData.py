#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 11:47:39 2019
@author: kevinml
Version Python: 3.7
"""
# Pre Procesado - Datos Categóricos

###########################################################
#                       Input Dataset                     #
###########################################################

# Importamos las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Importar el data set
dataset = pd.read_csv('./Data.csv')

# Generamos un subdataset con las variales independientes y otro con las dependientes
# INDEPENDIENTES (matriz)
X = dataset.iloc[:, :-1].values
# DEPENDIENTES (vector)
y = dataset.iloc[:, 3].values


###########################################################
#                  Tratamiento de los NAs                 #
###########################################################

# Importamos las librerías
# https://scikit-learn.org/stable/modules/impute.html
from sklearn.impute import SimpleImputer

# Creamos una funcion para reemplazar los valores faltantes (NaN/np.nan) con la MEDIA/mediana/most_frequent/etc de los valores de la COLUMNA (axis=0).
imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")

# Hacemos unos ajustes a nuestra funcion para solo aplicarla a las columnas con datos faltantes.
imputer = imputer.fit(X[:, 1:3])
# Sobreescribimos nuestra matriz, haciendo las sustituciones correspondientes.
X[:, 1:3] = imputer.transform(X[:,1:3])


###########################################################
#            Codificacion de Datos Categoricos            #
###########################################################

# Importamos las librerías
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
# https://scikit-learn.org/stable/modules/generated/sklearn.compose.make_column_transformer.html
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import make_column_transformer

# Le pasamos a la funcion tuplas (transformador, columnas) que especifiquen los
# objetos del transformador que se aplicarán a los subconjuntos de datos.
# Las columnas no seleccionadas se ignoraran

# Codificaremos cada uno de los nombres de los paises
onehotencoder = make_column_transformer((OneHotEncoder(), [0]), remainder = "passthrough")
X = onehotencoder.fit_transform(X).toarray()

# Codificamos el valor de Purchase "Yes", "No" por "1", "0"
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

###########################################################
#             Training & Testing Splitting                #
###########################################################

# Dividir el data set en conjunto de entrenamiento y conjunto de testing

# Importamos las librerías
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
from sklearn.model_selection import train_test_split

# Obtenemos 4 variables; caracteristicas y etiquetas, de entrenamiento y testing respectivamente. Colocamos semilla en 42.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

###########################################################
#                 Escalado de Variables                   #
###########################################################

# Esto se hace devido a que el rango dinamico de cada una de las variables diferentes
# y al momento de operar con ellas, como al momento de sacar distancias euclidianas
# el valor de las variables de mayor rango, puede opacar el de aquellas cuyo rango sea menor.
# Obtenderemos variables entre -1 y 1.

# Importamos las librerías
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
from sklearn.preprocessing import StandardScaler

# Creamos el objeto escalador
sc_X = StandardScaler()
# Generamos un escalador de acuerdo con nuestros datos de entrenamiento.
X_train = sc_X.fit_transform(X_train)
# Utilizamos es escalador obtenido en el paso anterior, para escalar nuestros datos de testing.
X_test = sc_X.transform(X_test)