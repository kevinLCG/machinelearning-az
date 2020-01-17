#!/home/kevinml/anaconda3/bin/python3.7
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 18:38:15 2019

@author: juangabriel and Kevin Meza
"""
# =======================================================================================================
#
# Las SVM (Support Vector Machines) sirven para resolver problemas de REGRESION y de CLASIFICACION, en
# este caso de trabaja con regresion, por lo que se les llama SVR (Support Vector Regresion).
#
# =======================================================================================================
#
# IDEA DE SVR
#
# Se hace uso de SVM, que sirven tanto para Regresiones Lineales y NO Lineales, el Kernel determina el
# tipo de regresion que se llevara a cabo.
# La idea es ajustar el pasillo (o calle) intentando mantener la mayor cantidad de observaciones
# posibles del conjunto de datos dentro del pasillo en torno a la recta. Ajustando y limitando la anchura
# del pasillo, conocido como "Margen Maximo"; Esta se controla mediante el hiperparameto epsilon, entre
# mayor sea el valor de este, mayor sera la anchura de pasillo.
# Cada uno de las resctas que representan los bordes/limites del pasillo, representan los potenciales
# vectores de soporte.
#
# El algoritmo hara una regresion lineal en un espacio vectorial de dimension superior a la dimension de
# los datos; cada punto del conjunto de entrenamiento representa su propia dimension. Por lo que al
# evaluar el kernel entre un punto de test y uno de entrenamiento, el resultado sera una coordenada
# trasladada a esa dimension superior.
#
# Al aplicarle la evaluacion de la SVR, a un punto de testing, este se trasladara al espacio de dimension
# superior, ese nuevo vector se representara como "k". Teniendo este vector en esta dimension, ahora si
# se lleva a cabo la regresion.
#
# El resultado es la transformacion de la recta en el espacio de dimension superior, al proyectarla en
# la dimension del conjunto de datos.
#
# # La SVM aproximara la funcion f, tal que aplicada al dominio "X", me de como resultados los puntos "y".
#
# =======================================================================================================
#
# PASOS
# 1.- Tener un conjunto de Entrenamiento. Se necesita que este cubra todo el dominio de interes y
#     vaya acompañado de las soluciones en dicho dominio. Este conjunto estara formado por la matriz
#     de caracteristixa "X" y la variable "y" a predecir
#
# 2.- Elegir un nucleo (una funcion) y sus parametros. Llevar a cabo cuanlquier regularizacion necesaria
#     (como eliminar el ruido del conjunto de entrenamiento).
#     Nucleos:
#            * Lineal (x,y)
#            * No Lineal (φ(x),φ(y)) = matriz K(x,y)
#               - Gaussiano
#     Regularizacion:
#            * Ruido
#
# 3.- Crear la matriz de Correlaciones "K".
#
# 4.- Se resuelve el sistema de ecuaciones Kα = y; Donde y = vector de valores del conjunto de entrenamiento,
#     K = matriz de correlacion, α = conjunto de incognitas para las que se resuelve el sist. de ec.
#     Esto de resuelve como: α = K(^-1)*y - PASO DE OPTIMIZACION
#     Se resuelve de forma EXACTA si se invierte la matriz o de forma APROXIMADA si se utiliza algun
#     metodo numerico, para obtener los coeficientes de contraccion "α".
#
# 5.- Utilizar los coeficientes anteriores y el kernel para crear un estimador "y*" que sea capaz de dar
#     la prediccion, tal que  y* = f(X,α,x*).
#     Calculando primero le vector de correlaciones "k^->" y luego se obtiene la prediccion "y*" a partir
#     de un conjunto de Testing "x*", donde: y* = α(^->)*k(^->)
#
# =======================================================================================================
#
# DIFERENCIAS CON LA REGRESION LINEAL SIMPLE
# La SVR tiene como objetivo que los errores no superen el umbral establecido, mientrs que en la regresion
# lineal simple se intenta minimizar el error entre la prediccion y los datos.
#
# =======================================================================================================
#
# RECURSOS
# http://alex.smola.org/papers/2004/SmoSch04.pdf
# https://www.mathworks.com/help/stats/understanding-support-vector-machine-regression.html
# https:/stats.stackexchange.com/questions/82044/how-does-support-vector-regression-work-intuitively
#
# EJEMPLOS
# https://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html
#
# =======================================================================================================


# CLASIFICACION
# Ajustan el major pasillo posible entre 2 clases, ajustando la anchura de este.
# En el caso de clasificacion los vectores "X" se utilizan para definir un hiperplano que separe las 2
# categorias de la solucion. Estos vectores se utilizan para llevar a cabo la regresion lineal.
# Los vectores que quedan mas cercanos al conjunto de Testing, son los llamados "VECTORES DE SOPORTE".
# CUIDADO: Podemos evaluar nuestra funcion en cualquier lugar, por lo que cualquier vector podria estar
# mas cerca en el conjunto de evaluacion.

# SVR

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

################################################
###          IMPORTAR EL DATA SET            ###
################################################

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


#################################################################################
### Dividir el data set en conjunto de entrenamiento y conjunto de testing    ###
#################################################################################

"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""

################################################
#            Escalado de variables             #
################################################

# En este caso es muy importante hacer este paso, sino pueden ocurrir errores
# Escalamos el escalado de la variable "X" y de "y", antes se habia hecho del
# conjunto de Entrenamiento y de Testing.

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
# redimensionamos a y para que sea un vector columna (matriz).
y = sc_y.fit_transform(y.reshape(-1, 1))

# Lo malo de escalar las variables, es que al hacer la grafica, los
# ejes pierden el sentido que tenian. Aunque hay una funcion inversa que
# te permite darle sentido a las predicciones.

########################################################
#  Ajustar el modelo con el dataset de Entrenamiento   #
########################################################

from sklearn.svm import SVR
# Por defecto el kernel es rbf (Radial Base Function), un tipo de gaussiano. Puedes poner
# uno lineal, uno polinomial, etc.
regression = SVR(kernel="rbf", gamma="auto")
regression.fit(X, y)

################################################
#                PREDICCION                    #
################################################

# Predicción de nuestros modelos con SVR
# Aplicamos la prediccion al escalado del valor 6.5 y 2, luego invocamos a la
# funcion inversa de transform, para que el resultado obtenido tenga sentido
# nuevamente.
y_pred = sc_y.inverse_transform(regression.predict(
    sc_X.transform(np.array([[6.5], [2]]))))

################################################
#        VISUALIZACION DE RESULTADOS           #
################################################

# Visualización de los resultados del SVR
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color="red")
plt.plot(X_grid, regression.predict(X_grid), color="blue")
plt.title("Modelo de Regresión (SVR)")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()

# La SVR menosprecia aquellos valores atipicos o outlayers, por lo que no se ajusta
# muy bien con el sueldo del CEO
