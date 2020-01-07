#!/home/kevinml/anaconda3/bin/python3.7
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 13:15:02 2019

@author: juangabriel and Kevin Meza
"""

# Clustering Jerárquico

# =======================================================================================================
# PASOS
#
# Hay 2 tipos de agrupaciones jerarquicos: "Aglomerativo" (de abajo hacia arriba) y "Divisitivo" (de arriba
# hacia abajo).
#
# Clustering Jerárquico Aglomerativo
# Junta 1 por 1 los elementos similares para formar grupos.
#
# PASOS
# 1.- Hacer que cada punto sea un cluster.
# 2.- Se eligen los 2 puntos mas cercanos y se juntan en un unico cluster.
# 3.- Se eligen los 2 clusters mas cercanos y se juntan en un unico cluster.
# 4.- Repetir el paso 3 hasta tener un unico cluster.
#
# Para definir los puntos o clusters mas cercanos, se hace uso de Distancias Euclidianas generalmente;
# tambien se puede hacer uso de Distancia Manhattan, Distancia Minkowski, etc.
#
# Distancia entre Clusters
# ###########################
# OPCION 1: Se cacula a partir de los puntos mas cercanos entre los clusters.
# OPCION 2: Se cacula a partir de los puntos mas lejanos entre los clusters.
# OPCION 3: Se cacula la distancia media.
# 			Se calculan todas las combinaciones de distancias entre los puntos de un cluster y el otro.
# OPCION 4: Se cacula la distancia entre los baricentros de los clusters.
#
# Se representan de forma visual con un DENDOGRAMA
# Una vez con el dendograma, para obtener el numero de cluster en los que se dividen los datos, se tiene
# que elegir un umbral de distancia Euclidiana (que representa disimilaridad) para cortar el Dendograma.
# Dependiendo del umbral que se elija, es el munero de clusters resultantes.
#
# Numero Optimo de Clusters
# ############################
# Una REGLA es que la linea que corta el dendograma debe pasar por la linea vertical mas alta, que
# representa la disimilaridad de un cluster, con respecto al anterior. CON LA CONDICION, de que esta linea
# vertical NO cruce ningunarecta horizontal.
# El numero de clusters sera la cantidad de lineas verticales que corte la linea horizontal.
#
# Mas Dimensiones
# Se necesita aplcar una tecnica de reduccion de Dimensionalidad y luego aplicar el metodo. 
#
# =======================================================================================================

# Importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

################################################
###          IMPORTAR EL DATA SET            ###
################################################

dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:, [3, 4]].values

#####################################################
###          NUMERO OPTIMO DE CLUSTERS            ###
#####################################################

# Utilizar el dendrograma para encontrar el número óptimo de clusters

# Importamos la libreria para generar el dendograma
import scipy.cluster.hierarchy as sch
# El parametro "method", hace referencia al metodo para encontrar los clusters,
# la opcion "ward", minimiza la varianza entre los puntos de cada cluster.
dendrogram = sch.dendrogram(sch.linkage(X, method="ward"))
plt.title("Dendrograma")
plt.xlabel("Clientes")
plt.ylabel("Distancia Euclídea")
plt.show()

#################################################################
#         AJUSTAR EL CLUSTERING JERARQUICO AL DATASET           #
#################################################################

# Importamos la libreria para generar los clusters.
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
from sklearn.cluster import AgglomerativeClustering
# El parametro "affinity", hace referencia al tipo de distancia que se va a utilizar
# El parametro "linkage", hace referencia al metodo con el que se unen los clusters.
# Dado que el dendograma se uso el metodo de "ward" se usara tambien, otras opciones son la
# distancia minima o la distancia media.
hc = AgglomerativeClustering(
    n_clusters=5, affinity="euclidean", linkage="ward")
y_hc = hc.fit_predict(X)

####################################
#  Visualización de los clusters   #
####################################

plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s=100, c="red", label="Cautos")
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1],
            s=100, c="blue", label="Estandard")
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1],
            s=100, c="green", label="Objetivo")
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1],
            s=100, c="cyan", label="Descuidados")
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s=100,
            c="magenta", label="Conservadores")
plt.title("Cluster de clientes")
plt.xlabel("Ingresos anuales (en miles de $)")
plt.ylabel("Puntuación de Gastos (1-100)")
plt.legend()
plt.show()
