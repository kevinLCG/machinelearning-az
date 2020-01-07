#!/home/kevinml/anaconda3/bin/python3.7
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 12:39:36 2019

@author: juangabriel and Kevin Meza
"""

# K-Means

# =======================================================================================================
# PASOS
# 
# 1.- Elegir el numero "k" de CLusters
# 2.- Seleccionar al azar "k" puntos cualesquiera del dominio donde se esta trabajando (no tenen que 
#     pertenecer al dataset), los Baricentros o Centros de Gravedad HIPOTETICOS. El baricentro es el punto
#     que queda en medio de cada cluster.
# 3.- Asignar cada uno de los puntos del dataset al baricentro mas cercano. Formando asi "k" clusters.
# 4.- Calcular y asignar nuevos baricentros para cada Cluster.
# 5.- Reasignar cada punto al Baricentro mas cercano.
# 6.- Repetir los pasos 4 y 5 hasta que ningun punto sea asignado a un BAricentro diferente.
# 
# =======================================================================================================
# 
# Si se utiliza la distancia Euclidiana para establecer que puntos estan mas cerca de cada baricentro,
# los clusters seran redonditos, mientras que si se utiliza la distancia Manhattan, los clusters quedaran
# mas cuadrados. en resumen, la forma de los clusters, depende del tipo de distancia que se elija, y esta
# as su vez depende del tipo de problema que se trate. 
# 
# =======================================================================================================
# 
# TRAMPA DE LA INICIACION ALEATORIA DE BARICENTROS
#
# Si se hace una mala eleccion de baricentros o si se hace una seleccion de ellos al azar, se pueden
# obtener, resultados finales, sesgados por esta eleccion.
# Esiste una solucion y por eso se creo el algoritmo: "k-means++"
#
# =======================================================================================================
# 
# SELECCION DEL NUMERO DE CLUSTERS
#
# Se usa una metrica para evaluar si un cierto numero de clusters es mejor que otro. Se usa "WCSS" 
# /(suma de los cuadrados de los centros de los Clusters). Que es una suma compuesta de tantas sumatorias
# como clusters se esten evaluando, donde cada sumatoria calcula la suma de todas las distancias de los
# puntos de un cluster con respecto al centro geometrico del mismo.
# Se busca minimizar WCSS; aunque claro que si se crean tantos clusters como puntos WCSS seria cero (0),
# pues tiende a disminuir su magnitud entre mayor sea el numero de clusters, por lo que se busca aumentar
# el numero de clusters hasta que ya no haya una disminucion significativa del valor de WSCC.
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

# Método del codo para averiguar el número óptimo de clusters
from sklearn.cluster import KMeans
wcss = []
# En el for indicamos el numero de clusters para los que se calculara el valor de WSCC.
for i in range(1, 11):
    # El parametro "init", hace referencia a la inicializacion de los baricentros, se selecciona
    # en este caso se selecciona "k-means++" para que la inicializacion de los baricentros no sea tan aleatoria.
    # Con el parametro "n_init", elegimos la inicializacion de los baricentros.
    kmeans = KMeans(n_clusters=i, init="k-means++",
                    max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    # ".inertia_" es un parametro que permite calcular el WSCC
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title("Método del codo")
plt.xlabel("Número de Clusters")
plt.ylabel("WCSS(k)")
plt.show()

# Despues de ver el plot decidimos el numero optimo de clusters y lo colocamos en el paso posterior.

#################################################################
###          APLICACION DEL CLUSTERING POR K-MEANS            ###
#################################################################

# Aplicar el método de k-means para segmentar el data set
kmeans = KMeans(n_clusters=5, init="k-means++",
                max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X)


####################################
#  Visualización de los clusters   #
####################################

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1],
            s=100, c="red", label="Cautos")
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1],
            s=100, c="blue", label="Estandard")
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1],
            s=100, c="green", label="Objetivo")
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1],
            s=100, c="cyan", label="Descuidados")
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1],
            s=100, c="magenta", label="Conservadores")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[
            :, 1], s=300, c="yellow", label="Baricentros")
plt.title("Cluster de clientes")
plt.xlabel("Ingresos anuales (en miles de $)")
plt.ylabel("Puntuación de Gastos (1-100)")
plt.legend()
plt.show()
