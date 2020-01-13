#!/home/kevinml/anaconda3/bin/python3.7
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 19:05:55 2019

@author: juangabriel and Kevin meza
"""

# Apriori

# =======================================================================================================
#  Este algoritmo tiene 3 soportes matematicos:
#
# - Soporte
#   Ej 1.  Recomendacion de peliculas. El soporte de una pelicula "M1" esta dado por:
#   sop(M1) = Usuarios que ven la pelicula M1 / Usuarios Totales
#   Ej 2.  Optimizacion de la compra en supermerado. El soprte  de un producto "I1" esta dado por:
#   sop(I1) = Transacciones que contienen al producto I1 / Transacciones Totales
#
# - Confianza
#   Ej 1.  Recomendacion de peliculas. La confianza de que cuando se ve una pelicula "M1" tambien se ve
#   otra pelicula "M2" esta dada por:
#   conf(M1 -> M2) = Usuarios que ven la pelicula M1 y M2 / Ususarios que ven la pelicula M1
#   Ej 2.  Optimizacion de la compra en supermerado. La confianza de que cuando una transaccion incluye
#   al producto "I1" tambien incluye al producto "I2" esta dada por:
#   conf(I1 -> I2) = Transacciones que contienen al producto I1 e I2 / Transacciones que contienen al producto I1

#   En otras palabras, se refiere a que tan frencuente es ver una pelicula o comprar un objeto del total de eventos.
#   Entre mayor sea la confianza, mas relevantes son las reglas resultantes. Reduciendo el sesgo creado por
#   los objetos mas frecuentes.
#
# - Lift
#   Es una forma de "mejorar" una respuesta aleatoria, conociendo algo "a riori".
#   Ej 1.  Recomendacion de peliculas.
#   lift(M1 -> M2) = conf(M1 -> M2) / sop(M1)
#   Interpretacion: cual es la probabilidad que vea la pelicula M2, dado que le gusto la pelicula M1
#   Ej 2.  Optimizacion de la compra en supermerado.
#   lift(I1 -> I2) = conf(I1 -> I2) / sop(I1)
#   Interpretacion: cual es la probabilidad que la transacion incluya al producto M2, dado que incluye al
#   producto M1.
#
#
# PASOS
# 1.- Decidir un soprte y nivel de confianza minimo a utilizar.
# 2.- Elegir todos los subconjuntos de transacciones con soporte superior al minimo elegido.
# 3.- Ordenar todas las reglas anteriores por "soporte" descendiente.
#
# =======================================================================================================

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

################################################
###          IMPORTAR EL DATA SET            ###
################################################

dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)
transactions = []
# Vamos a generar una lista de listas que contenga la misma informacion que la tabla.
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])


################################################
#      Entrenar el algoritmo de Apriori        #
################################################

# Importamos una libreria local para generar un objeto de clase apriori.
from apyori import apriori
# En los parametros se puede elegir el nivel de soprte y de confianza MINIMOS que tndran las Reglas de Asociacion.
# Un soporte de 0.05 representa atodos aquellos objetos con una frecuencia mayor al 5%
# Tambien se puede elegie un valor minimo para el "lift".
# El parametro "min_length",  hace referencia al numero minimo de objetos, de los cuales
# se inferira la compra de otro producto.

rules = apriori(transactions, min_support=0.003, min_confidence=0.5,
                min_lift=3, min_length=2)

# Estos datos representan las ventas de una semana, para saber el soporte que incluiria
# a todos aquellos objetos que se vendieron por lo menos 3 veces diarias se obtendria el
# siguiente soporte minimo: 3*7/total de ventas.

################################################
#        VISUALIZACION DE RESULTADOS           #
################################################

results = list(rules)

# Para ver los resultados de una manera legible, acceder a los valores del objeto "results" por medio
# del indice.
# Por la propia construccion del objeto, los primeros casos correspondemn a aquellos con mayor valor de "lift".
results[0]
