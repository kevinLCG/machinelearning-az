#!/home/kevinml/anaconda3/bin/python3.7
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 12:41:01 2019

@author: juangabriel and kevin Meza
"""

# Upper Confidence Bound (UCB)

# =======================================================================================================
# PASOS
#
# NOTAS: Se considera que cada recompensa puede ser diferente en cada ronda.
#        Entre MAYOR es la "n", MENOR es la amplitud del intervalo de confianza.
#
# 1.- En cada ronda "n" se consideran 2 numeros para cada anuncio "i":
#       N(n) = Numero de veces que el anuncio "i" se selecciona en la ronda "n".
#       R(n) = La suma de recompensas del anuncio "i" hasta la ronda "n".
# 2.- A partir de estos 2 numeros, se calcula:
#     - La recompensa media del anuncio "i" hasta la ronda "n".
#       r(n)=  R(n)/ N(n)
#
#     - El intervalo de confianza de la ronda "n".
#       ( r(n)-Δ(n) , r(n)+Δ(n) );     Donde:
#       Δ(n) = sqrt( 3*log(n) / 2*N(n) )
#
# 3.- Se selecciona el anuncio "i" con mayor limite superior del intervalo de confianza (UCB)
#
# En un inicio, se parte del supuesto de que las medias y los intervalos de confianza de cada una de las
# distribuciones son iguales y con al paso del tiempo al juntar observaciones, se va definiendo el valor
# medio de recompensa de cada una, al igua que los intervalos de confianza. Recordando que ntre mayor sea
# la "n", mrnor sera la amplitud del intervalo de confianza.
#
# Primero se comienza a tirar en todas las maquinas (muestreando asi todas las distribuciones) y despues
# de ciertas iteraciones, se comienza a tirar (muestrear) la maquina (la distribucion) con el mayor limite
# superior del intervalo de confianza (UCB), hasta que el algoritmo converja.
#
# =======================================================================================================

# Importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

################################################
###          IMPORTAR EL DATA SET            ###
################################################

# El dataset tiene en las filas las ronas en las que se presentan los anuncios y en las columnas estan los 20 anuncios.
# los 1's y 0's representan si el usuario dio "click" en el anuncio.
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

################################################
#    Implementacion del Algoritmo de UCB       #
################################################

import math
N = 10000  # no. de observaciones
d = 10  # no. de anuncios
# aqui se guardara el numero de veces que se muestreo cada anuncio. Vectro inicializado con 0's de tamaño igual al no. de observaciones.
number_of_selections = [0] * d
sums_of_rewards = [0] * d  # aqui se guardara la recompenza de cada anuncio
ads_selected = []  # vector con el numero de anuncio elegido en cda ronda
total_reward = 0  # recompenza total
for n in range(0, N):
    max_upper_bound = 0 # Contiene el UCB de la ronda
    ad = 0  # Contiene el numero del anuncio con el mayor intervalo de confianza

    # En la ronda actual, para cada anuncio, se obtiene la "recompensa media" y el limite superior del intervalo de confianza
    # y se actualiza el UCB si es necesario
    for i in range(0, d):
        if(number_of_selections[i] > 0):
            # Se obtiene la recompensa media
            average_reward = sums_of_rewards[i] / number_of_selections[i]
            # Se obtiene Δn, para sacar el intervalo de confianza; sumamos 1 para no dividir entre cero
            delta_i = math.sqrt(3 / 2 * math.log(n + 1) /
                                number_of_selections[i])
            # Se obtiene el limite superior del intervalo de confianza
            upper_bound = average_reward + delta_i
        else:
            # Para las primeras rondas cuando no se ha seleccionado el anuncio, se le asigna como como "upper confidence bound" el numero 10^400
            # Asi ningun anuncio sera mejor que otro en la primera ronda.
            # En la primera ronda se eligira el primer anuncio, en la siguiente ronda el segundo, despues el tercero y asi sucesivamente,
            # esto con la intencion de que al menos todos sean muestreados 1 vez, por eso el numero "10^400".
            upper_bound = 1e400
        # Si el limite superior del intervalo de confianza del actual anuncio supera al UCB, este pasa a ser el nuevo UCB
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i

    # se añade a la lista correspondiente el anuncio "elegido", es decir, con el UCB hasta esa ronda
    ads_selected.append(ad)
    # se le suma 1 al vector que contiene cuantas veces se ha elegido el anuncio
    number_of_selections[ad] = number_of_selections[ad] + 1
    # Se guarda la recompensa de seleccionar ese anuncio
    reward = dataset.values[n, ad]
    # A la recompenza previa del anuncio "elegido", se le suma la recompenza conseguida en esta ronda
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    # Se suma la recompenza de esta ronda a la recompenza total
    total_reward = total_reward + reward

# En cada ronda, siempre se va seleccionar el anuncio con el UCB

################################################
#        VISUALIZACION DE RESULTADOS           #
################################################

# Histograma de resultados
plt.hist(ads_selected)
plt.title("Histograma de anuncios")
plt.xlabel("ID del Anuncio")
plt.ylabel("Frecuencia de visualización del anuncio")
plt.show()
