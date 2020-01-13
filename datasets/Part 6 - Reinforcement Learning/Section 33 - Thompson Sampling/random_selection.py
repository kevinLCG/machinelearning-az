#!/home/kevinml/anaconda3/bin/python3.7
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 12:21:01 2019

@author: juangabriel and Kevin Meza
"""

# Selección Aleatoria
# Este script selcciona de manera aleatoria 10,000 rondas y 10,000 anuncios

# Importar las librarias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

################################################
###          IMPORTAR EL DATA SET            ###
################################################

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

########################################
# Implementrar una Selección Aleatoria #
########################################

import random
N = 10000  # no. de observaciones
d = 10  # no. de anuncios
ads_selected = []
total_reward = 0

# Para cada ronda se selecciona un anuncio al azar y ese se añada a la lista "ads_selected".
# A la variable "reward" se le suma el valor que este en la coordenada de la ronda y del anuncio  correspondiente.S
for n in range(0, N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    total_reward = total_reward + reward

# El valor final de reward representa el numero de veces que un usuario hizo "click" de las 10,000 veces que se mostro el anuncio.

######################################
#  Visualización de los Resultados   #
######################################

# Visualizar los resultados - Histograma
plt.hist(ads_selected)
plt.title('Histograma de selección de anuncios')
plt.xlabel('Anuncio')
plt.ylabel('Número de veces que ha sido visualizado')
plt.show()
