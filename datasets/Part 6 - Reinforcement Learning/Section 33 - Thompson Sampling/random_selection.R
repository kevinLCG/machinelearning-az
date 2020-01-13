# Random Selection

################################################
###          IMPORTAR EL DATA SET            ###
################################################

setwd("~/Documentos/Udemy/machinelearning-az/datasets/Part 6 - Reinforcement Learning/Section 33 - Thompson Sampling")
dataset = read.csv('Ads_CTR_Optimisation.csv')

########################################
# Implementrar una Selección Aleatoria #
########################################

N = 10000 # no. de observaciones
d = 10 # no. de anuncios

# Para cada ronda se selecciona un anuncio al azar y ese se añada a la lista "ads_selected".
# A la variable "reward" se le suma el valor que este en la coordenada de la ronda y del anuncio  correspondiente.S
ads_selected = integer(0)
total_reward = 0
for (n in 1:N) {
  ad = sample(1:10, 1)
  ads_selected = append(ads_selected, ad)
  reward = dataset[n, ad]
  total_reward = total_reward + reward
}

# El valor final de reward representa el numero de veces que un usuario hizo "click" de las 10,000 veces que se mostro el anuncio.

######################################
#  Visualización de los Resultados   #
######################################

hist(ads_selected,
     col = 'blue',
     main = 'Histogram of ads selections',
     xlab = 'Ads',
     ylab = 'Number of times each ad was selected')