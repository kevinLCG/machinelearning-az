# Muestreo de Thompson

# =======================================================================================================
# PASOS
#
# NOTAS: Se considera que cada recompensa puede ser diferente en cada ronda.
#        Entre MAYOR es la "n", MENOR es la amplitud del intervalo de confianza.
#
# 1.- En cada ronda "n" se consideran 2 numeros para cada anuncio "i":
#       N^1(n) = Numero de veces que el anuncio "i" recibe una recompensa 1 en la ronda "n".
#       N^0(n) = Numero de veces que el anuncio "i" recibe una recompensa 0 en la ronda "n".
# 2.- Para cada anuncio "i", se elige un valor aleatorio generado a partir de la distribucion Beta:
#     θ(n) = β(N^1(n) + 1, N^0(n) + 1)
# 3.- Se elige el anuncio con mayor valor.
# 
# 
# Casa uno de los anuncios tendra un valor de retorno; el objetivo es maximizar el retorno o bien, los
# que se le hacen al anuncio.
# Si imaginamos que en un plano, el eje "x" representa el valor de retorno.
# Las barras verticales  de cada anuncio, representan el valor promedio de retorno.
#    En un principiio se escoge un valor aleatorio para representar el valor medio de cada distribucion
# El "Muestreo de Thompson" al principio da valores de prueba, para cada distribucion, de tal forma que 
# se puede generar una idea inicila de la forma de las distribuciones. Esta distribuciones iran cambiando
# conforme avancen las rondas. Se elige cada anuncio para tomar mas muestras e nferir las distribucione y
# obtener el "valor medio" hipotetico de cada ditribucion. 
# Despues se elige al anuncio con el "valor medio" mas alto y se comienza a mostrar mas y eveltualmente 
# se obtendra un nuevo valor medio mas cerca del el "real"
# Siguiente Ronda
# Nuevamente se eligen valores aleatorios de cada distribucion y despues se juega mas en la que tenga el
# "valor medio hipotetico" mas alto y asi sucesivamente.
#    De esta forma se iran refinando las distribuciones, claramente aquellas que tuvieron el valor medio
# mas alto en mas ocasiones, tendran una distribucion mas fina y mas cercana a la realidad.
#
# =======================================================================================================
#
# SE APLICAN CONCEPTOS DE INFERENCIA BAYESIANA; origen de la Distribucion Beta
# El anuncio "i", da una recompensa "y" que sigue una Distribucion de Bernoulli:   p(y|θ) = B(θ)
# "θ" es desconocido, pero se supone que tiene una distribucion uniforme p(θ) ~ U([0,1]), llamada
# "Distribucion a priori".
#
# REGLA DE BAYES: Aproximamos a θ(n) con la distribucion a posteriori:
# p(θ|y) = (p(y|θ)*p(θ))/(∫ p(y|θ)*p(θ) dθ)   α   p(y|θ) * p(θ)
#
# Obtenemos que:  p(θ|y) ~ β(no. de exitos+1, no. de fracasos+1)
#
# A cada ronda "n" obtenemos un valor aleatorio θ(n) de la distribucion a posteriori p(θ|y), para cada "i"
# A cada ronda "n" seleccionamos el anuncio "i" con el mayor valor θ(n).
#
# =======================================================================================================

################################################
###          IMPORTAR EL DATA SET            ###
################################################

setwd("~/Documentos/Udemy/machinelearning-az/datasets/Part 6 - Reinforcement Learning/Section 33 - Thompson Sampling")
dataset = read.csv("Ads_CTR_Optimisation.csv")

##############################################################
#    Implementacion del Algoritmo de Muestreo de Thompson    #
##############################################################

# θ representa la probabilidad de exito, dado por las distribuciones de cada anuncio

d = 10
N = 10000
number_of_rewards_1 = integer(d) # contiene el numero de veces que el anuncio "i" recibe una recompensa 1 hasta la ronda "n". NUMERO DE EXITOS
number_of_rewards_0 = integer(d) # contiene el numero de veces que el anuncio "i" recibe una recompensa 0 hasta la ronda "n". NUMERO DE FRACASOS
ads_selected = integer(0) # vector con el numero de anuncio elegido en cada ronda
total_reward = 0 # recompenza total
for(n in 1:N){
  max_random = 0 # maximo valor (θ) hipotetico de todas distribuciones de probabilidad en la ronda
  ad = 0 # Contiene el numero del anuncio con el mayor valor θ
  
  # Para cada anuncio se su probabilidad de exito (θ) de la distribucion durante la ronda
  for(i in 1:d){
    # Se "crea" la propia distrubucion para el anuncio actual y se elige un numero basado en el numero basado en esta
    # Se elige un valor aleatorio generado a partir de la Distribucion Beta:
    random_beta = rbeta(n = 1,
                        shape1 = number_of_rewards_1[i]+1, # numero de exitos + 1
                        shape2 = number_of_rewards_0[i]+1) # numero de fracasos + 1
    # Si en la ronda presente, el numero θ del anuncio actual supera al maximo θ, este pasa a ser el maximo valor θ
    if(random_beta > max_random){
      max_random = random_beta
      ad = i
    }
  }
  # se añade a la lista correspondiente el anuncio "elegido", es decir, con la probabilidad de exito (θ) de la ronda
  ads_selected = append(ads_selected, ad)
  # Se guarda la recompensa de seleccionar ese anuncio en la ronda
  reward = dataset[n, ad]
  # Si la recompenza fue de 1 se suma 1 al numero de exitos de ese anuncio, pero si la recompensa es 0, se suma 1 al numero de fracasos
  if(reward == 1){
    number_of_rewards_1[ad] = number_of_rewards_1[ad] + 1
  }else{
    number_of_rewards_0[ad] = number_of_rewards_0[ad] + 1
  }
  # Se suma la recompenza de esta ronda a la recompenza total
  total_reward = total_reward + reward
}

################################################
#        VISUALIZACION DE RESULTADOS           #
################################################

# Histograma de resultados
hist(ads_selected,
     col = "lightblue",
     main = "Histograma de los Anuncios",
     xlab = "ID del Anuncio",
     ylab = "Frecuencia absoluta del anuncio")