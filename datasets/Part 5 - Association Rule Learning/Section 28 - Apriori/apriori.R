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
# 
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
# 3.- Elegir todas las reglas de estos subconjuntos con nivel de confianza superior al minimo elegido.
# 4.- Ordenar todas las reglas anteriores por "lift" descendiente.
#
# =======================================================================================================

# Preprocesado de Datos
# install.packages("arules")
# Cargamos el paquete para Reglas de Asociacion
library(arules)

################################################
###          IMPORTAR EL DATA SET            ###
################################################

setwd("~/Documentos/Udemy/machinelearning-az/datasets/Part 5 - Association Rule Learning/Section 28 - Apriori")
dataset = read.csv("Market_Basket_Optimisation.csv", header = FALSE)
# PASO IMPORTANTE
# En este paso, lo que era una tabla donde cada fila era una observacion y cada
# columna tenia el producto que se compro (informacion desorganizada, una columna tiene diferentes productos),
# se convierte en una "Matriz Dispersa" o ""Matriz Parser donde cada fila sea una observacion y cada columna un producto diferente; la 
# matriz se llena de 1's y 0's, dependiendo si se compro el producto en cada observacion.

# Volvemos a leer el archivo, indicando el separador de las columnas
# El parametro "rm.duplicates" sirve para borrar los elementos diplicados, para que no haya filas repetidas.
dataset = read.transactions("Market_Basket_Optimisation.csv",
                            sep = ",", rm.duplicates = TRUE)
# Hacemos un summary, para obtener informacion de este objeto
summary(dataset)
# En el summary, la densidad de las columnas se refiere a que proporcion de la informacion
# no es cero, generalmente casi todo va a ser cero.

##################################
#     Plot de Frecuencias        #
##################################

# Hacemos un plot para mostrar los objetos y su frecuencia.
# ¡SE SUELE VER LA POWER LAW!
# El parametro "topN", hace referencia a cuantos productos se muestran de los mas frecuentes.
itemFrequencyPlot(dataset, topN = 20)

################################################
#      Entrenar el algoritmo de Apriori        #
################################################

# En los parametros se puede elegir el nivel de soprte y de confianza MINIMOS que tendran las Reglas de Asociacion.
# Un soporte de 0.05 representa atodos aquellos objetos con una frecuencia mayor al 5%
rules = apriori(data = dataset, 
                parameter = list(support = 0.004, confidence = 0.5))

# Estos datos representan las ventas de una semana, para saber el soporte que incluiria
# a todos aquellos objetos que se vendieron por lo menos 3 veces diarias se obtendria el
# siguiente soporte minimo: 3*7/total de ventas.
  
################################################
#        VISUALIZACION DE RESULTADOS           #
################################################

# Ordenamos al objeto "rules", con respecto a su "lift" y seleccionamos las primeras 10.
inspect(sort(rules, by = 'lift')[1:10])
  
  
  