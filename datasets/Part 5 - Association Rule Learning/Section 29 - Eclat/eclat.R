# Eclat

# =======================================================================================================
#  Este algoritmo se basa en Conjuntos.
# 
#  Este algoritmo usa 1 concepto matematico:
# 
# - Soporte
#   Ej 1.  Recomendacion de peliculas. El soporte de un conjunto de peliculas "C1" esta dado por:
#   sop(C1) = Usarios que ven las pelicula del conjunto C1 / Usuarios Totales
#   Ej 2.  Optimizacion de la compra en supermerado. El soprte de un conjunto de productos "C2" esta dado por:
#   sop(C2) = Transacciones que contienen al conjunto de productos producto C2 / Transacciones Totales
# 
# Se habla del soporte de Conjuntos, no de objetos individuales.
# 
# PASOS
# 1.- Decidir un soprte minimo a utilizar.
# 2.- Elegir todos los subconjuntos de transacciones con soporte superior al minimo elegido.
# 3.- Elegir todas las reglas de estos subconjuntos con nivel de confianza superior al minimo elegido.
# 4.- Ordenar todas las reglas anteriores por "lift" descendiente.
#
#
# =======================================================================================================

# Preprocesado de Datos
#install.packages("arules")
library(arules)

################################################
###          IMPORTAR EL DATA SET            ###
################################################

setwd("~/Documentos/Udemy/machinelearning-az/datasets/Part 5 - Association Rule Learning/Section 29 - Eclat")
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
#      Entrenar el algoritmo de Eclat          #
################################################

# En los parametros se puede elegir el nivel de soprte que tendran los conjuntos.
# El parametro "minlen",  hace referencia al numero minimo de objetos que tendra cada conjunto.
rules = eclat(data = dataset, 
                parameter = list(support = 0.003, minlen = 2))

################################################
#        VISUALIZACION DE RESULTADOS           #
################################################

# El resultado seran los conjuntos de productos que aparecen juntos en mas ocasiones.
inspect(sort(rules, by = 'support')[1:10])

