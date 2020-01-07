# Clusterting Jerárquico

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
#           Se calculan todas las combinaciones de distancias entre los puntos de un cluster y el otro.
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

################################################
###          IMPORTAR EL DATA SET            ###
################################################

setwd("~/Documentos/Udemy/machinelearning-az/datasets/Part 4 - Clustering/Section 25 - Hierarchical Clustering")
dataset = read.csv("Mall_Customers.csv")
X = dataset[, 4:5]

#####################################################
###          NUMERO OPTIMO DE CLUSTERS            ###
#####################################################

# Utilizar el dendrograma para encontrar el número óptimo de clusters

# Primero se calcula una matriz de distancias Euclidianas entre los puntos del dataset.
# el parametro "method",, hace referencia al metodo para encontrar los clusters,
# la opcion "ward", minimiza la varianza entre los puntos de cada cluster.
dendrogram = hclust(dist(X, method = "euclidean"), 
                    method = "ward.D")
plot(dendrogram,
     main = "Dendrograma",
     xlab = "Clientes del centro comercial",
     ylab = "Distancia Euclidea")

#################################################################
#         AJUSTAR EL CLUSTERING JERARQUICO AL DATASET           #
#################################################################

hc = hclust(dist(X, method = "euclidean"), 
                    method = "ward.D")
# Una vez que se observo el dendograma, decidimos el numero de clusters
# para posteriormente cortar el arbol.
# El output es un vector con las clases a las que pertenecen cada una de
# las observaciones.
y_hc = cutree(hc, k=5)

####################################
#  Visualización de los clusters   #
####################################

#install.packages("cluster")
clusplot(X, 
         y_hc,
         lines = 0,
         shade = TRUE, # marcar el contenido del cluster
         color = TRUE,
         labels = 4, # Etiquetas de cada punto
         plotchar = FALSE, # Dieferente forma de los puntos de cada cluster
         span = TRUE, # Sirve para rodear el cluster
         main = "Clustering de clientes",
         xlab = "Ingresos anuales",
         ylab = "Puntuación (1-100)"
         )