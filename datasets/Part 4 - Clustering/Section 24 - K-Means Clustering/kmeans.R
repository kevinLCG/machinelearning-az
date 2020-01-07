# Clustering con K-means

################################################
###          IMPORTAR EL DATA SET            ###
################################################

setwd("~/Documentos/Udemy/machinelearning-az/datasets/Part 4 - Clustering/Section 24 - K-Means Clustering")
dataset = read.csv("Mall_Customers.csv")
X = dataset[, 4:5]

#####################################################
###          NUMERO OPTIMO DE CLUSTERS            ###
#####################################################

# Método del codo
set.seed(6)
wcss = vector()
# En el for indicamos el numero de clusters para los que se calculara el valor de WSCC.
for (i in 1:10){
  # En la funcion "kmeans", le damos como parametros, la matriz de caracteristicas "X" y el numero
  # de clusters "i"; del output seleccionamos "withinss", que es un vector que contiene las suma de los cuadrados de
  # las distancia de cada cluster y hacemos una suma de ese vector para obtener el valor de WSCC.
  #    Los valores de WSCC de cada no. de clusters se guardan en el vector wcss[].  
  wcss[i] <- sum(kmeans(X, i)$withinss)
}
plot(1:10, wcss, type = 'b', main = "Método del codo",
     xlab = "Número de clusters (k)", ylab = "WCSS(k)")

#################################################################
###          APLICACION DEL CLUSTERING POR K-MEANS            ###
#################################################################

# Aplicar el algoritmo de k-means con k óptimo
set.seed(29)
kmeans <- kmeans(X, 5, iter.max = 300, nstart = 10)

####################################
#  Visualización de los clusters   #
####################################

#install.packages("cluster")
library(cluster)
clusplot(X, 
         kmeans$cluster,
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
