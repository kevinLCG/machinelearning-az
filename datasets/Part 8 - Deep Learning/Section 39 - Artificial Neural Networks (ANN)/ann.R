# Redes Neuronales Artificiales

################################################
###          IMPORTAR EL DATA SET            ###
################################################

setwd("~/Documentos/Udemy/machinelearning-az/datasets/Part 8 - Deep Learning/Section 39 - Artificial Neural Networks (ANN)")
dataset = read.csv('Churn_Modelling.csv')
dataset = dataset[, 4:14]

################################################
#     Codificarcion de  datos categóricos      #
################################################

# Codificar los factores para la RNA
dataset$Geography = as.numeric(factor(dataset$Geography,
                                      levels = c("France", "Spain", "Germany"),
                                      labels = c(1, 2, 3)))
dataset$Gender = as.numeric(factor(dataset$Gender,
                                  levels = c("Female", "Male"),
                                  labels = c(1,2)))

#################################################################################
### Dividir el data set en conjunto de entrenamiento y conjunto de testing    ###
#################################################################################

# install.packages("caTools")
library(caTools)
set.seed(123)
split = sample.split(dataset$Exited, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)

################################################
#            Escalado de variables             #
################################################

training_set[,-11] = scale(training_set[,-11])
testing_set[,-11] = scale(testing_set[,-11])

################################################
#           Crear la red Neuronal              #
################################################

# Hay otros paquetes de Aprendizaje Profundo
# Neural NEt: sirve unicamente para hacer Regresion
# Nnet: Sirve para general redes con 1 sola capa oculta
# Deep Net
# h2o: Tiene la facilidad de conectarse a la GPU, tiene muchas opciones de configuracion, 
#     tiene muchos parametros de optimizacion.
#     Es un paquete de codigo abierto y necesita una conexion a una estancia de h2o.
#install.packages("h2o")
library(h2o)
# Nos conectamos a una instancia de h20, te puedes conectar en un servidor.
h2o.init(nthreads = -1)
# El parametro "y", indica la variable dependiente y el "training_frame", se refiere a la matriz de caracteristicas
# y automaticamente ignora la variable "dependiente". Se convierte a un objeto de h2o
# Elegimos la funcion de activacion como relu/Rectificador Lineal Unitario
# El parametro "hidden", indica la cantidad de capas ocultas y los nodos de cada capa.
# El parametro "train_samples_per_iteration", hace referencia a cada cuantas observaciones se actualizaran los pesos,
# el valor -2, se usa el valor optimo que considere el algoritmo.

# Para elegir el numero de capas ocultas, por convencion se suele elegir el promedio de capas de entrada y de salida
# en este caso serien 6.
classifier = h2o.deeplearning(y = "Exited",
                              training_frame = as.h2o(training_set),
                              activation = "Rectifier",
                              hidden = c(6, 6),
                              epochs = 100,
                              train_samples_per_iteration = -2)

################################################
#                PREDICCION                    #
################################################

# Las predicciones es este caso son probabilidades (de que deje el banco en este caso).
prob_pred = h2o.predict(classifier, 
                        newdata = as.h2o(testing_set[,-11]))
# Convertimos las probabilidades a categoerias definiendo un umbral de decision. 
# Ahora se tiene un vector booleano con True y False.
y_pred = (prob_pred>0.5)
# Convertimos el vector de prediccion de un objeto de h2o a un vector.
y_pred = as.vector(y_pred)

################################################
#        EVALUACION DEL RENDIMIENTO            #
################################################

# Crear la matriz de confusión
cm = table(testing_set[, 11], y_pred)

# INTERPRETACION: Las filas representan el dato real, mientras que las columnas la prediccion.
# La diagonal de izq. a der. representa aquellos casos en los que lagoritmo acerto, mientras
# que la diagonal que va de der. a izq. representa aquellos caso en los que el algoritmo fallo.
# VN  FP
# FN  VP

# Se puede mejorar el scor con "Cross Validation" con & con "XGBoost"

# Cerrar la sesión de H2O
h2o.shutdown()
