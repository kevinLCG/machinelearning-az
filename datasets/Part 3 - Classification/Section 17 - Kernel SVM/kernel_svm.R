# Kernel SVM


# =======================================================================================================
# IDEA
#
# Puede ser que los datos no sean linealmente separables. Puede ser que se necesite hacer un circulo, una
# esfera o una elipse, como el el caso de que una clase esta rodeada por otra.
# La SVM tiene una hipotesis, y es que el limite de decision se tiene que decidir a prioro o por lo menos
# la forma que tiene el limie optimo.
#
# Se pueden llevar los datos a dimensiones superiores para que estos ya sean linealmente separables.
#
# =======================================================================================================
# TRANSFORMACION A ESPACIOS DE DIMENSION SUPERIOR
#
# Lo que hay que hacer es conseuir una funcion que aumente de dimension los datos y los haga linealmente
# separables, cuando sea posible.
# Ej. 1D -> 2D
# Si se tienen 9 datos en 1 dimension;3 pertenecen a una clase "A" y 6 a otra clae "B", los 3 puntos de la
# clase "A" se encuentran en medio de los de la clase "B", con 3 de cada lado. Se puede llevar a los puntos
# de la clase "A" cerca del cero al restarle un valos a todos los datos y loego elevar todos los datos
# y formar una parabola, en donde los datos de la clase "A" quedaran mas abajo que los de la clase "B".
# Pudiendose asi separa linealmente.
#
# Ej. 1D -> 2D
# Se tienen 2 clases "A" y "B", en donde la clae "A" se encuentra rodeada por la clase "B".
# Se le aplica a los datos una "Funcion de Tansformacion" (ahi salen el kernel radial, kernel polinomico,
# etc), para aumentarle una dimension a los datos y que queden en 3D , para que los datos puedan ser
# separados por un "hiperplano separador" (un plano en este caso).
# Se proyecta este "hiperplano separador" en la dimension original y resultaria un circulo.
#
# =======================================================================================================
# CONTRAS
#
# Mapear los datos a una dimension superior es my costoso computacionalmente, en el caso de que haya
# muchos datos, el algoritmo puede tardar muchisimo en converger.
#
# =======================================================================================================
# TRUCO DEL KERNEL
#
# En lugar de crear espacios de dimension superior, se puede permanecer en la dimension original de los
# datos y realizar la separacion, ya no con sepradores lineales, sino con otros kernels.
#
# KERNEL GAUSSIANO O RBF (Radial Base Function)
#
# Util cuando los datos tienen como limite de separacion un circulo.
#
# Se le aplica a la observacion	(x) y al "landmark" (l) (pto. de referencia con el que se va acontrastar
# donde cae la observacion) una transformacion. Se transforma c/punto en un numero que va del cero al uno,
# calculado por: k(x,l) = e^( |x-l|^2 / 2σ^2 )
# Sigma (σ) modifica la amplitud de la campana de Gauss y de este depende que los datos de una clase
# resulten levantados y los datos de la otra clase terminen practicamente planos. Para valores grandes de
# sigma, la amplitud es mayor, mientras que para valores pequeños, la amplitud es menor. Es un valor que
# se define a priori.
#
#	La funcion gaussiana se mira como una campana tridimensional, en donde todo esta concentrado a un
# punto y entre mas alejado seeste respecto de ese punto, todo queda aplanado. Este punto cuspide es el
# "landmark" o punto central, y todos los puntos se comparan contra este; entre mas alejado esten, menor
# es la exponencial y se obtiene un numero mas bajo y viceversa.
# 	A cualquir punto se le puede calcular su kernel. Primero se calcula la distancia al "landmark", esta
# distancia se eleva al cuandrado y se divide entre 2 veces el valor de "σ^2".
#
#  OJO
# Los kernels se pueden sumar k(x,l) + k(x,l). Como por ejemplo cuando el vorde de una categoria y otra
# tenga forma de "binocular". En este caso se usan 2 kernels; de igual forma, entre mas alejado est un
# punto de los kernels tendera a ser plano, mientras que entre mas cerca este, estara mas elevado.
#
# KERNEL SIGMOIDE
#
# Se elige un punto de referencia, con ayuda de una serie de parametros y dependiendo de que tan lejos
# un dato esta del punto de referencia, se obtiene un numero menor o mayor a 0.5.
# Sirve para clasificar de forma binaria hacia un lado u otro de la distribucion. A partir de un punto
# hacia la izq. todo sea de una clase y hacia la derecha sea de otra clase.
#
# KERNEL POLINOMICO
#
# Se puede elegir el grado.
#
# =======================================================================================================


################################################
###          IMPORTAR EL DATA SET            ###
################################################

setwd("~/Documentos/Udemy/machinelearning-az/datasets/Part 3 - Classification/Section 17 - Kernel SVM")
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[, 3:5]

#################################################################################
### Dividir el data set en conjunto de entrenamiento y conjunto de testing    ###
#################################################################################

# install.packages("caTools")
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)

################################################
#            Escalado de variables             #
################################################

training_set[,1:2] = scale(training_set[,1:2])
testing_set[,1:2] = scale(testing_set[,1:2])

##########################################
##   Ajustar el modelo con el dataset    #
##########################################

#install.packages("e1071")
library(e1071)
# El parametro "type" hace referencia al tipo de SVM a utilizar, y el valor "C-classification"
# indica que es para un problema de clasificacion.
# Se puede elegir un kernel Gaussiano "radial", Polinomial "polynomial" o sigmoide "sigmoid".
classifier = svm(formula = Purchased ~ .,
                 data = training_set, 
                 type = "C-classification",
                 kernel = "radial")

################################################
#                PREDICCION                    #
################################################

# Predicción de los resultados con el conjunto de testing
y_pred = predict(classifier, newdata = testing_set[,-3])

################################################
#        EVALUACION DEL RENDIMIENTO            #
################################################

# Crear la matriz de confusión
cm = table(testing_set[, 3], y_pred)

# INTERPRETACION: Las columnas representan el dato real, mientras que las fila la prediccion.
# La diagonal de izq. a der. representa aquellos casos en los que lagoritmo acerto, mientras
# que la diagonal que va de der. a izq. representa aquellos caso en los que el algoritmo fallo.
# VN  FN
# FP  VP

################################################
#        VISUALIZACION DE RESULTADOS           #
################################################

# Visualización del conjunto de entrenamiento
#install.packages("ElemStatLearn")
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'SVM Kernel (Conjunto de Entrenamiento)',
     xlab = 'Edad', ylab = 'Sueldo Estimado',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))


# Visualización del conjunto de testing
set = testing_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'SVM Kernel (Conjunto de Testing)',
     xlab = 'Edad', ylab = 'Sueldo Estimado',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))
