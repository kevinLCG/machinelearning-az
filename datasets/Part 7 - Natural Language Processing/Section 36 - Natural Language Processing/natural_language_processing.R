# Natural Language Processing

################################################
###          IMPORTAR EL DATA SET            ###
################################################

setwd("~/Documentos/Udemy/machinelearning-az/datasets/Part 7 - Natural Language Processing/Section 36 - Natural Language Processing")
# Con el parametro "quote", indica como incian los caracteres, dejamos commillas vacias porque un string no comienza de una forma en particular
# Se tiende a convertir los textos a factores y evitamos eso al poner como FALSE el parametro "stringsAsFactors".
dataset_original = read.delim("Restaurant_Reviews.tsv", quote = '',
                     stringsAsFactors = FALSE)

######################################
###       Limpieza de texto        ###
######################################

# Se elimina la puntuacion y los numeros
# Las palabras que son declinaciones de una raiz se agrupan en un mismo conjunto, para hacer una bolsa de palabras no tan grande

# install.packages("tm") (text minig)
# install.packages("SnowballC")
library(tm)
library(SnowballC)
corpus = VCorpus(VectorSource(dataset_original$Review))
corpus = tm_map(corpus, content_transformer(tolower))
# Consultar el primer elemento del corpus
# as.character(corpus[[1]])
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords(kind = "en"))
corpus = tm_map(corpus, stemDocument)
corpus = tm_map(corpus, stripWhitespace)

######################################
###     Crear el Bag of Words      ###
######################################

dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm, 0.999)

dataset = as.data.frame(as.matrix(dtm))
dataset$Liked = dataset_original$Liked

# Codificar la variable de clasificación como factor
dataset$Liked = factor(dataset$Liked, levels = c(0,1))

#################################################################################
### Dividir el data set en conjunto de entrenamiento y conjunto de testing    ###
#################################################################################

# install.packages("caTools")
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.80)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)

#################################################################
#     AJUSTAR EL RANDOM FOREST AL DATASET DE ENTRENAMIENTO      #
#################################################################

#install.packages("randomForest")
library(randomForest)
classifier = randomForest(x = training_set[,-692],
                          y = training_set$Liked,
                          ntree = 10)

################################################
#                PREDICCION                    #
################################################

# Predicción de los resultados con el conjunto de testing
y_pred = predict(classifier, newdata = testing_set[,-692])

################################################
#        EVALUACION DEL RENDIMIENTO            #
################################################

# Crear la matriz de confusión
cm = table(testing_set[, 692], y_pred)

# INTERPRETACION: Las columnas representan el dato real, mientras que las fila la prediccion.
# La diagonal de izq. a der. representa aquellos casos en los que lagoritmo acerto, mientras
# que la diagonal que va de der. a izq. representa aquellos caso en los que el algoritmo fallo.
# VN  FN
# FP  VP
