# =================================================================================================
# NOTAS IMPORTANTES:
#
# Para que podamos aplicar la regresion lineal multiple, hay que tener en cuenta que nuestro modelo
# tiene que cumplir con lo siguiente:
# 1.- Linealidad
# 2.- Homocedasticidad = Cuando la varianza del error condicional a las variables explicativas es
#                        constante a lo largo de las observaciones.
# 3.- Normalidad Multivariable
# 4.- Independencia de los Errores
# 5.- Ausencia de Multicolinealidad
#
# Si hay variables categoricas, se tienen que convertir a OneHot Encoders (variables Dummy).
# El (numero de categorias-1) sera la cantidad de parametros que tendra el modelo de regresion lineal.
# Ej: categorias: rojo, azul. Entonces: y = b0 + ... + bn*D1 (no ponemos bn+1*D2)
# Esto porque aunque no la coloquemos, toda la información ya es tomada en cuenta por el modelo.
# Tomarla en cuenta causaria Multicolinealidad.
#
# =================================================================================================
# CREACION DEL MODELO DE REGRESION LINEAL MULTIPLE:
#
# 1.- Exahustivo (All-in):	Incluir todas las variables. Util como preparacion previa para luego implementar el 2.
# 2.- Eliminacion hacia atras:
#                               a) Elegir nivel de significación (S.L.) para que una variable permanezca en el modelo; normalmente se usa S.L.= 0.05.
#                               b) Se calcula el modelo con todas las variables predictoras (all-in)
#                               c) Se elige la variable predictora con el p-value mas grande. Si su p-value es mayor que S.L. se elimina dicha variable.
#                               d) Ajustar el modelo sindicha variable.
#                               e) Repetir pasos c) y d) hasta que la variable con el p-value mas grande sea menor al S.L.
# 3.- Seleccion hacia adelante:
#                               a) Elegir nivel de significación (S.L.) para que una variable entre en el modelo; normalmente se usa S.L.= 0.05.
#                               b) Ajustamos todos los modelos de regresion lineal simple; utilizando cada una de las variables independientes con respecto a la dependiente (y~Xn).
#                                  Se elige el modelo con el p-value mas pequeño y conservamos dicha variable
#                               c) Ajustamos todos los posibles modelos con una variable extra a las que ya tenga el modelo hasta ese momento.
#                               d) Consideramos la variable predictora con el p-value mas pequeño. Si su p-value es menor que S.L., repetimos c), sino FIN y nos quedamos con el modelo anterior.
# 4.- Eliminacion Bidireccional:
#                               a) Elegir nivel de significación (S.L.) para que una variable entre en el modelo (SLENTER) y otra para que permanezca (SLSTAY); normalmente se usa S.L.= 0.05.
#                               b) Realozamos el paso b) de "Seleccion hacia adelante" y nos quedamos con un modelo.
#                               c) Realizamos TODOS los pasos de "Eliminacion hacia atras". Las variables antiguas deben tener p-value < SLSTAY (pudiendo eliminar mas de una).
#                               d) Repetir b) y c) hasta que no haya variables ni para entrar ni para salir.
# 5.- Compacarion de scores
#
# NOTA: 2,3 y 4 forman parte de la familia "Regresion Paso a Paso"
#
# Si un coeficiente de alguna variable, dentro de la fórmula de regresion lineal multiple es cero o cercano a cero,
# la variable no deberia de ser considerada.
# El p-valueu nos da una idea sobre que tan probable es que un coeficiente de la RLM sea igual a cero.
# =================================================================================================

# REGRESIÓN LINEAL MÚLTIPLE

################################################
###          IMPORTAR EL DATA SET            ###
################################################

dataset = read.csv('/home/kevinml/Documentos/Udemy/machinelearning-az/datasets/Part 2 - Regression/Section 5 - Multiple Linear Regression/50_Startups.csv')
#dataset = dataset[, 2:3]

# Codificar las variables categóricas
dataset$State = factor(dataset$State,
                         levels = c("New York", "California", "Florida"),
                         labels = c(1, 2, 3))

# ===========================================================================
# ¡OJO!
# Aqui no se tuvo la necesidad de evitar la "trampa de las variables Dummy",
# esto porque la libreria catools ya lo hace al convertir las variables categoricas
# a factores, evitando asi el problema de multicolinealidad.
# ===========================================================================

#################################################################################
### Dividir el data set en conjunto de entrenamiento y conjunto de testing    ###
#################################################################################

library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)

################################################
#            Escalado de variables             #
################################################

# training_set[,2:3] = scale(training_set[,2:3])
# testing_set[,2:3] = scale(testing_set[,2:3])

###########################################################################
#  Ajustar la regresión lineal múltiple con el dataset de entrenamiento   #
###########################################################################

# El punto hace refrencia a todas las demas variables, es lo mismo que: R.D.Spend + Administration + Marketing.Spend + State
regression = lm(formula = Profit ~ .,
                data = training_set)
summary(regression)

# Predecir los resultados con el conjunto de testing
y_pred = predict(regression, newdata = testing_set)

########################################################################
# Construir el modelo óptimo de RLM utilizando ELIMINACIÓN HACIA ATRÁS #
########################################################################

# Empezamos  utilizando un modelo que considera todas las variables.
SL = 0.05
regression = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
                data = dataset)
summary(regression)

# Eliminamos la columna State, fijarse que se quitan todas las variables dummy,
# pues se esta eliminando la columna completa.
regression = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
                data = dataset)
summary(regression)

regression = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
                data = dataset)
summary(regression)

regression = lm(formula = Profit ~ R.D.Spend,
                data = dataset)
summary(regression)


############################################################################
# ELIMINACION HACIA ATRAS AUTOMATICA
backwardElimination <- function(x, sl) {
numVars = length(x)
for (i in c(1:numVars)){
  regressor = lm(formula = Profit ~ ., data = x)
  maxVar = max(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"])
  if (maxVar > sl){
    j = which(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"] == maxVar)
    x = x[, -j]
  }
  numVars = numVars - 1
}
return(summary(regressor))
}

SL = 0.05
dataset = dataset[, c(1,2,3,4,5)]
backwardElimination(training_set, SL)
 
############################################################################
