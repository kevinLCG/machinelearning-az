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
#                               c) Se elige la variable predictora con el -p-value mas grande. Si su p-value es mayor que S.L. se elimina dicha variable.
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
# 5.- Compacarion de scores: Generar TODOS los modelos posibles, todos los de 1 variable, todos los de 2 variables, etc. 
#                            En total habra (2^n)-1 modelos; donde n es el numero de variables.
#                            NOTA: No es el mejor modelo cuando se tienen muchas variables.
#                            a) Elegir un criterio de bondad del ajuste (p.e. criterio bayesiano basandonos en R^2, Akaike)
#                            b) Selecionar el modelo con le mejor scroe de acuerdo al criterio elejido.
#
# NOTA: 2,3 y 4 forman parte de la familia "Regresion Paso a Paso"
#
# =================================================================================================

# Regresión Lineal Múltiple

# Importar el dataset
dataset = read.csv('50_Startups.csv')
#dataset = dataset[, 2:3]

# Codificar las variables categóricas
dataset$State = factor(dataset$State,
                         levels = c("New York", "California", "Florida"),
                         labels = c(1, 2, 3))



# Dividir los datos en conjunto de entrenamiento y conjunto de test
# install.packages("caTools")
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)

# Escalado de valores
# training_set[,2:3] = scale(training_set[,2:3])
# testing_set[,2:3] = scale(testing_set[,2:3])

# Ajustar el modelo de Regresión Lineal Múltiple con el Conjunto de Entrenamiento
regression = lm(formula = Profit ~ .,
                data = training_set)

# Predecir los resultados con el conjunto de testing
y_pred = predict(regression, newdata = testing_set)

# Construir un modelo óptimo con la Eliminación hacia atrás
SL = 0.05
regression = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
                data = dataset)
summary(regression)

regression = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
                data = dataset)
summary(regression)

regression = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
                data = dataset)
summary(regression)

regression = lm(formula = Profit ~ R.D.Spend,
                data = dataset)
summary(regression)
