#!/home/kevinml/anaconda3/bin/python3.7
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 13:10:07 2019

@author: juangabriel and Kevin Meza
"""

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

# Importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


# Codificar datos categóricos
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

# Codificaremos cada uno de los nombres de los paises
onehotencoder = make_column_transformer(
    (OneHotEncoder(), [3]), remainder="passthrough")
X = onehotencoder.fit_transform(X).astype(float)
# Evitar la trampa de las variables ficticias al quitar la primera columna
X = X[:, 1:]

# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)


# Escalado de variables
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Ajustar el modelo de Regresión lineal múltiple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

# Predicción de los resultados en el conjunto de testing
y_pred = regression.predict(X_test)

##################################################################
#               CREACION DEL MODELO DE RL MULTIPLE               #
##################################################################

# Vamos a ver si el modelo utilizando todas las variables que utilizamos es el mejor
# o si al quitar variables se obtiene un mejor resultado.

# Construir el modelo óptimo de RLM utilizando ELIMINACIÓN HACIA ATRÁS

# Podria ser que lo que fuera muy cercano a cero fuera el termino independiente, pero no hay un modo de saberlo
# directamente. Entonces se agrega 1 columa al conjunto de datos con puros 1's y el coeficiente que los tome en cuenta
# sera el del termino independiente; asi podemos calcular tambien su p-value y ver si ese valor esta cercano de cero.
import statsmodels.formula.api as sm
# El append mete los "values" hasta el final del "array", axis =1 significa columna y el 0 significa fila
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)
# Elegimos un nivel de significacion
SL = 0.05

# Creamos un array que tendra las variables optimas, al principio son todas las variables.
X_opt = X[:, [0, 1, 2, 3, 4, 5]]


# =====================================================================================
# Para la libreria sm necesitamos crear nuevamente el modelo con todas las variables
# otra vez (pues manejaa diferentes tipos de onjetos que la libreria LinearRegression).
# OLS (Ordinary List Squares), tecdnica de los minimos cuadrados ordinarios
# =====================================================================================

# Creamos una nueva regresion, "endog" es la variable a predecir y "exog" representa la
# matriz de las variables/caracteristicas. Con fit(), se hace el ajuste
regression_OLS = sm.OLS(endog=y, exog=X_opt).fit()
# Al hacer el summary, lo importante son los coeficientes, el p-value y el intervalo de confianza.
# El p-value indica que tan probables que que el coeficiente sea igual a cero; si el 0 esta dentro 
# del intervalo de confianza, el p*value sera alto.
regression_OLS.summary()

# Eliminamos la variable con el p-value mas elevado, que en este caso se encuentra en la columna 2.
# Volvemos a ajustar el modelo
X_opt = X[:, [0, 1, 3, 4, 5]]
regression_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regression_OLS.summary()

# Eliminamos la variable con el p-value mas elevado.
# Volvemos a ajustar el modelo
X_opt = X[:, [0, 3, 4, 5]]
regression_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regression_OLS.summary()

# Eliminamos la variable con el p-value mas elevado.
# Volvemos a ajustar el modelo
X_opt = X[:, [0, 3, 5]]
regression_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regression_OLS.summary()

# Eliminamos la variable con el p-value mas elevado.
# Volvemos a ajustar el modelo
X_opt = X[:, [0, 3]]
regression_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regression_OLS.summary()

# PODEMOS QUE ESTE ULTIMO P-VALUE ES 0.06, MUY CERCA DE 0.05. 
# EN ESTOS CASOS ES CONVENIENTE USAR AIC O BIC PARA DECIDIR DE MEJOR MANERA SI SE CONSERVA O NO UNA VARIABLE.