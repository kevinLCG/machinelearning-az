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
#                               b) 
#                               c) 
#                               d) 
#                               e) 
# 5.- Compacarion de scores
#
# NOTA: 2,3 y 4 forman parte de la familia "Regresion Paso a Paso"
#
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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

# Evitar la trampa de las variables ficticias
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

# Construir el modelo óptimo de RLM utilizando la Eliminación hacia atrás
import statsmodels.formula.api as sm
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)
SL = 0.05

X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regression_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regression_OLS.summary()

X_opt = X[:, [0, 1, 3, 4, 5]]
regression_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regression_OLS.summary()

X_opt = X[:, [0, 3, 4, 5]]
regression_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regression_OLS.summary()

X_opt = X[:, [0, 3, 5]]
regression_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regression_OLS.summary()

X_opt = X[:, [0, 3]]
regression_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regression_OLS.summary()
