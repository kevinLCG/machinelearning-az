#!/home/kevinml/anaconda3/bin/python3.7
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 19:20:59 2019

@author: juangabriel and Kevin Meza
"""

# Redes Neuronales Convolucionales

# Instalar Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Instalar Tensorflow y Keras
# conda install -c conda-forge keras

# ===================================================================================================
#
# NOTAS:
# Se necesita tener en 2 carpetas separadas el congunto de "Entrenamiento" y el conjunto de "Testing".
# Adentro de las carpetas anteriores, debera haber tantas carpetas como clases haya en el problema de
# clasificacion, cada una conteniendo imagenes correspondiendo a esa clase.
# En este caso hay 4,000 imagenes de cada clase para entrenar y 1,000 imagenes de cada clase para
# evaluar, OJO: teniendo la misma cantidad de datos para cada clase.
# En este caso no es necesaria la parte de "Preprocesado de datos", pues se hace directamente con keras,
# ademas de que la informacion de entrada son pixeles.
#
# Pensar si es necesario mantener las imagenes a color y si eso influenciara en los resultados, sino
# se puede quitar el color y reducir el coste computacional.
#
# ===================================================================================================
# Parte 1 - Construir el modelo de CNN

# Importar las librerías y paquetes
# para inicializar la RN con pesos aleatorios
from keras.models import Sequential
from keras.layers import Conv2D  # para crear una capa de convolucion 2D
from keras.layers import MaxPooling2D  # para hacer la capa de Max Pooling
from keras.layers import Flatten  # para hacer el Flattening
# sirve para crear capas completamente conectadas en una RN
from keras.layers import Dense

# Inicializar la CNN
classifier = Sequential()

# ----------- Paso 1 - Convolución -----------

# Generamos la coleccion de "mapas de caracteristicas", es decir, la "capa de convolucion".
# Podemos elegir el numero de filtros, es decir el no. de mapas de caracteristicas que tendra la capa.
# Con el parametro "kernel_size", se puede elegir el tamaño de los filtros.
# En los parametros se puede modificar el "stride" y el "padding".
# Con "input_shape", se especifica el tamaño de entrada en el que seran leidas las imagenes y los canales de color
# Con "input_shape", se elige la funcion de activacion que se utilizara una vez que se pase la capa de convolucion.
classifier.add(Conv2D(filters=32, kernel_size=(3, 3),
                      input_shape=(64, 64, 3), activation="relu"))

# A veces se suele agregar otras capas de convolucion con 64, luego con 128 y finalmente con 256 mapas de caracteristicas

# ----------- Paso 2 - Max Pooling -----------

# Con el parametro "pool_size", se define las dimensiones de la ventana.
# Se puede modificar el "stride".
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Al elegir dimensiones grandes de la ventana se reduce mas la informacion y se facilita la convergencia del algoritmo,
# aunque se pierde mas informacion. Hay que mantener un balance en esto; 2*2 es una buena opcion.

# Una segunda capa de convolución y max pooling
classifier.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))

classifier.add(MaxPooling2D(pool_size=(2, 2)))

# -----------Paso 3 - Flattening -----------

classifier.add(Flatten())

# ----------- Paso 4 - Full Connection -----------

# El parametro "units", hace referencia a la salida, numero de neuronas que tiene la primera capa oculta,
# se suele poner la media entre los nodos de la capa de entrada y los nodos de salida. Pero en este caso seria un
# numero extremadamente grande, y se eligira 128.
classifier.add(Dense(units=128, activation="relu"))
# Añadimos la capa de salida con una funcion de activacion sigmoide
classifier.add(Dense(units=1, activation="sigmoid"))

"""
Se pueden crean mas de una capa oculta con el siguiente codigo:
no_capas_profundas = 2
for i in range(1:no_capas_profundas+1):
    classifier.add(Dense(units=128, activation="relu"))
"""

# ----------- Compilar la CNN -----------
# El parametro "loss", al algoritmo que se utiliza para encontrar el conjunto optimo de pesos.
# se puede elegir: "gradient descent" o "esthocastic gradient descent"
# El parametro "loss", hace referencia a la funcion de coste a utilizar.
# En "metrics", le damos una lista para evaluar el modelo y buscar que aumenenten estas metricas durante el entrenamiento.

classifier.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# ===================================================================================================
# Parte 2 - Ajustar la CNN a las imágenes para entrenar
from keras.preprocessing.image import ImageDataGenerator

# Limpieza de imagenes para evitar el sobreajuste, utilizando la tecnica "aumento de imagen".

# Definimos el reescalador para los datos de Entrenamiento
# Se hacen transformaciones aleatorias para que no solo se tengan en cuenta las imagenes de partida,
# sino modificaciones de estas, que le permitan a la red neuronal detectar mas detalles.
train_datagen = ImageDataGenerator(
    rescale=1. / 255, # Hacemos que los pixeles tengan valores entre 0 y 1, en vez de entre 0 y 255
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# Definimos el reescalador para los datos de Testing
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Cargamos y le Aplicamos el reescalado de imagenes al dataset de Entrenemiento
training_dataset = train_datagen.flow_from_directory('dataset/training_set',
                                                     target_size=(64, 64), # definimos el tamaño de entrada de las imagenes
                                                     batch_size=32, # cantidad de imagenes que pasaran por la RN antes de actualizar los pesos
                                                     class_mode='binary') 

# Cargamos y le Aplicamos el reescalado de imagenes al dataset de Testing
testing_dataset = test_datagen.flow_from_directory('dataset/test_set',
                                                   target_size=(64, 64), # definimos el tamaño de entrada de las imagenes
                                                   batch_size=32, # cantidad de imagenes que pasaran por la RN antes de actualizar los pesos
                                                   class_mode='binary')

# Ajustamos el modelo al conjunto de Entrenamiento
# Nota: Mientras se va entrenando, se pueder ir probando el rendimiento con el conjunto de Testing
classifier.fit_generator(training_dataset,
                         steps_per_epoch=8000, # no. de muestras a tomar por epoch
                         epochs=25, # no. de epochs
                         validation_data=testing_dataset, # Conjunto de validacion para ir probando al modelo
                         validation_steps=2000) # Total de validaciones a hacer

"""
Mejora en la Prediccion
Para mejorar la prediccion lo que se puede hacer es añadir una nueva capa de convolucion (se añaden mas filtros) y otra de Max Pooling.
Es normal añadir varias capas de estas; algo normal son 2 capas de 32 filtros y una 3ra de64 filtros.
Notas: Se elimina el tamaño de entrada de las imagenes de los parametros de la capa de convolucion.
       Siempre va una capa de convolucion y una de Max Pooling.
Ademas se pueden añadir mas capas ocultas a la red Neuronal.
"""