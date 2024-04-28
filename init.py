import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # habilitar la aceleracion de hardware para la red neuronal con tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # minimizar los mensajes de advertencia de tensorflow
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer() # stemmizar: reducir una palabra a su raiz
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import random

with open("./contenido.json", encoding='utf-8') as file: # abrir el archivo json en formato utf-8
    data = json.load(file)

palabras = []
tags = []

auxX=[]
auxY=[]

for content in data["contenido"]:
    for pattern in content["patrones"]:
        auxPalabra = nltk.word_tokenize(pattern) # tokenizar las palabras
        palabras.extend(auxPalabra) # agregar a la lista de palabras tokenizadas la palabra tokenizada
        auxX.append(auxPalabra) # agregar a la lista de palabras tokenizadas
        auxY.append(content["tag"]) # agregar a la lista de tags
        
        if content["tag"] not in tags: # si el tag no esta en la lista de tags agregarlo
            tags.append(content["tag"])

palabras = [stemmer.stem(w.lower()) for w in palabras if w != "?"] # stemmizar las palabras y pasarlas a minusculas
palabras = sorted(list(set(palabras))) # ordenar y eliminar duplicados, se usa ser para que no se repitan las palabras
tags = sorted(tags) # ordenar los tags

entrenamiento = []
salida = []
emptyRow = [0 for _ in range(len(tags))] # crear una lista de 0 con la longitud de los tags

for x, document in enumerate(auxX):
    box = []
    auxPalabra = [stemmer.stem(w.lower()) for w in document]
    for w in palabras:
        if w in auxPalabra: # si la palabra esta en la lista de palabras tokenizadas agregar 1 si no 0
            box.append(1)
        else:
            box.append(0)
    outputRow = emptyRow[:]
    outputRow[tags.index(auxY[x])] = 1 # si el tag esta en la lista de tags agregar 1
    entrenamiento.append(box)
    salida.append(outputRow)

entrenamiento = np.array(entrenamiento)
salida = np.array(salida)

modelo = keras.models.Sequential() # crear un modelo secuencial
modelo.add(keras.Input(shape=(len(entrenamiento[0]),))) # capa de entrada con la longitud de la lista de entrenamiento 
modelo.add(keras.layers.Dense(10, activation='relu')) # capa oculta con 10 neuronas y activacion relu, activacion relu: si el valor es menor a 0 se convierte en 0
modelo.add(keras.layers.Dense(len(salida[0]), activation='softmax')) # capa de salida con la longitud de la lista de salida, softmax para obtener la probabilidad de cada tag

modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # regresion de la red categorical_crossentropy:

modelo.fit(entrenamiento, salida, epochs=500, batch_size=8) # entrenar la red con 1000 epocas, 8 muestras por epoca y mostrar metricas
modelo.save("output/modelo.keras") # guardar el modelo

def bot():
    while True:
        entrada = input("Tu: ")
        if entrada.lower() == "salir":
            break
        box = [0 for _ in range(len(palabras))] # crear una lista de 0 con la longitud de las palabras
        procesarEntrada = nltk.word_tokenize(entrada)
        procesarEntrada = [stemmer.stem(palabra.lower()) for palabra in procesarEntrada]
        for palabra in procesarEntrada:
            for i, w in enumerate(palabras):
                if w == palabra:
                    box[i] = 1 # si la palabra esta en la lista de palabras tokenizadas agregar 1
        box = np.array([box])
        prediccion = modelo.predict(box) # predecir la entrada
        prediccionIndex = np.argmax(prediccion) # obtener el indice con mayor probabilidad de prediccion
        tag = tags[prediccionIndex] # obtener el tag con mayor probabilidad de prediccion
        for content in data["contenido"]:
            if content["tag"] == tag:
                print(f"Bot: {random.choice(content['respuestas'])}") # imprimir una respuesta aleatoria del tag encontrado
        print(f"Probabilidad: {prediccion[0][prediccionIndex]}") # imprimir la probabilidad de prediccion
        print("Length entrada:",len(entrenamiento[0]),"Length salida:", len(salida[0]))
bot()