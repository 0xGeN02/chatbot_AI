import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer() # stemmizar: reducir una palabra a su raiz
import numpy
import tflearn
import tensorflow
import json
import random
import pickle

with open("./contenido.json") as file:
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
print("Palabras:",palabras)# palabras que se van a utilizar
print("")
print("auxX:",auxX)# palabras tokenizadas
print("")
print("auxY",auxY)# tags
print("")
print("tags",tags)# tags que se van a utilizar y no repetidos
print("")
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
print("Entrenamiento:",entrenamiento)
print("")
print("Salida",salida)
print("")

