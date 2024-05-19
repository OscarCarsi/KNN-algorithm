import numpy as np #Se importa numpy para realizar operaciones matemáticas
import pandas as pd #Se importa pandas para la manipulación de datos
from sklearn.model_selection import train_test_split #Se importa train_test_split para dividir los datos en entrenamiento y prueba
from collections import Counter #Se importa Counter para contar los elementos de una lista

#Se define la función para calcular la distancia euclidiana, la cual recibe dos arreglos de atributos y retorna la distancia entre ellos
def calculate_distance(attributes1, attributes2 ):
    return np.sqrt(np.sum((attributes1 - attributes2) ** 2)) #Se retorna la raíz cuadrada de la suma de los cuadrados de la resta de los atributos

#Se define la función para el algoritmo KNN, la cual recibe los atributos de entrenamiento, las clases de entrenamiento, los atributos de prueba y el valor de k
def knn(attributes_train, classes_train, attribute, k):
    distances = []	#Se crea una lista vacía para almacenar las distancias
    for i in range(len(attributes_train)): #Se recorren los atributos de entrenamiento
        distance = calculate_distance(attributes_train[i], attribute) #Se calcula la distancia entre los atributos de entrenamiento y los atributos de prueba
        distances.append((distance, classes_train[i])) #Se añade la distancia y la clase a la lista de distancias
    distances.sort(key=lambda x: x[0]) #Se ordenan las distancias de menor a mayor
    neighbors = distances[:k] #Se seleccionan los k vecinos más cercanos
    classes = [neighbor[1] for neighbor in neighbors] #Se obtienen las clases de los vecinos más cercanos
    vote = Counter(classes).most_common(1)[0][0] #Se obtiene la clase más común
    return vote

def main():
    data = pd.read_csv('iris.data', header=None) #Se cargan los datos
    attributes = data.iloc[:, :3].values #Se obtienen los atributos
    classes = data.iloc[:, 4].values #Se obtienen las clases
    attributes_train, attributes_test, classes_train, classes_test = train_test_split(attributes, classes, test_size=50, random_state=42) #Se dividen los datos en entrenamiento y prueba, siendo el tamaño de entrenamiento 100 y el tamaño de prueba el resto
    predictions = [] #Se crea una lista vacía para almacenar las predicciones
    k = 3 #Se define el valor de k
    for attribute in attributes_test: #Se recorren los atributos de prueba
        predictions.append(knn(attributes_train, classes_train, attribute, k)) #Se añade la predicción a la lista de predicciones

    correct_counts = Counter() #Se crea un contador para almacenar las predicciones correctas
    total_counts = Counter() #Se crea un contador para almacenar el total de predicciones

    for prediction, actual in zip(predictions, classes_test): #Se recorren las predicciones y las clases de prueba
        total_counts[actual] += 1 #Se añade 1 al contador de total de predicciones
        if prediction == actual: #Si la predicción es igual a la clase de prueba
            correct_counts[actual] += 1 #Se añade 1 al contador de predicciones correctas

    for class_, total in total_counts.items(): #Se recorren las clases y el total de predicciones
        correct = correct_counts[class_] #Se obtiene el total de predicciones correctas
        print(f"Class {class_}: {correct} out of {total} correct") #Se imprime el total de predicciones correctas y el total de predicciones

    print("Accuracy: ", np.sum(predictions == classes_test) / len(classes_test)) #Se imprime la precisión del modelo
    
if __name__ == "__main__":
    main()  