import numpy as np
import cv2
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import time

def get_images(path):
    
    # Leer todas las imágenes junto con sus etiquetas
    dataPath =path
    emotionsList = os.listdir(dataPath)
    # Hacer una matriz para almacenar todas las imágenes y etiquetas.
    label = 0
    images = []
    labels = []
    for nameDir in emotionsList:
        emotionsPath = dataPath + '/' + nameDir
        # Leer imagenes
        for fileName in os.listdir(emotionsPath):
            image = Image.open(emotionsPath+'/'+fileName).convert('L')
            image = np.array(image, 'uint8')
            image = image.flatten()
            images.append(image)
            labels.append(label)
        label = label + 1
    # Conviertir la imagen y la matriz de etiquetas en Numpy arrays
    images = np.array(images, 'uint8')
    labels = np.array(labels, 'uint8')
    # Return de images list y labels list
    return images, labels


def split_training_and_test(images,labels):
    # Dividir los datos en Teste y Train
    images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size = 0.2, random_state = 0)
    return images_train, images_test, labels_train, labels_test


def Eigenface_model(images_train, labels_train):
    # Sustraer la cara media de todas las imagenes
    images_train = images_train.T
    mean = images_train.mean(axis=1, keepdims=True)
    images_train = images_train - mean
    # computar la matriz de covarianza
    cov = np.matmul(np.transpose(images_train), images_train)
    # Obtener los eigen valores y eigen vectores de X'X
    eigval, eigv = np.linalg.eig(cov)
    # Obtener los eigen vectores de XX'
    eigu = np.matmul(images_train, eigv.real)
    # normalizar los eigen vectores
    ssq = np.sum(eigu ** 2, axis=0)
    ssq = ssq ** (1 / 2)
    eigu = eigu / ssq

    # Organizar los eigen valores, en orden descendente de los valores propios
    idx = np.argsort(-eigval.real)
    eigval = eigval[idx].real
    eigu = eigu[:, idx].real

    return eigval, eigu, mean


def Eigenface_test(images_test, labels_test, labels_train, weights, kbest, mean):

    # Noormalizar test images
    images_test = images_test.T- mean
    labels_test = labels_test.T

    # calcular los pesos test image
    testweights = np.matmul(kbest.T, images_test)

    correct=0
    for i in range(0, len(labels_test)):
        # Calcular el error de cada una de las imagenes de test image
        testweight = np.resize(testweights[:, i], (testweights.shape[0], 1))
        err = (weights - testweight) ** 2
        
        # Calcular la suma del cuadrado del error
        ssq1 = np.sum(err ** (1/2), axis=0)

        # Encuentra la cara más cercana en test image
        dist= ssq1.min(axis=0, keepdims=True)
        match=labels_train[ssq1.argmin(axis=0)]

        # Imprimir el numero de la Etiqueta de la emocion
        if dist < 50000:
            if labels_test[i] == match:
                correct+=1
                print("Emocion %d Identificado correctamente como %d con distancia %f" %(labels_test[i], match, dist.real))
            else:
                print ("Emocion %d Identificado INcorrectamente como %d con distancia %f" %(labels_test[i], match, dist.real))
        else:
            print ("Emocion no encontrada en la base de datos")
    print("The accuracy of Eigenfaces is %f percent" % (correct*100 / len(labels_test)))
    return

def Eigenface_test_ima(images_test, labels_train, weights, kbest, mean):

     # Normalizar test images
    images_test = images_test.T- mean
    

     # Calcular el error de cada una de las imagenes de test image
    testweights = np.matmul(kbest.T, images_test)



    # Calcular la suma del cuadrado del error
    testweight = np.resize(testweights[:, 0], (testweights.shape[0], 1))
    err = (weights - testweight) ** 2
        

    # Calcular la suma del cuadrado del error
    ssq1 = np.sum(err ** (1/2), axis=0)

    # Encuentra la cara más cercana en test image
    dist= ssq1.min(axis=0, keepdims=True)
    match=labels_train[ssq1.argmin(axis=0)]

    
    return dist, match


def Valores_Eigenface(path):

# Obtener las imagenes y etiquetas del path
    #path = "D:/jorge/Documents/Maestria/1 Semestre/Vision computacion/DATA"
    images_train, labels_train = get_images(path)
   
    # Obtener los eign vectores y valores de las imagenes
    eigval, eigu, mean= Eigenface_model(images_train, labels_train)

    # Obtener los k mejores eige vectores
    sum1 = np.sum(eigval, axis=0)
    k = 0
    for i in range(0, len(labels_train)):
        k += eigval[i] / sum1
        if k > 0.95:
            break
    kbest = eigu[:, 3:i + 3]

    # Obtener los pesos de los e eigenfaces de cada imagen
    weights = np.matmul(kbest.T, images_train.T- mean)

    return labels_train,weights,kbest, mean




# Main program

if __name__ == "__main__":

    # Obtener las imagenes y etiquetas del path para el entrenamiento
    inicio = time.time()
    path = 'D:/jorge/Documents/Maestria/1 Semestre/Vision computacion/EntrenamientoYO'
    images_train, labels_train = get_images(path)
    tiempoEntrenamiento = time.time()-inicio
    print("Tiempo de Lectura Entrenamiento : ", tiempoEntrenamiento)
   
   # Obtener las imagenes y etiquetas del path para el test
    inicio = time.time()
    path = "D:/jorge/Documents/Maestria/1 Semestre/Vision computacion/EmoTest"
    images_test, labels_test = get_images(path)
    tiempoEntrenamiento = time.time()-inicio
    print("Tiempo de Lectura Test : ", tiempoEntrenamiento)

    inicio = time.time()
    print("Entrenamiento...")
    # Obtener los eign vectores y valores de las imagenes
    eigval, eigu, mean= Eigenface_model(images_train, labels_train)
    tiempoEntrenamiento = time.time()-inicio
    print("Tiempo de entrenamiento: ", tiempoEntrenamiento)

    # Obtener los k mejores eige vectores
    sum1 = np.sum(eigval, axis=0)
    k = 0
    for i in range(0, len(labels_train)):
        k += eigval[i] / sum1
        if k > 0.95:
            break
    kbest = eigu[:, 3:i + 3]

    # Obtener los pesos de los e eigenfaces de cada imagen
    weights = np.matmul(kbest.T, images_train.T- mean)

    # Realizar al modelo de  Eigenface un Test e imprimir los resultados
    Eigenface_test(images_test, labels_test, labels_train, weights, kbest, mean)
    print("Tiempo de entrenamiento: ", tiempoEntrenamiento)