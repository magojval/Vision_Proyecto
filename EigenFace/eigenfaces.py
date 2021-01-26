import numpy as np
import cv2
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import time

def get_images(path):
    # Create a Haarcascade classifier for cropping face image
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)
    # Read all the images along with their tags
    #paths = [os.path.join(path, l) for l in os.listdir(path)]
    dataPath =path
    emotionsList = os.listdir(dataPath)
    # Make a matrix to store all the images and labels
    label = 0
    images = []
    labels = []
    for nameDir in emotionsList:
        emotionsPath = dataPath + '/' + nameDir
        # Read image
        for fileName in os.listdir(emotionsPath):
            image = Image.open(emotionsPath+'/'+fileName).convert('L')
            image = np.array(image, 'uint8')
            # Image label
            #label = int(os.path.split(emotionsPath+'/'+fileName)[1].split(".")[0].replace("subject", ""))
            # To crop the face
            # faces = faceCascade.detectMultiScale(image)
            # for (x, y, w, h) in faces:
            #     image=image[y: y + h, x: x + w]
            #image = cv2.resize(image,(150,150),interpolation= cv2.INTER_CUBIC)
            image = image.flatten()
            images.append(image)
            labels.append(label)
        label = label + 1
    # Convert image and label matrix into Numpy arrays
    images = np.array(images, 'uint8')
    labels = np.array(labels, 'uint8')
    # Return the images list and labels list
    return images, labels


def split_training_and_test(images,labels):
    # Split the total data into training and test data
    images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size = 0.2, random_state = 0)
    return images_train, images_test, labels_train, labels_test


def Eigenface_model(images_train, labels_train):
    # subtract mean face from all faces
    images_train = images_train.T
    mean = images_train.mean(axis=1, keepdims=True)
    images_train = images_train - mean
    # compute covariance matrix
    cov = np.matmul(np.transpose(images_train), images_train)
    # get the eigen values and eigen vectors for X'X
    eigval, eigv = np.linalg.eig(cov)
    # get eigen vectors for XX'
    eigu = np.matmul(images_train, eigv.real)
    # normalize eigen vectors
    ssq = np.sum(eigu ** 2, axis=0)
    ssq = ssq ** (1 / 2)
    eigu = eigu / ssq

    # arrange eigen values and hence eigen vecs in descending order of eigen values
    idx = np.argsort(-eigval.real)
    eigval = eigval[idx].real
    eigu = eigu[:, idx].real

    return eigval, eigu, mean


def Eigenface_test(images_test, labels_test, labels_train, weights, kbest, mean):

    # Normalize the test images
    images_test = images_test.T- mean
    labels_test = labels_test.T

    # calculate test image weights
    testweights = np.matmul(kbest.T, images_test)

    correct=0
    for i in range(0, len(labels_test)):
        # calculate error for each test image
        testweight = np.resize(testweights[:, i], (testweights.shape[0], 1))
        err = (weights - testweight) ** 2
        #stddiv = np.std(weights, axis=0, keepdims=True)
        #err = err / stddiv

        # calculate the sum of square of error
        ssq1 = np.sum(err ** (1/2), axis=0)

        # Find the closest face to the test image
        dist= ssq1.min(axis=0, keepdims=True)
        match=labels_train[ssq1.argmin(axis=0)]

        # print the subject number
        if dist < 50000:
            if labels_test[i] == match:
                correct+=1
                print("subject %d identified correctly as %d with distance %f" %(labels_test[i], match, dist.real))
            else:
                print ("subject %d identified incorrectly as %d with distance %f" %(labels_test[i], match, dist.real))
        else:
            print ("subject face not match in database")
    print("The accuracy of Eigenfaces is %f percent" % (correct*100 / len(labels_test)))
    return

def Eigenface_test_ima(images_test, labels_train, weights, kbest, mean):

    # Normalize the test images
    images_test = images_test.T- mean
    #labels_test = labels_test.T

    # calculate test image weights
    testweights = np.matmul(kbest.T, images_test)



        # calculate error for each test image
    testweight = np.resize(testweights[:, 0], (testweights.shape[0], 1))
    err = (weights - testweight) ** 2
        #stddiv = np.std(weights, axis=0, keepdims=True)
        #err = err / stddiv

        # calculate the sum of square of error
    ssq1 = np.sum(err ** (1/2), axis=0)

        # Find the closest face to the test image
    dist= ssq1.min(axis=0, keepdims=True)
    match=labels_train[ssq1.argmin(axis=0)]

        # print the subject number
    # if dist < 50000:
    #     if labels_test[i] == match:
    #         correct+=1
    #         print("subject %d identified correctly as %d with distance %f" %(labels_test[i], match, dist.real))
    #     else:
    #         print ("subject %d identified incorrectly as %d with distance %f" %(labels_test[i], match, dist.real))
    # else:
    #     print ("subject face not match in database")
    
    return dist, match


def Valores_Eigenface(path):

# Get the images and labels from path
    #path = "D:/jorge/Documents/Maestria/1 Semestre/Vision computacion/DATA"
    images_train, labels_train = get_images(path)
   
    # Perform Eigenface analysis and get Eigenface vectors
    eigval, eigu, mean= Eigenface_model(images_train, labels_train)

    # Get the k best eigen vectors
    sum1 = np.sum(eigval, axis=0)
    k = 0
    for i in range(0, len(labels_train)):
        k += eigval[i] / sum1
        if k > 0.95:
            break
    kbest = eigu[:, 3:i + 3]

    # Get the weights of the of eigenfaces for each input image
    weights = np.matmul(kbest.T, images_train.T- mean)

    return labels_train,weights,kbest, mean




# Main program

if __name__ == "__main__":

    # Get the images and labels from path
    inicio = time.time()
    path = 'D:/jorge/Documents/Maestria/1 Semestre/Vision computacion/FaceDetection-master/Data'
    images_train, labels_train = get_images(path)
    tiempoEntrenamiento = time.time()-inicio
    print("Tiempo de Lectura Entrenamiento : ", tiempoEntrenamiento)
   
    # Split the data into training and test
    inicio = time.time()
    path = "D:/jorge/Documents/Maestria/1 Semestre/Vision computacion/EmoPrueba"
    images_test, labels_test = get_images(path)
    tiempoEntrenamiento = time.time()-inicio
    print("Tiempo de Lectura Test : ", tiempoEntrenamiento)

    inicio = time.time()
    # Perform Eigenface analysis and get Eigenface vectors
    eigval, eigu, mean= Eigenface_model(images_train, labels_train)
    tiempoEntrenamiento = time.time()-inicio
    print("Tiempo de entrenamiento: ", tiempoEntrenamiento)

    # Get the k best eigen vectors
    sum1 = np.sum(eigval, axis=0)
    k = 0
    for i in range(0, len(labels_train)):
        k += eigval[i] / sum1
        if k > 0.95:
            break
    kbest = eigu[:, 3:i + 3]

    # Get the weights of the of eigenfaces for each input image
    weights = np.matmul(kbest.T, images_train.T- mean)

    # Test the Eigenface model on the test images and print the result
    Eigenface_test(images_test, labels_test, labels_train, weights, kbest, mean)