# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 12:46:51 2021

@author: jorge
"""
import cv2
import os
import numpy as np
import time
from eigenfaces import Valores_Eigenface
from eigenfaces import Eigenface_test_ima
from eigenfaces import get_images


dataPath = 'D:/jorge/Documents/Maestria/1 Semestre/Vision computacion/EntrenamientoYO' #Cambia a la ruta donde hayas almacenado Data
imagePaths = os.listdir(dataPath)
print('imagePaths=',imagePaths)

path_Entr='D:/jorge/Documents/Maestria/1 Semestre/Vision computacion/EntrenamientoYO'

print("Entrenando....")
inicio = time.time()
labels_train,weights,kbest, mean= Valores_Eigenface(path_Entr)
print("Fin del entrenamiento")
tiempoEntrenamiento = time.time()-inicio
print("Tiempo de entrenamiento: ", tiempoEntrenamiento)

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
images = []
images.append(0)
images=np.array(images, 'uint8')

while True:

    ret,frame = cap.read()
    if ret == False: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()



    faces = faceClassif.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        images = []
        rostro = auxFrame[y:y+h,x:x+w]
        rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
        rostro=np.array(rostro, 'uint8')
        rostro = rostro.flatten()
        images.append(rostro)
        
        # path = "D:/jorge/Documents/Maestria/1 Semestre/Vision computacion/EmoPrueba"
        # images_test, labels_test = get_images(path)
        images=np.array(images, 'uint8')
        dist, match=Eigenface_test_ima(images, labels_train, weights, kbest, mean)
        
        cv2.putText(frame,str(dist),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)
        
        # dist=5000
        # match=0
        # EigenFaces

        if dist < 50000:
            
            cv2.putText(frame,'{}'.format(imagePaths[match]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)

        else:
            cv2.putText(frame,'No identificado',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)


    cv2.imshow('nFrame',frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()