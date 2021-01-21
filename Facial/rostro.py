# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 14:20:27 2021

@author: jorge
"""
# -*- coding: latin-1 -*-
"""
    Código 1.2 - Detección facial en un vídeo en tiempo real
    Este programa detecta rostros con una webcam.
 
    Escrito por Glare y Transductor
    www.robologs.net
"""
import cv2
 
#Cargamos nuestro classificador de Haar:
cascada_rostro = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
# Si utilizas otro clasificador o lo tienes guardado en un directorio diferente al de este script python,
# tendrás que cambiar 'haarcascade_frontalface_alt.xml' por el path a tu fichero .xml.
 
#Iniciar la webcam:
webcam = cv2.VideoCapture(0)
# NOTA 1: Si no funciona puedes cambiar el índice 0 por otro, o cambiarlo por la dirección de tu webcam (p.ej. '/dev/video0')
# NOTA 2: también debería funcionar si en vez de una webcam utilizas un fichero de vídeo.
 
#Recordamos al usuario cuál es la tecla para salir:
print("\nRecordatorio: pulsa 'ESC' para cerrar.\n")
 
 
while(1):
 
    #Capturar una imagen con la webcam:
    valido, img = webcam.read()
 
    #Si la imagen es válida (es decir, si se ha capturado correctamente), continuamos:
    if valido:
 
        #Convertir la imagen a gris:
        img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
 
        #Buscamos los rostros:
        coordenadas_rostros = cascada_rostro.detectMultiScale(img_gris, 1.3, 5)
 
 
        #Recorremos el array 'coordenadas_rostros' y dibujamos los rectángulos sobre la imagen original:
        for (x,y,ancho, alto) in coordenadas_rostros:
            cv2.rectangle(img, (x,y), (x+ancho, y+alto), (0,0,255) , 3)
 
 
        #Abrimos una ventana con el resultado:
        cv2.imshow('Output', img)
 
        #Salir con 'ESC':
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            break
 
webcam.release()