# Import libraries
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import cv2
import os
import time
from pygame import mixer
mixer.init()
sound = mixer.Sound('alarm.wav')
from keras.models import load_model

# load the model to get the co-ordinates of face
face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')

# load the model to get the co-ordinates of eyes
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

# Load the CNN trained classification model
model = load_model('Drowsiness_CNN_Model_tf1.h5')

path = os.getcwd()  # Return a unicode string representing the current working directory.

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score = 0
thick = 2
label = ['Closed', 'Open']

cap = cv2.VideoCapture(0)

while (True):

    _, frame = cap.read()
    height, width = frame.shape[:2]

    # convert color image into grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Get the co-ordinates of face
    face_dim = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))

    # Get the co-ordinates of eye
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    for (x, y, w, h) in face_dim:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (230, 230, 230), 1)

    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y + h, x:x + w
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (150, 150))
        l_eye = l_eye / 255
        l_eye = l_eye.reshape(150, 150, -1)
        l_eye = np.expand_dims(l_eye, axis=0)

        l_pred = model.predict_classes(l_eye)

        if l_pred[0] == 0:
            label = 'Closed'
        if l_pred[0] == 1:
            label = 'Open'

        break

    for (x, y, w, h) in right_eye:
        r_eye = frame[x:x + w, y:y + h]
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (150, 150))
        r_eye = r_eye / 255
        r_eye = r_eye.reshape(150, 150, -1)
        r_eye = np.expand_dims(r_eye, axis=0)

        r_pred = model.predict_classes(r_eye)

        if r_pred[0] == 0:
            label = 'Closed'
        if r_pred[0] == 1:
            label = 'Open'
        break

    # If both the eyes are closed
    if (l_pred[0] == 0 and r_pred[0] == 0):
        score = score + 1
        cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    else:
        score = score - 1
        cv2.putText(frame, 'Open', (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if score < 0:
        score = 0
    cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    # If driver is closing the eye for long time, then the alarm will buzz
    if score > 15:
        cv2.imwrite(os.path.join(path, 'image.jpg'), frame)

        try:
            sound.play()

        except:
            pass

        if thick < 16:
            thick = thick + 2
        else:
            thick = thick - 2
            if thick < 2:
                thick = 2
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thick)

    if score > 45:
        score = 45

    # Display the final pic
    cv2.imshow('frame', frame)

    # Set up termination key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()