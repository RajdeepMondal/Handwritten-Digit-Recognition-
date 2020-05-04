import cv2
import imutils

from dnn_softmax import *
import numpy as np
import pickle

with open("parameters.pkl", "rb") as f:
    parameters = pickle.load(f)
file = "digits.jpg"
img = cv2.imread(file)
img = cv2.resize(img, (700, 200))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    dig = thresh[y:y + h, x:x + w]
    dig = cv2.resize(dig, (28, 28))
    dig = np.pad(dig, ((12, 12),), 'constant', constant_values=(0,))
    dig = cv2.resize(dig, (28, 28))
    dig = np.array(dig)
    dig = dig.flatten()
    dig = dig.reshape(dig.shape[0], 1)
    AL, _ = L_layer_forward(dig, parameters)
    ans3 = np.argmax(AL)
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    img = cv2.putText(img, str(ans3), (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
cv2.imshow("frame", img)
cv2.waitKey(0)
cv2.imwrite("digrec.jpg", img)
