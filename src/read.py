import cv2 as cv
import numpy as np
capture = cv.VideoCapture(0)
values = [0, 0, 0, 0, 0]

while capture.isOpened(): 
    isTrue, frame = capture.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    haar_cascade = cv.CascadeClassifier('haar_cascades/haarcascade_frontalface_default.xml')

    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7)

    if(len(values) > 5):
        values.pop(0)
    values.append(len(faces_rect))

    thresh = 0 if (sum(values) / 4 < 0) else 1

    for (x,y,w,h) in faces_rect:
        cv.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), thickness=2)

    print(f'Number of faces detected = {len(faces_rect)}')

    print(thresh)

    cv.imshow('Video', cv.resize(frame, (300,200)))

    if cv.waitKey(20) & 0xFF==ord('x'):
        break

capture.release()
cv.destroyAllWindows()

