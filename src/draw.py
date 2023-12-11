import cv2 as cv
import numpy as np

blank = np.zeros((500,500,3), dtype='uint8')
# changing bg color
blank[:] = 255,255,255
cv.rectangle(blank, (0,0), (blank.shape[1]//2, blank.shape[0]), (0,0,0), thickness=cv.FILLED)

cv.imshow('blank', blank)

cv.waitKey(0)