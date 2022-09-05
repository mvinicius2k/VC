import sys
import cv2
import numpy as np

lineFile = cv2.imread('imagens/line.jpg')
circleFile = cv2.imread('imagens/circle.jpg')

ret, line = cv2.threshold(lineFile, 0,50, cv2.THRESH_BINARY)

cv2.imshow('resultado', cv2.cvtColor(line, cv2.COLOR_GRAY2RGB))



cv2.waitKey(0)
cv2.destroyAllWindows()