

import sys
import cv2
import numpy as np

im = cv2.imread('imagens/jato.jpg')

img_gamma = im.copy()
c_value = 1.0

b_gamma = 0.001
g_gamma = 1
r_gamma = 4

img_gamma[:,:,0] = c_value * ((im[:,:,0] / 255.0) ** (1.0 / b_gamma)) * 255
img_gamma[:,:,1] = c_value * ((im[:,:,1] / 255.0) ** (1.0 / g_gamma)) * 255
img_gamma[:,:,2] = c_value * ((im[:,:,2] / 255.0) ** (1.0 / r_gamma)) * 255

cv2.imshow('original', im)
cv2.imshow('resultado', img_gamma)



cv2.waitKey(0)
cv2.destroyAllWindows()