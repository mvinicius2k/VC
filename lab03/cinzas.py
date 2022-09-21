# QUESTÃO 2

import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("imagens/imagem.png")


#kernels da formula Y=(0,3×R)+(0,59×G)+(0,11×B), usando o centro da matriz para reduzir cada um dos canais com seus respectivos pesos


kernel_r = np.float32([[0,0,0],
                    [0,0.3,0],
                    [0,0,0]])

kernel_g = np.float32([[0,0,0],
                    [0,0.59,0],
                    [0,0,0]])                    

kernel_b = np.float32([[0,0,0],
                    [0,0.11,0],
                    [0,0,0]])

b,g,r = cv2.split(img)


#aplicando kernels
b = cv2.filter2D(b, -1, kernel_b)
r = cv2.filter2D(r, -1, kernel_r)
g = cv2.filter2D(g, -1, kernel_g)

merged = cv2.merge([b,g,r])

#conversão final para cinza
result = cv2.cvtColor(merged, cv2.COLOR_BGR2GRAY)



cv2.imshow('original', img)
cv2.imshow('colorida', merged)
cv2.imshow('resultado', result)



cv2.waitKey(0)
cv2.destroyAllWindows()