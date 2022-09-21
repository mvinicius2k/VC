#QUESTÃO 3

import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("imagens/imagem.png")


# kernel para deixar a imagem com filtro de sépia
sepia_kernel = np.float32([[0.272, 0.534, 0.131,],
                            [0.349, 0.686, 0.168,],
                            [0.393, 0.769, 0.189]])

# aplicando o kernel com transform
sepia = cv2.transform(img, sepia_kernel)

cv2.imshow('original', img)
cv2.imshow('sépia', sepia)



cv2.waitKey(0)
cv2.destroyAllWindows()