# QUESTÃO 1

import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Melhora o ruído de vários pontos (halfotone.png)
def points(img: cv2.Mat) -> cv2.Mat:
  gaussian = cv2.GaussianBlur(img, (21,21),0,0) #Aplica o filtro gaussiano para cobrir o ruído "espalhando" os pixels. Raio 21 deu um "bom" resultado
  
  return gaussian

# Aumenta o contraste da imagem muito acinzentada (pieces.png) ajustando os "níveis"
def less_gray(img: cv2.Mat) -> cv2.Mat:
  inMin = np.array([30], dtype=np.float32) #mínimo valor para preto na entrada
  inMax = np.array([147], dtype=np.float32) #máximo para branco na entrada
  inMiddle = np.array([0.517], dtype=np.float32) # gamma

  # máximos e mínimos para a saída
  outMin = np.array([0], dtype=np.float32) 
  outMax = np.array([255], dtype=np.float32)

  gray = img.copy()

  # aplicando os níveis
  gray = np.clip((gray - inMin) / (inMax - inMin), 0, 255)                            
  gray = (gray ** (1/inMiddle)) * (outMax - outMin) + outMin
  gray = np.clip( gray, 0, 255).astype(np.uint8)
  return gray


# deixar a cor uniforme numa área usando a mediana de cores dessa área reduz esses ruídos de chiados
def noise(img: cv2.Mat) -> cv2.Mat:
  clean = np.zeros([img.shape[0],img.shape[1]])

  for i in range(1, img.shape[0] - 2):
    for j in range(1, img.shape[1] - 2):
      #mediana usando uma área de 5 pixels
      area5x5 = [img[i-2, j-2], img[i-2, j-1], img[i-2, j], img[i-2, j+1], img[i-2, j+2],
                img[i-1, j-2], img[i-1, j-1], img[i-1, j],img[i-1, j+1],img[i-1, j+2],
                img[i, j-2], img[i, j-1], img[i, j],img[i, j+1],img[i, j+2],
                img[i+1, j-2], img[i+1, j-1], img[i+1, j],img[i+1, j+1],img[i+1, j+2],
                img[i+2, j-2], img[i+2, j-1], img[i+2, j],img[i+2, j+1], img[i+2, j+2]]
      area5x5 = sorted(area5x5)
      clean[i,j] = area5x5[13]

  return clean



halftone = cv2.imread("imagens/halftone.png",0)
pieces = cv2.imread("imagens/pieces.png",0)
salt_noise = cv2.imread("imagens/salt_noise.png",0)

gaussian = points(halftone)

plt.subplot(321), plt.imshow(halftone, cmap='gray')
plt.subplot(322), plt.imshow(gaussian, cmap='gray')

gray = less_gray(pieces)

plt.subplot(323), plt.imshow(pieces, cmap='gray')
plt.subplot(324), plt.imshow(gray, cmap='gray')

chip = noise(salt_noise)

plt.subplot(325), plt.imshow(salt_noise, cmap='gray')
plt.subplot(326), plt.imshow(chip, cmap='gray')

plt.show()

