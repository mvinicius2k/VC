#questão 1
import numpy as np
import cv2
from matplotlib import pyplot as plt

filename = "imagens/placas-transito.jpg"
img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img,  cv2.COLOR_RGB2GRAY)

#shapes
width  = img.shape[1]
height = img.shape[0]

output = img.copy()
placas_circulares = np.ones((height, width,3), np.uint8)


circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,20 ,minRadius=50,maxRadius=100)


if circles is not None: # Há circulos encontrados
  circles = np.round(circles[0, :]).astype("int")
  for (x, y, r) in circles:
    placas_circulares[y-r:y+r,x-r:x+r,:] = img[y-r:y+r,x-r:x+r,:] #pegando círculos e escrevendo na imagem para placas circulares



plt.subplot(311),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(312),plt.imshow(output),plt.title('Edges')
plt.xticks([]), plt.yticks([])
plt.subplot(313),plt.imshow(placas_circulares),plt.title('Circulares')
plt.xticks([]), plt.yticks([])

plt.show()