import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

filename = sys.argv[1]
img = cv2.imread(filename)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


low_blue = np.array([170/2, 70,37])
high_blue = np.array([215/2,255,255])
low_green = np.array([30/2, 70,37]) #verde quase amarelo
high_green = np.array([150/2,255,255])

mask_blue = cv2.inRange(hsv, low_blue, high_blue)
mask_green = cv2.inRange(hsv, low_green, high_green)


h,s,v = cv2.split(hsv)
h += 130
blue_pixels = cv2.merge((h,s,v))
blue_pixels = cv2.cvtColor(blue_pixels, cv2.COLOR_HSV2BGR)

h,s,v = cv2.split(hsv)
h += 60
green_pixels = cv2.merge((h,s,v))
green_pixels = cv2.cvtColor(green_pixels, cv2.COLOR_HSV2BGR)

result = img.copy()

result[np.where(mask_blue == 255)] = blue_pixels[np.where(mask_blue == 255)] #Atribuindo índices com máscara em comum (cor branca)
result[np.where(mask_green == 255)] = green_pixels[np.where(mask_green == 255)] 

cv2.imshow('original',img)
cv2.imshow('máscara verde',mask_green)
cv2.imshow('máscara azul',mask_blue)
cv2.imshow('resultado',result)
cv2.waitKey(0)
cv2.destroyAllWindows()





plt.show()