import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

filename = sys.argv[1]
img = cv2.imread(filename)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

low_blue = np.array([170/2, 70,37])
high_blue = np.array([215/2,255,255])

mask = cv2.inRange(hsv, low_blue, high_blue)

blue_pixels = cv2.bitwise_and(img, img, mask = mask)
blue_pixels[:,:,0] = 10 #Matiz do azul para verde

result = img.copy()
result[np.where(mask == 255)] = blue_pixels[np.where(mask == 255)]
#https://stackoverflow.com/questions/51365126/combine-2-images-with-mask

cv2.imshow("orignal",img)
cv2.imshow("pixels azuis",blue_pixels)
cv2.imshow("resultado final", result)

cv2.waitKey(0)
cv2.destroyAllWindows()