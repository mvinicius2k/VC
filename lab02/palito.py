import sys
import cv2
import numpy as np


lineFile = cv2.cvtColor(cv2.imread('imagens/line.jpg'), cv2.COLOR_BGR2GRAY)
circleFile = cv2.cvtColor(cv2.imread('imagens/circle.jpg'), cv2.COLOR_BGR2GRAY)

# formas isoladas
line = cv2.inRange(lineFile, 0,20) 
circle = cv2.inRange(circleFile, 0,20)

#posições e medidas
circleCenter = (circleFile.shape[0] / 2 , circleFile.shape[1] / 2)
lineShape = (lineFile.shape[0], lineFile.shape[1])
mainOffset = (np.int16((300/2) - (lineFile.shape[0]/2)), np.int16((300/2) - lineFile.shape[1]/2)) # centro
mainCenter = (np.int16(300/2), np.int16(300/2))

#matrizes de rotação
line90 = cv2.getRotationMatrix2D((mainCenter[0], mainCenter[1]), 90, 1)
line180 = cv2.getRotationMatrix2D((mainCenter[0], mainCenter[1]), 180, 1)
line45 = cv2.getRotationMatrix2D((mainCenter[0], mainCenter[1]), 45, 1)
linen45 = cv2.getRotationMatrix2D((mainCenter[0], mainCenter[1]), -45, 1)

# planos 300x300
main =  np.zeros((300,300), np.uint8)
main_line =  main.copy()
main_head= main_line.copy()

# inserindo formas no meio dos planos
main_line[mainOffset[0]:mainOffset[0]+lineFile.shape[0], mainOffset[1]:mainOffset[1]+lineFile.shape[1]] = line
main_head[mainOffset[0]:mainOffset[0]+circle.shape[0], mainOffset[1]:mainOffset[1]+circleFile.shape[1]] = circle 

#criando partes do boneco

body = cv2.warpAffine(main_line, line90, main_line.shape)

head = cv2.warpAffine(main_head, np.float32([[1,0,0],[0,1,-60]]), main_head.shape)

l_arm = cv2.warpAffine(main_line, np.float32([[0.75,0, 0], [0, 1,0]]), main_line.shape)
l_arm = cv2.warpAffine(l_arm, np.float32([[1,0,5],[0,1,-30]]),main_line.shape)

r_arm = cv2.warpAffine(main_line, np.float32([[0.75,0, 0], [0, 1,0]]), main_line.shape)
r_arm = cv2.warpAffine(r_arm, np.float32([[1,0,70],[0,1,-30]]),main_line.shape)

l_leg = cv2.warpAffine(main_line, line45,main_line.shape)
l_leg = cv2.warpAffine(l_leg, np.float32([[1,0,-30],[0,1,70]]),main_line.shape)

r_leg = cv2.warpAffine(main_line, linen45,main_line.shape)
r_leg = cv2.warpAffine(r_leg, np.float32([[1,0,28],[0,1,70]]),main_line.shape)

#merclando as partes com or

main = cv2.bitwise_or(main, body)
main = cv2.bitwise_or(main, l_arm)
main = cv2.bitwise_or(main, r_arm)
main = cv2.bitwise_or(main, l_leg)
main = cv2.bitwise_or(main, r_leg)
main = cv2.bitwise_or(main, head)



main = cv2.bitwise_not(main)
cv2.imshow('resultado', main)



cv2.waitKey(0)
cv2.destroyAllWindows()