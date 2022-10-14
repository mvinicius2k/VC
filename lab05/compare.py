from typing import Literal
from cv2 import AKAZE, AKAZE_DESCRIPTOR_KAZE, BRISK
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
import imutils

FILE1 = "imagens/lago1.jpg"
FILE2 = "imagens/lago2.jpg"

SIFT_DESCRIPTOR = "SIFT_DESCRIPTOR"
ORB_DESCRIPTOR = "ORB_DESCRIPTOR"
SURF_DESCRIPTOR = "SURF_DESCRIPTOR" #não achei no cv
BRISK_DESCRIPTOR = "BRISK_DESCRIPTOR"
KAZE_DESCRIPTOR = "KAZE_DESCRIPTOR" 
AKAZE_DESCRIPTOR = "AKAZE_DESCRIPTOR" 



def fusion(image1, keypoints1, des1, image2, keypoints2, des2, ratio = 0.75, proj = 4.0):

  smatches = []

  for m in matches:
    if len(m) == 2 and m[0].distance < m[1].distance * ratio:
      smatches.append((m[0].trainIdx, m[0].queryIdx))

  if len(smatches) <= 4:
    print("poucos smatches. Homography nao foi possível de computar")
    return None


  n_points1 = np.float32([keypoints1[i] for (_, i) in smatches])
  n_points2 = np.float32([keypoints2[i] for (i, _) in smatches])
  M = (H, status) = cv2.findHomography(n_points1, n_points2, cv2.RANSAC,proj)

  if M == None:
    print("Nenhum match")
    return None
  
  result = cv2.warpPerspective(image1, H, (image1.shape[1] + image2.shape[1], image1.shape[0]))
  result[0:image1.shape[0], 0:image2.shape[1]] = image2
  return result
  

def detect_and_compute(img: cv2.Mat, descriptor_name: Literal):
  #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  if descriptor_name == SIFT_DESCRIPTOR:
    descriptor = cv2.SIFT_create()
  elif descriptor_name == ORB_DESCRIPTOR:
    descriptor = cv2.ORB_create()
  elif descriptor_name == BRISK_DESCRIPTOR:
    descriptor = cv2.BRISK_create()
  elif descriptor_name == KAZE_DESCRIPTOR:
    descriptor = cv2.KAZE_create()
  elif descriptor_name == AKAZE_DESCRIPTOR:
    descriptor = cv2.AKAZE_create()
  else:
    print(descriptor_name + " não é suportado")

  (kps, des) = descriptor.detectAndCompute(img, None)
  kps = np.float32([kp.pt for kp in kps])
  return (kps, des)

 



def match(descriptor_name: Literal, crosscheck, des1, des2):
  #bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crosscheck)
  #matches = bf.match(des1,des2)
  #return sorted(matches, key = lambda x:x.distance)
  knn = cv2.DescriptorMatcher_create("BruteForce")
  return knn.knnMatch(des1, des2, 2)
  


img1 = cv2.imread( FILE1,0) 
img2 = cv2.imread( FILE2,0) 



(kp1, des1) = detect_and_compute(img1, ORB_DESCRIPTOR)
(kp2, des2) = detect_and_compute(img2, ORB_DESCRIPTOR)

#points1 = cv2.drawKeypoints(img1, kp1, None)
#points2 = cv2.drawKeypoints(img2, kp2, None)

matches = match(ORB_DESCRIPTOR, False, des1, des2)



#img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

result = fusion(img1, kp1, des1, img2, kp2, des2)

#cv2.imshow("Img1 Keypoints", points1)
#cv2.imshow("Img2 Keypoints", points2)
#cv2.imshow("Matches", img3)
cv2.imshow("Resultado", result)
cv2.waitKey(0)
cv2.destroyAllWindows()