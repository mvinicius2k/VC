from typing import Literal
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt

FILE1 = "imagens/6.jpg"
FILE2 = "imagens/7.jpg"

SIFT_DESCRIPTOR = "SIFT_DESCRIPTOR"
ORB_DESCRIPTOR = "ORB_DESCRIPTOR"
SURF_DESCRIPTOR = "SURF_DESCRIPTOR" #não achei no cv
BRISK_DESCRIPTOR = "BRISK_DESCRIPTOR"
KAZE_DESCRIPTOR = "KAZE_DESCRIPTOR" 
AKAZE_DESCRIPTOR = "AKAZE_DESCRIPTOR" 

matches = []


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
  knn = cv2.DescriptorMatcher_create("BruteForce")
  return knn.knnMatch(des1, des2, 2)
  

def panoramica(method, img1, img2):
    global matches
    (kp1, des1) = detect_and_compute(img1, method)
    (kp2, des2) = detect_and_compute(img2, method)

    matches = match(method, False, des1, des2)

    result = fusion(img1, kp1, des1, img2, kp2, des2)
    return result


img1 = cv2.imread( FILE1) 
img2 = cv2.imread( FILE2) 



result_orb = panoramica(ORB_DESCRIPTOR, img1, img2)
result_brisk = panoramica(BRISK_DESCRIPTOR, img1, img2)
result_sift = panoramica(SIFT_DESCRIPTOR, img1, img2)
result_kaze = panoramica(KAZE_DESCRIPTOR, img1, img2)
result_akaze = panoramica(AKAZE_DESCRIPTOR, img1, img2)



plt.subplot(321),plt.imshow(cv2.cvtColor(result_orb, cv2.COLOR_BGR2RGB)),plt.title('ORB')
plt.xticks([]), plt.yticks([])
plt.subplot(322),plt.imshow(cv2.cvtColor(result_brisk, cv2.COLOR_BGR2RGB)),plt.title('BRISK')
plt.xticks([]), plt.yticks([])
plt.subplot(323),plt.imshow(cv2.cvtColor(result_sift, cv2.COLOR_BGR2RGB)),plt.title('SIFT')
plt.xticks([]), plt.yticks([])
plt.subplot(324),plt.imshow(cv2.cvtColor(result_kaze, cv2.COLOR_BGR2RGB)),plt.title('KAZE')
plt.xticks([]), plt.yticks([])
plt.subplot(325),plt.imshow(cv2.cvtColor(result_kaze, cv2.COLOR_BGR2RGB)),plt.title('AKAZE')
plt.xticks([]), plt.yticks([])

plt.show()
