import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt

def draw_circles(frontal_result, facemask):
    for (x,y,w,h) in frontal_result:
        center = (int(x + w / 2), int(y + h / 2))
        radius = int(w / 2)
        color = (255,0,0)
        cv2.circle(original, center, radius, color, 1)
        cv2.circle(facemask, center, radius, (255,255,255), -1)

def get_objects(gray_img, model_path, scale, neightboor, min_size, max_size, flipped = False):
    detector = cv2.CascadeClassifier(model_path)
    results = detector.detectMultiScale(gray_img, scaleFactor=scale,minNeighbors=neightboor,minSize=min_size, maxSize=max_size) # calculando de fato
    if not flipped:
        return results

    flipped_results = []
    for (x,y,w,h) in results:
        nx = width - x - w
        flipped_results.append((nx,y,w,h))
        
    return flipped_results


if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    filename = "./images/kgb.jpg"

img = cv2.imread(filename)
original = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

height, width, channels = img.shape

frontalface_model = "./models/frontalface.xml"
profileface_model = "./models/profileface.xml"

frontal_scale = 1.1
frontal_neightboor = 3
frontal_min_size = (150,150)
frontal_max_size = (500, 500)

profile_scale = 1.1
profile_neightboor = 1
profile_min_size = (50,50)
profile_max_size = (500, 500)

frontal_result = get_faces(gray, frontalface_model, frontal_scale, frontal_neightboor, frontal_min_size, frontal_max_size)
profile_resultr = get_faces(gray, profileface_model, profile_scale, profile_neightboor, profile_min_size, profile_max_size)
profile_resultl = get_faces(cv2.flip(gray,1), profileface_model, profile_scale, profile_neightboor, profile_min_size, profile_max_size, True)


facemask = np.zeros((height, width), np.uint8)


draw_circles(frontal_result, facemask)
draw_circles(profile_resultr, facemask)
draw_circles(profile_resultl, facemask)


blurred_img = img.copy()


dst = cv2.bitwise_and(blurred_img, blurred_img, mask=facemask)
blurred_img = cv2.resize(blurred_img, (int(width / 15),int(height / 15)))
blurred_img = cv2.GaussianBlur(blurred_img, (3,3), cv2.BORDER_CONSTANT)
blurred_img = cv2.resize(blurred_img, (int(width),int(height)),interpolation=cv2.INTER_CUBIC )

img[np.where(facemask == 255)] = blurred_img[np.where(facemask == 255)]


plt.subplot(211),plt.imshow(original),plt.title('Original')
plt.subplot(212),plt.imshow(img),plt.title('Borrada')
plt.xticks([]), plt.yticks([])
plt.show()