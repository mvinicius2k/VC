#pip install seam-carving
#https://pypi.org/project/seam-carving/

from enum import Enum
import numpy as np
import cv2
import sys
import argparse
import seam_carving
from DetectParams import DetectParams

class EnergyMode(Enum):
    FOWARD = "forward"
    BACKWARD = "backward"
class Order(Enum):
    HEIGHT_FIRDT = "height-first"
    WIDTH_FIRST = "width-first"

current_color = 0
colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]

def next_color() -> tuple[int,int,int]:
    new_color = colors.pop(0)
    colors.append(new_color)
    return new_color

def draw_mask(mask: cv2.Mat, result: tuple[int,int,int,int]):
    for (x,y,w,h) in result:
        cv2.rectangle(mask,(x,y),(x+w,y+h),(255,255,255),-1)

def draw_rects(img: cv2.Mat, result: tuple[int,int,int,int], color = (255,0,0)):
    for (x,y,w,h) in result:
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)

def get_objects(gray_img: cv2.Mat, model_path: str, params: DetectParams, flipped = False) -> list[tuple[int,int,int,int]]:
    detector = cv2.CascadeClassifier(model_path)
    results = detector.detectMultiScale(
        gray_img,
         scaleFactor=params.scale,
         minNeighbors=params.neightboor,
         minSize=params.min_size, 
         maxSize=params.max_size) # calculando de fato
    
    if not flipped:
        return results

    flipped_results = list[tuple[int,int,int,int]]
    for (x,y,w,h) in results:
        nx = width - x - w
        flipped_results.append((nx,y,w,h))
        
    return flipped_results


def apply_seam(img: cv2.Mat, height: int, width: int, mask: cv2.Mat):
    new_size = (int(width), int(height))
    img_seam = seam_carving.resize(
    img, new_size,
    energy_mode='forward',   
    order='width-first',
    keep_mask=mask)
        
    return img_seam


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,	help="Arquivo de imagem a ser redimensionada")
ap.add_argument("-nw", "--width", type=int,required=True, help="Nova largura em pixels")
ap.add_argument("-nh", "--height", type=int,required=True, help="Nova altura em pixels")
ap.add_argument("-d", "--draw",action='store_true', help="Salva uma cópia mostrando a detecção")
ap.add_argument("-s", "--save",type=str, default="output.jpg", help="Nome do arquivo a salvar")

args = vars(ap.parse_args())

filename = args["image"]


img = cv2.imread(filename)

if not img.any():
    print(f"Não foi possível abrir a imagem {filename}")

original = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


height, width, channels = img.shape

fullbody_model = "./models/fullbody.xml"
upperbody_model = "./models/upperbody.xml"
lowerbody_model = "./models/lowerbody.xml"

person_params = DetectParams(1.05, 1, (50,50), (1000,1000))
lowerperson_params = DetectParams(1.05, 3, (30,30), (1000,1000))
upperperson_params = DetectParams(1.05, 3, (30,30), (1000,1000))

try:
    body_result = get_objects(gray, fullbody_model, person_params)
    lowerbody_result = get_objects(gray, upperbody_model, lowerperson_params)
    upperbody_result = get_objects(gray, lowerbody_model, upperperson_params)
except:
    print("Erro ao detectar as pessoas. Verifique os parâmetros DetectParams")
    exit()

mask = np.zeros_like(gray)

results = [body_result, lowerbody_result, upperbody_result]

draw_detect = args["draw"]


detection_img = img.copy()
for result in results:
    if draw_detect:
        draw_rects(detection_img, result, next_color())
        
    draw_mask(mask, result)

new_width = args["width"]
new_height = args["height"]

try:
    img_seam = apply_seam(img, new_height, new_width, mask)
except:
    print("Erro ao aplicar redimensionamento")
    if(new_height <= 0 or new_width <= 0):
        print(f"Novas dimensões ({new_width}, {new_height}) inválidas")

savename = args["save"]
try:
    cv2.imwrite(savename, img_seam)
    if(draw_detect):
        cv2.imwrite(f"detection_{savename}", detection_img)
except Exception as e:
    print("Não foi possível salvar a imagem. Erro em imwrite")
    print(e)
