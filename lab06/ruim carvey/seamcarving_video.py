# Tentativa de fazer o Seam Carving melhorado para vídeo, usando o tempo de acordo com https://dl.acm.org/doi/abs/10.1145/1360612.1360615
# Demora uma eternidade, não cheguei a ver completar 100%
import os
import cv2
import argparse
import carve as sc
import numpy as np
from PIL import Image


PAD_WIDTH = 4

# cria as imagens no eixo t, que representam as mudanças de um pixels pelo tempo
def create_t_image(images: list, new_width) -> np.ndarray:
    t_height = int(len(images))
    t_width = new_width
    temporal_imgs = np.zeros((height,t_height,t_width,3)) 
    print("Gerando imagem cubo")
    for t in range(height):
        for h in range(t_height):
            for w in range(t_width):
                temporal_imgs[t][h][w][:] = images[h][t][w][:] #mapeando de vídeo para a imagem cubo
        print("Camada ", t, " de ", height)
    return temporal_imgs

# Argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", default="../video/roadski.mov",	help="Arquivo de vídeo a ser redimensionada")
ap.add_argument("-x", "--x_scale", type=int,default="80", help="Percentual de redimensionamento em x (0 a 100)")
ap.add_argument("-s", "--save", default="video_seam_carving.avi", help="Lugar para salvar o arquivo")
args = vars(ap.parse_args())



vidcap = cv2.VideoCapture(args["video"])
success,image = vidcap.read()
count = 0

if not success:
    print("Não foi possível ler o vídeo")

images = [] #array dos frames originais

while success:
  images.append(image)
  success,image = vidcap.read()
  print("Iterando frame ", count)
  count += 1


print("\nFrames lidos")

if len(images) == 0:
    print("Não há imagens")
    exit()

height, width, channels =  images[0].shape
n_width = int((args["x_scale"]/100) * width)
n_height = int(height)

## objetos de video
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
video = cv2.VideoWriter('seam_video.avi', fourcc,float(15), (n_width, n_height))
##

print("\nCriando vídeo")

re_imgs = images.copy() #array que será usado para armazenar o vídeo redimensionado


# As imagens a serem redimensionadas são as de x (largura) e t (altura). 
for w in range(width  - n_width): #iterando quantos pixels a redimensionar. O redimensionamento em x e y é de pixel em pixel
    cube_imgs = create_t_image(re_imgs, width - w) #obtendo a imagem xyt
    print("Fazendo t seam carving ", w, " de ", width  - n_width)
    print("Obtendo seams")
    img = re_imgs[0]
    #o resize da biblioteca foi modificado para retornar o caminho achado pelo seam carving
    result, seams = sc.resize(np.array(re_imgs[0]), (width - w -1, n_height), energy_mode="forward") #obtendo curva do seam_carving para fazer as mascaras na primeira linha das imagens x t
    print("Aplicando seams em t")
    re_imgs = np.zeros((len(images),height, width - w -1,3))
    for i in range(len(cube_imgs)): #percorrendo o cubo de imagens pelo eixo y
        c_mask = np.zeros((len(images), width - w))
        c_mask[0][:] = ~seams[i][:] #Definindo onde o seam_carvey deve começar por mascara

        result, seams_t = sc.resize(np.array(cube_imgs[i]), (width - w-1, len(images)), energy_mode="forward", keep_mask=c_mask) #redimentionando  em x t 
        for h in range(result.shape[0]):
            for nw in range(result.shape[1]):
                re_imgs[h][i][nw][:] = result[h][nw][:] #mapeando as imagens redimensionadas em x t para vídeo normal (xyt)
        
        print(i," de ", len(cube_imgs))

for i in range(len(re_imgs)):
    print("Escrevendo frame ", i)
    video.write(re_imgs[i])
    if True:
        os.path.isdir("frames") or os.makedirs("frames")
        cv2.imwrite("frames/frame%d.jpg" % i, re_imgs[i])

video.release()

