# Seam Carving para vídeo aplicando o algoritmo frame a frame
import os
import cv2
import argparse
import seam_carving as sc
import numpy as np

PAD_WIDTH = 4

# Argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", default="video/roadski.mov",	help="Arquivo de vídeo a ser redimensionada")
ap.add_argument("-x", "--x_scale", type=int,default="80", help="Percentual de redimensionamento em x (0 a Infinito)")
ap.add_argument("-y", "--y_scale", type=int,default="100", help="Percentual de redimensionamento em y (0 a Infinito)")
ap.add_argument("-s", "--save", default="video.avi", help="Lugar para salvar o arquivo")
ap.add_argument("-f", "--frames", default=False, help="Salvar frames? Serão salvos em ./frames")
args = vars(ap.parse_args())


## objetos para ler vídeo
vidcap = cv2.VideoCapture(args["video"])
success,image = vidcap.read()
count = 0

if not success:
    print("Não foi possível ler o vídeo")

images = [] # array que armazenará os frames

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
n_height = int((args["y_scale"]/100) * height)

## objetos de escrita de video
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
video = cv2.VideoWriter(args["save"], fourcc,float(15), (n_width, n_height))
##

print("\nCriando vídeo")

re_imgs = []
for i in range(len(images)):
    print("Seam Carving em frame ", i)

    img_seam = sc.resize(np.array(images[i]), (n_width, n_height), energy_mode="forward") #fazendo seam carving frame a frame
    
    re_imgs.append(img_seam)

    print("Escrevendo frame ", i)
    video.write(img_seam)
    if args["frames"]:
        os.path.isdir("frames") or os.makedirs("frames")
        cv2.imwrite("frames/frame%d.jpg" % i, img_seam)

video.release()
print("\nFeito")
