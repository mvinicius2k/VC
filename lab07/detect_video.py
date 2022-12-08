# rosto de frente = verde
# olho direito = ciano
# olho esquerdo = azul
# rosto de perfil = vermelho
# boca (sorriso) = rosa

# o vídeo "meu_result.mp4" foi renderizado com resolução de escala 1.3
# o vídeo usado para testar os tempos foi o Inception_Trim_Curto.mp4

import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt
import timeit



QUESTAO = 2 # Questão 1 só calcula rosto (e de perfil); questão 2 tenta pegar olhos e boca





VIDEO_SRC = "./video/Inception_Trim_Curto.mp4" # ou via parametro na linha de comando
FRONTAL_MODEL = "models/frontalface.xml"
LEFTEYE_MODEL = "models/lefteye.xml"
PROFILE_MODEL = "models/profileface.xml"
RIGHTEYE_MODEL = "models/righteye.xml"
SMILE_MODEL = "models/smilemodel.xml"

RESOLUTIONS = [1.5, 1.4, 1.3, 1.1] # resoluções a testar

# Últimas regiões em que foi detectada uma face
last_face_p1 = []
last_face_p2 = []


def detect(rgb_frame, gray_frame, path, scale, neightBoor, size, color, flipped = False):
    global last_face_p1
    global last_face_p2
    detector = cv2.CascadeClassifier(path) # passando o modelo
    results = detector.detectMultiScale(gray_frame, scaleFactor=scale,minNeighbors=neightBoor,minSize=size) # calculando de fato
    

    # Desenhando quadrados
    for (x,y,w,h) in results:
        if(path == FRONTAL_MODEL):
            color = (0,255,0)
        elif(path == PROFILE_MODEL):
            color = (0,0,255)
        elif(path == LEFTEYE_MODEL):
            color = (255,0,0)
        elif(path == RIGHTEYE_MODEL):
            color = (255,255,0)
        
        # o modelo para detectar de perfil só pegava de um lado da face, então espelhar o vídeo funciona para pegar o outro lado
        p1 = (width - x, y) if flipped else (x, y)
        p2 = (width - x - w, y + h) if flipped else (x + w, y + h) 

            
        # se detectando rosto, armazenar nos arrays 
        if(path in [FRONTAL_MODEL, PROFILE_MODEL]):
            last_face_p1.append(np.asarray(p1))
            last_face_p2.append(np.asarray(p2))
        
        # se detectando boca, considerar só quem estiver dentro das regiões dos ultimos rostos
        if path == SMILE_MODEL and last_face_p1 is not None and last_face_p2 is not None:
            
            for i in range(len(last_face_p1)):
                # dist1 =  np.linalg.norm(p1 - last_face_p1[i])
                # dist2 = np.linalg.norm(p2 - last_face_p2[i])
                # if dist1 < 60 or dist2 < 60:
                #     cv2.rectangle(rgb_frame,p1,p2,color,2)
                #     continue
                xinside = p1[0] > last_face_p1[i][0] and p1[0] < last_face_p2[i][0]
                yinside = p1[1] > last_face_p1[i][1] and p1[1] < last_face_p2[i][1]
                if xinside and yinside and last_face_p2[i][1]:
                    cv2.rectangle(rgb_frame,p1,p2,color,2)

        else:
            cv2.rectangle(rgb_frame,p1,p2,color,2)



# Abre a imagem
if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    filename = VIDEO_SRC

vidcap = cv2.VideoCapture(filename)
success,image = vidcap.read()
count = 0

if not success:
    print("Não foi possível ler o vídeo")

rgb_frames = [] # array que armazenará os frames
gray_frames = [] # array que armazenará os frames

# lendo vídeo e armazenando os frames
while success:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rgb = image
    gray_frames.append(gray)
    rgb_frames.append(rgb)
    success,image = vidcap.read()
    print("Iterando frame ", count)
    count += 1

print("\nFrames lidos")

if len(gray_frames) == 0 or len(rgb_frames) == 0:
    print("Não há imagens")
    exit()


# Medidas
height, width, channels = rgb_frames[0].shape

#Detecção
for r in RESOLUTIONS:
    r_rgb_frames = [f.copy() for f in rgb_frames]
    
    start = timeit.default_timer() #iniciando cronometro

    for i in range(len(gray_frames)):
        print("Processando frame ", i, "/", len(gray_frames))
        
        try:
            detect(r_rgb_frames[i], gray_frames[i], FRONTAL_MODEL, r, 5, (75,75), (0,255,0))
            detect(r_rgb_frames[i], gray_frames[i], PROFILE_MODEL, r, 3, (75,75), (255,0,0))
            detect(r_rgb_frames[i], cv2.flip(gray_frames[i],1), PROFILE_MODEL, r, 3, (75,75), (255,0,0), True) # o modelo só detecta um lado do perfil
            if QUESTAO == 2:
                detect(r_rgb_frames[i], gray_frames[i], LEFTEYE_MODEL, r, 1, (2,10), (0,0,255))
                detect(r_rgb_frames[i], gray_frames[i], RIGHTEYE_MODEL, r, 1, (2,10), (0,255,255))
                detect(r_rgb_frames[i], gray_frames[i], SMILE_MODEL, r, 5, (5,20), (255,0,255))

        except:
            print("Erro ao processar")
        

        # limpando arrays para o proximo frame
        last_face_p1.clear()
        last_face_p2.clear()

    stop = timeit.default_timer()
    timecount = stop - start

    print("Algoritmo executado em ", timecount)

    ## objetos de escrita de video e arquivo
    with open(f"tempos questao {QUESTAO}.txt", 'a+') as f:
        f.write(f"Algoritmo executado em {timecount}s, com resolução de escala reduzida em {r}\n")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    videoname = f"resultado questao {QUESTAO} com escala {r}.mp4"
    video = cv2.VideoWriter(videoname, fourcc,float(24),(width, height))
    ##

    print("\nCriando vídeo")
    try:
        for i in range(len(r_rgb_frames)):
            print("Escrevendo frame ", i)
            video.write(r_rgb_frames[i])

        video.release()
    except:
        print("Erro ao renderizar vídeo")
        video.release()



print("Feito")