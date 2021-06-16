import cv2
import numpy as np          # Importação de Bibliotecas
from time import sleep

largura_min = 30  # Largura minima do retangulo
altura_min = 30  # Altura minima do retangulo

offset = 6  # Erro permitido entre pixel

pos_linha = 550  # Posição da linha de contagem

delay = 60  # FPS do vídeo

detec = []         # Função Detec vazia e função carros = 0, para que cada vez que o
carros = 0  # código seja executado, sua contagem seja 0, assim somará a partir da primeira captação


def pega_centro(x, y, w, h):
    x1 = int (w / 2)    # Lado X do objeto
    y1 = int (h / 2)    # Lado Y do objeto
    cx = x + x1         # Largura do objeto
    cy = y + y1         # Altura do objeto
    return cx, cy       # Tupla que contém as coordenadas do centro de um objeto


cap = cv2.VideoCapture ('video/cam-monitoramento.mp4')      # Importa o vídeo de um local
subtracao = cv2.bgsegm.createBackgroundSubtractorMOG()      # Pega o fundo e subtrai do que está se movendo

while True:
    ret, frame1 = cap.read ()      # Pega cada frame do vídeo
    tempo = float (1 / delay)
    sleep (tempo)                  # Dá um delay entre cada processamento
    gray = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)  # Pega o frame e transforma para preto e branco
    blur = cv2.GaussianBlur (gray, (3, 3), 5)   # Faz um blur para tentar remover as imperfeições da imagem
    img_sub = subtracao.apply (blur)                 # Faz a subtração da imagem aplicada no blur
    dilat = cv2.dilate (img_sub, np.ones ((5, 5)))           # "Engrossa" o que sobrou da subtração
    kernel = cv2.getStructuringElement (cv2.MORPH_ELLIPSE, (5, 5))     # Cria uma matriz 5x5, em que seu formato é entre 0 e 1, formando-se assim uma elipse dentro
    dilatada = cv2.morphologyEx (dilat, cv2.MORPH_CLOSE, kernel)    # Tenta preencher todos os "buracos" da imagem
    dilatada = cv2.morphologyEx (dilatada, cv2.MORPH_CLOSE, kernel)
    contorno, h = cv2.findContours (dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)       # Cria um contorno

    cv2.line (frame1, (185, pos_linha), (1050, pos_linha), (255, 0, 0), 2)      # Linha de Contorno
    for (i, c) in enumerate (contorno):
        (x, y, w, h) = cv2.boundingRect (c)                # Definição de altura e largura mínima da captação
        validar_contorno = (w >= largura_min) and (h >= altura_min)         # Validação do Contorno
        if not validar_contorno:
            continue

        cv2.rectangle (frame1, (x, y), (x + w, y + h), (0, 0, 255), 2)      # Desenha um retângulo, quando algo é captado
        centro = pega_centro (x, y, w, h)           # Pega o centro do que foi captado
        detec.append (centro)
        cv2.circle (frame1, centro, 4, (0, 225, 255), -1)       # Desenha um circulo bem no centro, para ajudar na validação

        for (x, y) in detec:
            if y < (pos_linha + offset) and y > (pos_linha - offset):       # Linha para definir a nossa margem de erro
                carros += 1         # Quando detectado somará mais um
                cv2.line (frame1, (25, pos_linha), (700, pos_linha), (0, 127, 255), 3)
                detec.remove ((x, y))       # Caso tenha sido contado mais vezes, será removido
                print ("VEICULO DETECTADO: " + str (carros))       # Imprime na tela, a quantidade de veiculos detectado

    # .putText - Escreve na tela  e .imshow - Abre uma janela com o que foi pedido
    cv2.putText (frame1, "QUANTIDADE DE VEICULOS: " + str (carros), (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow ("MONITORAMENTO", frame1)
    cv2.imshow ("DETECCAO", dilatada)

    if cv2.waitKey (1) == 27:
        break                               # Função para quando apertarmos 'ESC', ele fechar todas as telas

cv2.destroyAllWindows ()
cap.release ()
