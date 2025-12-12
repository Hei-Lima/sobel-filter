import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

CAMINHO_DA_IMAGEM = 'exemplo_celulas.jpg' 

def analisar_imagem_separada(caminho):
    if not os.path.exists(caminho):
        print("Imagem não encontrada.")
        return

    img_original = cv2.imread(caminho)
    # Resize simples para garantir performance
    if img_original.shape[1] > 1000:
        fator = 1000 / img_original.shape[1]
        img_original = cv2.resize(img_original, None, fx=fator, fy=fator)

    gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 1. Sobel
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    magnitude = cv2.convertScaleAbs(magnitude)

    # 2. Threshold (Binarização)
    _, bordas_binarias = cv2.threshold(magnitude, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Fechamento (Preenche o interior)
    kernel_fechamento = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    objetos_fechados = cv2.morphologyEx(bordas_binarias, cv2.MORPH_CLOSE, kernel_fechamento)

    # 4. Erosão (Separa objetos grudados)
    # Aumente 'iterations' se elas ainda estiverem grudadas.
    # iterations=2 costuma ser suficiente para "cortar" as pontes.
    kernel_erosao = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    objetos_separados = cv2.erode(objetos_fechados, kernel_erosao, iterations=2)
    
    # Dica: Como a erosão diminui o tamanho da célula, podemos fazer uma 
    # Dilatação logo depois APENAS para recuperar o tamanho, se precisar. 
    # Mas para contagem, só a erosão já resolve.
    # ---------------------------

    # 5. Contagem
    contours, _ = cv2.findContours(objetos_separados, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_resultado = img_original.copy()
    contagem = 0
    areas = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50: continue # Ignora ruído
            
        contagem += 1
        areas.append(area)
        
        # Desenha (Verde = Normal, Vermelho = Grande/Agrupado)
        if len(areas) > 5 and area > np.mean(areas) * 2.0:
            cor = (0, 0, 255)
        else:
            cor = (0, 255, 0)

        cv2.drawContours(img_resultado, [cnt], -1, cor, 2)

    print(f"Total detectado (Separados): {contagem}")

    # Visualização
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("Onde o Sobel detectou bordas (Binarizado)")
    plt.imshow(bordas_binarias, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"Resultado Final ({contagem} células)")
    plt.imshow(cv2.cvtColor(img_resultado, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.show()

analisar_imagem_separada("sample/sangue.png")