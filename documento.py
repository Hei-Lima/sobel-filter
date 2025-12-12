import cv2
import numpy as np
import matplotlib.pyplot as plt

def scanner_com_sobel(caminho_imagem):
    img = cv2.imread(caminho_imagem)
    if img is None: return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    sobel_combined = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    blurred = cv2.GaussianBlur(sobel_combined, (3, 3), 0)

    _, threshold = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    found_doc = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            found_doc = approx
            break

    output = img.copy()
    if found_doc is not None:
        cv2.drawContours(output, [found_doc], -1, (0, 255, 0), 5)

        for point in found_doc:
            cv2.circle(output, tuple(point[0]), 10, (0, 0, 255), -1)

        print("Documento detectado com sucesso via Sobel.")
    else:
        print("Bordas do documento não estão claras o suficiente.")

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Imagem Original")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    plt.subplot(1, 3, 2)
    plt.title("Filtro Sobel (Bordas)")
    plt.imshow(sobel_combined, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title("Detecção Final")
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))

    plt.show()

# Imagens
# scanner_com_sobel('sample/receita.png')
scanner_com_sobel('sample/jogo.png')