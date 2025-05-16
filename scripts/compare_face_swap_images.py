import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def compare_images(img1_path, img2_path, title1='Imagem 1', title2='Imagem 2'):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        print(f"Erro ao carregar imagens: {img1_path}, {img2_path}")
        return
    if img1.shape != img2.shape:
        print(f"Shapes diferentes: {img1.shape} vs {img2.shape}")
        # Redimensiona img2 para o shape de img1
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    diff = cv2.absdiff(img1, img2)
    print(f"Comparando: {img1_path} <-> {img2_path}")
    print(f"Shape: {img1.shape}")
    print(f"Diferença absoluta média: {np.mean(diff):.2f}")
    print(f"Diferença absoluta máxima: {np.max(diff)}")
    print(f"Diferença absoluta mínima: {np.min(diff)}")
    fig, axs = plt.subplots(1,3, figsize=(15,5))
    axs[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axs[0].set_title(title1)
    axs[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axs[1].set_title(title2)
    axs[2].imshow(diff)
    axs[2].set_title('Diferença')
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python compare_face_swap_images.py <imagem1.jpg> <imagem2.jpg>")
        sys.exit(1)
    img1_path = sys.argv[1]
    img2_path = sys.argv[2]
    compare_images(img1_path, img2_path, title1=os.path.basename(img1_path), title2=os.path.basename(img2_path)) 