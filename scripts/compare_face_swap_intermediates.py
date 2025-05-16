import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def compare_npy(file1, file2, title1='Pipeline 1', title2='Pipeline 2', is_image=True):
    print(f"--- Comparando {title1} vs {title2} ---")
    try:
        arr1 = np.load(file1)
        arr2 = np.load(file2)
    except FileNotFoundError as e:
        print(f"ERRO: Arquivo não encontrado: {e.filename}")
        print("Verifique se os caminhos estão corretos e se os arquivos existem.")
        return

    print(f"Carregando: {file1} <-> {file2}")
    print(f"Shape: {arr1.shape} vs {arr2.shape}")
    print(f"Dtype: {arr1.dtype} vs {arr2.dtype}")
    print(f"Min:   {np.min(arr1):.6f} vs {np.min(arr2):.6f}")
    print(f"Max:   {np.max(arr1):.6f} vs {np.max(arr2):.6f}")
    print(f"Mean:  {np.mean(arr1):.6f} vs {np.mean(arr2):.6f}")

    if arr1.shape != arr2.shape:
        print("Shapes são diferentes, não é possível calcular a diferença direta.")
    else:
        diff = np.abs(arr1 - arr2)
        print(f"Diferença absoluta média: {np.mean(diff):.6f}")
        print(f"Diferença absoluta máxima: {np.max(diff):.6f}")
        print(f"Diferença absoluta mínima: {np.min(diff):.6f}")

    if is_image and arr1.shape == arr2.shape:
        # Para tensores (1,3,H,W) ou (1,H,W,3) ou (H,W,3)
        def to_img_displayable(arr_orig):
            arr = arr_orig.copy() # Evitar modificar o array original
            if arr.ndim == 4: # (B, C, H, W) ou (B, H, W, C)
                arr = arr[0] # Pega o primeiro item do batch
            
            # Se for (C, H, W), transpõe para (H, W, C)
            if arr.shape[0] == 3 and arr.ndim == 3:
                arr = np.transpose(arr, (1, 2, 0))
            
            # Normalização para visualização:
            # Se os valores já estiverem em [0,1] ou [0,255], ótimo.
            # Senão, normaliza para [0,1] para visualização.
            min_val, max_val = arr.min(), arr.max()
            if not (min_val >= 0 and max_val <= 1.0) and not (min_val >= 0 and max_val <= 255.0):
                print(f"Aviso: Normalizando array com range [{min_val:.2f}, {max_val:.2f}] para [0,1] para visualização.")
                arr = (arr - min_val) / (max_val - min_val + 1e-6) # Adiciona epsilon para evitar divisão por zero
            
            # Se for float e max <= 1.0, escala para [0,255]
            if arr.dtype in [np.float32, np.float64] and arr.max() <= 1.0:
                arr = (arr * 255)
            
            # Garante que é uint8
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            return arr

        img1_display = to_img_displayable(arr1)
        img2_display = to_img_displayable(arr2)
        
        # Calcula a diferença visual em uint8
        # Para ter uma diferença visual mais clara, podemos escalar a diferença
        diff_calc = np.abs(arr1.astype(np.float32) - arr2.astype(np.float32))
        diff_img_display = to_img_displayable(diff_calc)

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Comparação Visual: {os.path.basename(file1)} vs {os.path.basename(file2)}", fontsize=16)
        
        axs[0].imshow(img1_display)
        axs[0].set_title(f"{title1}\\n{arr1.shape}, {arr1.dtype}\\nMin:{np.min(arr1):.2f} Max:{np.max(arr1):.2f} Mean:{np.mean(arr1):.2f}")
        
        axs[1].imshow(img2_display)
        axs[1].set_title(f"{title2}\\n{arr2.shape}, {arr2.dtype}\\nMin:{np.min(arr2):.2f} Max:{np.max(arr2):.2f} Mean:{np.mean(arr2):.2f}")
        
        axs[2].imshow(diff_img_display)
        axs[2].set_title(f"Diferença Absoluta\\nMin:{np.min(diff_calc):.2f} Max:{np.max(diff_calc):.2f} Mean:{np.mean(diff_calc):.2f}")
        
        for ax in axs:
            ax.axis('off')
        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Ajusta para o suptitle
        plt.show()
    print("--- Fim da Comparação --- \\n")

if __name__ == "__main__":
    # Caminhos dos arquivos para comparação
    nosso_input_path = "data/output/debug_inswapper128_input.npy"
    visomaster_input_path = r"D:/visiovox-fusion-platform/data/visomaster_npy_intermediarios/visomaster_target_input_singular.npy"
    
    nosso_embedding_path = "data/output/debug_embedding_input.npy"
    visomaster_embedding_path = r"D:/visiovox-fusion-platform/data/visomaster_npy_intermediarios/visomaster_source_embedding_singular.npy"

    print("Comparando INPUTS DO MODELO (TARGET FACE):")
    compare_npy(nosso_input_path, visomaster_input_path, 
                title1="Nosso Pipeline (Input)", title2="VisoMaster (Input)", 
                is_image=True)

    print("\\nComparando EMBEDDINGS DA FONTE:")
    compare_npy(nosso_embedding_path, visomaster_embedding_path, 
                title1="Nosso Pipeline (Embedding)", title2="VisoMaster (Embedding)", 
                is_image=False) # Embeddings não são imagens 