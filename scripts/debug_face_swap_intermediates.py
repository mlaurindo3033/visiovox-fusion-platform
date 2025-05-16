import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def show_image(img, title):
    if img is None:
        print(f"{title}: imagem não encontrada")
        return
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

def show_npy_tensor(npy_path, title, as_image=True):
    if not os.path.exists(npy_path):
        print(f"{title}: arquivo não encontrado: {npy_path}")
        return
    arr = np.load(npy_path)
    print(f"{title}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min():.3f}, max={arr.max():.3f}, mean={arr.mean():.3f}")
    if as_image:
        # Tenta converter para imagem
        if arr.ndim == 4:
            arr = arr[0]
        if arr.shape[0] == 3:
            arr = np.transpose(arr, (1,2,0))
        arr = np.clip(arr, 0, 1) if arr.dtype in [np.float32, np.float64] else arr
        if arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
        show_image(arr, title)

def main():
    base = 'data/output/'
    arquivos = [
        ('debug_aligned_target_face.jpg', True),
        ('debug_inswapper128_input.npy', False),
        ('debug_inswapper128_output.npy', False),
        ('debug_swapped_face_model_output.jpg', True),
        ('debug_warped_swapped_face_full_size.jpg', True),
        ('debug_blending_mask.jpg', True),
        ('debug_warped_mask.jpg', True),
        ('debug_final_blended_result.jpg', True),
        ('full_face_swap_result.jpg', True),
    ]
    for fname, is_img in arquivos:
        path = os.path.join(base, fname)
        if fname.endswith('.npy'):
            show_npy_tensor(path, fname, as_image=True)
        else:
            if os.path.exists(path):
                img = cv2.imread(path)
                print(f"{fname}: shape={img.shape}, dtype={img.dtype}, min={img.min()}, max={img.max()}, mean={img.mean():.2f}")
                show_image(img, fname)
            else:
                print(f"{fname}: arquivo não encontrado")

if __name__ == "__main__":
    main() 