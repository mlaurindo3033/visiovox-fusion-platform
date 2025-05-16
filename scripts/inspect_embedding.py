import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def inspect_embedding(npy_path):
    if not os.path.exists(npy_path):
        print(f"Arquivo não encontrado: {npy_path}")
        return
    emb = np.load(npy_path)
    print(f"Embedding: shape={emb.shape}, dtype={emb.dtype}, min={emb.min():.4f}, max={emb.max():.4f}, mean={emb.mean():.4f}")
    print(f"Primeiros 10 valores: {emb.flatten()[:10]}")
    plt.hist(emb.flatten(), bins=50, color='blue', alpha=0.7)
    plt.title('Histograma dos valores do embedding')
    plt.xlabel('Valor')
    plt.ylabel('Frequência')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python inspect_embedding.py <embedding.npy>")
        sys.exit(1)
    inspect_embedding(sys.argv[1]) 