import sys
import os
import numpy as np
import torch
from skimage import transform as trans

# Adicionar o diretório raiz ao path para poder importar módulos da aplicação
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.processors.utils import faceutil

def test_keypoints_format():
    """
    Testa a compatibilidade de diferentes formatos de keypoints 
    com a função de transformação.
    """
    # Obter o template de destino (formato esperado)
    dst = faceutil.get_arcface_template(image_size=112, mode='arcface')
    dst = np.squeeze(dst)
    print(f"Template de destino shape: {dst.shape}")
    
    # Verificar o formato real do template
    if len(dst.shape) == 3 and dst.shape[0] == 5 and dst.shape[1] == 5:
        # O formato é (5, 5, 2) - isso é incomum, vamos verificar o conteúdo
        print("Conteúdo do template:")
        print(dst)
        
        # Vamos pegar a primeira linha para cada um dos 5 pontos
        dst_simple = np.array([dst[i, 0] for i in range(5)])
        print(f"Template simplificado shape: {dst_simple.shape}")
        print("Conteúdo do template simplificado:")
        print(dst_simple)
    else:
        dst_simple = dst

    # Testar diferentes formatos de keypoints
    test_cases = [
        # Caso 1: Array 1D com 10 elementos [x1,y1,x2,y2,...]
        np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        
        # Caso 2: Formato correto 5x2
        np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]),
        
        # Caso 3: Mais de 5 pontos
        np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]),
        
        # Caso 4: Array não-contíguo
        np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]])[::1, ::1],
    ]

    for i, kps in enumerate(test_cases):
        print(f"\n--- Testando caso {i+1} ---")
        print(f"Keypoints originais shape: {kps.shape}, tipo: {type(kps)}")
        
        try:
            # Processar os keypoints para formato compatível
            face_kps_array = np.array(kps)
            if len(face_kps_array.shape) == 1 and face_kps_array.size >= 10:
                face_kps_array = face_kps_array.reshape(-1, 2)
                print(f"Remodelado para: {face_kps_array.shape}")
            
            if len(face_kps_array) != 5:
                print(f"Ajustando número de pontos de {len(face_kps_array)} para 5")
                face_kps_array = face_kps_array[:5]
            
            print(f"Keypoints processados shape: {face_kps_array.shape}")
            
            # Testar estimativa de transformação
            tform = trans.SimilarityTransform()
            
            # Usar o template simplificado se estivermos lidando com o formato 5x5x2
            if len(dst.shape) == 3 and dst.shape[0] == dst.shape[1] == 5:
                tform.estimate(face_kps_array, dst_simple)
            else:
                tform.estimate(face_kps_array, dst)
            
            print(f"✅ Caso {i+1} passou: Transformação estimada com sucesso")
            print(f"   Rotação: {tform.rotation:.4f}, Escala: {tform.scale:.4f}")
            print(f"   Translação: ({tform.translation[0]:.4f}, {tform.translation[1]:.4f})")
            
        except Exception as e:
            print(f"❌ Caso {i+1} falhou: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    print("Testando manipulação de formato de keypoints faciais...")
    test_keypoints_format()
    print("\nTestes concluídos.") 

    # --- TESTE AUTOMATIZADO DO SAM2 ---
    print("\nTestando uso do SAM2 na segmentação facial...")
    from app.processors.face_masks import FaceMasks
    from app.processors.models_processor import ModelsProcessor
    from PIL import Image

    # Caminho da imagem de teste
    img_path = os.path.join(os.path.dirname(__file__), '../testdata/image/ComfyUI_00007_.png')
    img = np.array(Image.open(img_path).convert('RGB'))

    # Parâmetros para usar SAM2
    parametros = {
        'FaceSegmentationMethodSelection': 'SAM 2',
        'SAM2ModelSelectionSelection': 'Rápido (Tiny)'
    }

    # Instanciar ModelsProcessor e FaceMasks
    models_processor = ModelsProcessor(main_window=None, device='cpu')
    face_masks = FaceMasks(models_processor)

    # Rodar segmentação
    mask = face_masks.apply_face_parser(img, parametros)

    # Verificar se o SAM2 foi usado
    if face_masks.foi_usado_sam2():
        print("✅ SAM2 foi utilizado corretamente na segmentação facial!")
    else:
        print("❌ SAM2 NÃO foi utilizado! Verifique a integração.") 