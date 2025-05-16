import sys, os
import numpy as np

# Adiciona src/ ao path para importar o pacote visiovox
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from visiovox.core.config_manager import ConfigManager
from visiovox.core.resource_manager import ResourceManager
from visiovox.modules.scene_analysis.landmark_extractor import LandmarkExtractor

if __name__ == '__main__':
    cfg = ConfigManager()
    rm = ResourceManager(cfg)
    extractor = LandmarkExtractor(cfg, rm)

    model_name = '2dfan4'
    if extractor.load_landmark_model(model_name):
        print(f"SUCCESS: Landmark model '{model_name}' carregado.")
        # Criar imagem dummy e bbox de teste
        test_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        test_bbox = (100, 100, 200, 200)
        landmarks = extractor.extract_landmarks(test_image, test_bbox)
        if landmarks is not None:
            print(f"SUCCESS: Extração de landmarks retornou array com shape {landmarks.shape}")
        else:
            print("ERROR: Falha ao extrair landmarks para bbox de teste.")
    else:
        print(f"ERROR: Falha ao carregar landmark model '{model_name}'.") 