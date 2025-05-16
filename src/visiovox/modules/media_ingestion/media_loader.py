import logging
import os
from typing import Optional

import cv2  # OpenCV para carregamento de imagem
import numpy as np

# Importar ConfigManager para o construtor, caso configurações futuras sejam necessárias
from visiovox.core.config_manager import ConfigManager


class MediaLoader:
    """
    Handles loading of various media types, starting with static images.
    """

    def __init__(self, config_manager: ConfigManager):
        """
        Initializes the MediaLoader.

        Args:
            config_manager (ConfigManager): An instance of the ConfigManager.
                                            May be used for future configurations
                                            related to media loading.
        """
        self.config_manager: ConfigManager = config_manager
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.logger.info("MediaLoader initialized.")

    def load_static_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Loads a static image from the specified file path.

        Args:
            image_path (str): The absolute or relative path to the image file.

        Returns:
            Optional[np.ndarray]: A NumPy array representing the loaded image
                                  (in BGR format, as loaded by OpenCV),
                                  or None if the image cannot be loaded
                                  (e.g., file not found, invalid format).
        """
        self.logger.info(f"Attempting to load static image from: {image_path}")

        if not isinstance(image_path, str) or not image_path:
            self.logger.error("Invalid image path provided: path is empty or not a string.")
            return None

        try:
            # Verificar se o arquivo existe antes de tentar carregar com OpenCV
            # Embora cv2.imread não lance exceção para arquivo não encontrado (retorna None),
            # logar explicitamente a ausência do arquivo pode ser útil.
            if not os.path.exists(image_path):
                self.logger.error(f"Image file not found at path: {image_path}")
                return None
            
            if not os.path.isfile(image_path):
                self.logger.error(f"Path is not a file: {image_path}")
                return None

            image_array = cv2.imread(image_path)

            if image_array is None:
                # cv2.imread retorna None se não puder decodificar a imagem ou o arquivo não for encontrado
                # (embora já tenhamos verificado a existência)
                self.logger.error(f"Failed to load or decode image from path: {image_path}. "
                                  "The file might be corrupted or an unsupported image format.")
                return None
            
            self.logger.info(f"Successfully loaded image from: {image_path}, shape: {image_array.shape}, dtype: {image_array.dtype}")
            return image_array

        except Exception as e:
            # Captura outras exceções inesperadas durante o processo de carregamento.
            self.logger.error(f"An unexpected error occurred while trying to load image {image_path}: {e}", exc_info=True)
            return None

    # Adicionando um alias para compatibilidade com a chamada no Orchestrator
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Alias for load_static_image to match Orchestrator calls."""
        return self.load_static_image(image_path)

# Exemplo de uso (seria chamado pelo Orchestrator ou em testes)
# if __name__ == '__main__':
#     # Configuração básica de logging para testar
#     logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
#    
#     # Mock do ConfigManager, já que MediaLoader o espera no construtor
#     class MockConfigManager:
#         def get(self, key, default=None): return default

#     cfg_mgr = MockConfigManager()
#     loader = MediaLoader(config_manager=cfg_mgr)

#     # Crie um arquivo de imagem em 'sample_image.jpg' ou 'sample_image.png' no mesmo diretório
#     # ou forneça um caminho válido para testar.
#     # Exemplo: Crie um arquivo chamado 'test_image.png'
    
#     # Teste com uma imagem existente
#     # Suponha que você tem 'test_image.png' no diretório do script ou forneça um caminho absoluto.
#     # Para este exemplo, vamos simular caminhos.
#     # Lembre-se que o Orchestrator usará caminhos relativos à raiz do projeto.
    
#     # Crie um diretório 'data/input' na raiz do projeto e coloque uma imagem lá.
#     # Por exemplo: visiovox-fusion-platform/data/input/test_image.png
#     project_root_for_test = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
#     test_image_path = os.path.join(project_root_for_test, "data", "input", "test_image.png") # Altere se necessário

#     # Verifique se a imagem de teste existe ou crie uma dummy para o teste passar
#     if not os.path.exists(test_image_path):
#         print(f"Warning: Test image not found at '{test_image_path}'. Creating a dummy image for test.")
#         dummy_img_dir = os.path.dirname(test_image_path)
#         if not os.path.exists(dummy_img_dir):
#             os.makedirs(dummy_img_dir, exist_ok=True)
#         dummy_array = np.zeros((100, 100, 3), dtype=np.uint8)
#         cv2.imwrite(test_image_path, dummy_array)


#     print(f"\n--- Teste com imagem existente ('{test_image_path}') ---")
#     loaded_image_array = loader.load_static_image(test_image_path)
#     if loaded_image_array is not None:
#         print(f"Imagem carregada com sucesso. Shape: {loaded_image_array.shape}")
#     else:
#         print("Falha ao carregar a imagem.")

#     # Teste com caminho de imagem não existente
#     print("\n--- Teste com imagem não existente ---")
#     non_existent_path = os.path.join(project_root_for_test, "data", "input", "non_existent_image.jpg")
#     loaded_image_array_ne = loader.load_static_image(non_existent_path)
#     if loaded_image_array_ne is not None:
#         print(f"Imagem carregada (inesperado). Shape: {loaded_image_array_ne.shape}")
#     else:
#         print("Falha ao carregar a imagem (esperado).")

#     # Teste com caminho inválido (e.g., None ou diretório)
#     print("\n--- Teste com caminho None ---")
#     loader.load_static_image(None)
    
#     print("\n--- Teste com caminho sendo um diretório ---")
#     loader.load_static_image(os.path.dirname(test_image_path)) 