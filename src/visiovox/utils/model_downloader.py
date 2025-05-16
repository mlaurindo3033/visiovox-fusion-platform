import os
import logging
import hashlib
import requests
from tqdm import tqdm
from typing import Optional

from visiovox.core.config_manager import ConfigManager

# Configurar o logger no nível do módulo
module_logger = logging.getLogger(__name__)

class ModelDownloadManager:
    """
    Gerencia o download e verificação de integridade de modelos ONNX e similares,
    conforme configurado em default_config.yaml.
    """
    def __init__(self, config_manager: ConfigManager):
        """
        Inicializa o ModelDownloadManager.
        
        Args:
            config_manager (ConfigManager): Instância do ConfigManager.
        """
        self.config_manager = config_manager

    @staticmethod
    def download_model(url: str, dest_path: str, expected_sha256: str) -> bool:
        """
        Baixa um modelo de uma URL, salva em dest_path e verifica o hash SHA256.

        Args:
            url (str): URL do modelo a ser baixado.
            dest_path (str): Caminho de destino para salvar o modelo.
            expected_sha256 (str): Hash SHA256 esperado do arquivo.

        Returns:
            bool: True se o download e a verificação forem bem-sucedidos ou se o arquivo
                  correto já existir, False caso contrário.
        """
        # Garante que o diretório de destino exista
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        try:
            if os.path.exists(dest_path):
                module_logger.info(f"Arquivo '{os.path.basename(dest_path)}' já existe em '{os.path.dirname(dest_path)}'. Verificando hash...")
                local_hash = ModelDownloadManager.calculate_sha256(dest_path)
                if local_hash.lower() == expected_sha256.lower():
                    module_logger.info(f"Hash de '{os.path.basename(dest_path)}' corresponde ao esperado. Download não necessário.")
                    return True
                else:
                    module_logger.warning(f"Hash de '{os.path.basename(dest_path)}' ({local_hash[:10]}...) difere do esperado ({expected_sha256[:10]}...). Baixando novamente.")
            
            module_logger.info(f"Iniciando download do modelo de: {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Lança uma exceção para códigos de status ruins (4xx ou 5xx)

            total_size_in_bytes = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte

            progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc=f"Baixando {os.path.basename(dest_path)}")
            with open(dest_path, 'wb') as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
            progress_bar.close()

            if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                module_logger.error(f"Erro no download: tamanho do arquivo esperado ({total_size_in_bytes} bytes) não corresponde ao tamanho baixado ({progress_bar.n} bytes).")
                return False
            
            module_logger.info(f"Download de '{os.path.basename(dest_path)}' concluído. Verificando hash...")
            downloaded_hash = ModelDownloadManager.calculate_sha256(dest_path)

            if downloaded_hash.lower() == expected_sha256.lower():
                module_logger.info(f"Hash de '{os.path.basename(dest_path)}' verificado com sucesso.")
                return True
            else:
                module_logger.error(f"Erro de verificação de hash para '{os.path.basename(dest_path)}'.")
                module_logger.error(f"Esperado: {expected_sha256.lower()}")
                module_logger.error(f"Obtido  : {downloaded_hash.lower()}")
                # Opcional: remover arquivo se o hash não corresponder
                # os.remove(dest_path) 
                return False

        except requests.exceptions.RequestException as e:
            module_logger.error(f"Falha no download do modelo de {url}: {e}")
            if os.path.exists(dest_path): # Se o arquivo foi parcialmente criado, remova-o
                try:
                    os.remove(dest_path)
                except OSError as ose:
                    module_logger.error(f"Não foi possível remover o arquivo parcialmente baixado {dest_path}: {ose}")
            return False
        except Exception as e:
            module_logger.error(f"Uma exceção inesperada ocorreu durante o download/verificação do modelo {os.path.basename(dest_path)}: {e}")
            if os.path.exists(dest_path) and 'downloaded_hash' not in locals(): # Evita remover se o erro foi na verificação de hash
                try:
                    os.remove(dest_path)
                except OSError as ose:
                    module_logger.error(f"Não foi possível remover o arquivo {dest_path} após erro: {ose}")
            return False 

    @staticmethod
    def calculate_sha256(file_path: str, block_size: int = 65536) -> str:
        """Calcula o hash SHA256 de um arquivo."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(block_size):
                sha256.update(chunk)
        return sha256.hexdigest() 