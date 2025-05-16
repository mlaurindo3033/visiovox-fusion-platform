import logging
import os  # Para construir caminhos absolutos, se necessário

# Importar os módulos e funções necessários do nosso pacote visiovox
from visiovox.core.logger_setup import setup_logging
from visiovox.core.config_manager import ConfigManager, ConfigError
from visiovox.core.resource_manager import ResourceManager
from visiovox.core.orchestrator import Orchestrator

def run_main_pipeline():
    """
    Função principal para configurar e executar o pipeline de teste.
    """
    # Configurar o logger para este script principal.
    logger = logging.getLogger("visiovox.main_script")
    logger.info("===================================================")
    logger.info("Iniciando a execução do pipeline VisioVox (teste)...")
    logger.info("===================================================")

    try:
        # 1. Instanciar ConfigManager
        logger.info("Inicializando ConfigManager...")
        config_manager = ConfigManager()
        logger.info("ConfigManager inicializado com sucesso.")

        # 2. Instanciar ResourceManager
        logger.info("Inicializando ResourceManager...")
        resource_manager = ResourceManager(config_manager=config_manager)
        logger.info("ResourceManager inicializado com sucesso.")

        # 3. Instanciar Orchestrator
        logger.info("Inicializando Orchestrator...")
        orchestrator = Orchestrator(config_manager=config_manager, resource_manager=resource_manager)
        logger.info("Orchestrator inicializado com sucesso.")

        # 4. Definir o caminho para a imagem de teste
        relative_image_path = "data/input/sample_test_image.png"
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        absolute_image_path = os.path.join(project_root, relative_image_path)
        
        logger.info(f"Caminho da imagem de teste (relativo): {relative_image_path}")
        logger.info(f"Caminho da imagem de teste (absoluto calculado): {absolute_image_path}")

        if not os.path.exists(absolute_image_path):
            logger.error(f"Arquivo de imagem de teste NÃO ENCONTRADO em: {absolute_image_path}")
            logger.error("Por favor, coloque uma imagem de teste (ex: .jpg ou .png com rostos) neste local.")
            logger.error("Verifique também se o nome do arquivo corresponde ao definido em 'models/face_detection/yolov8n-face.onnx' (se aplicável para o modelo).")
            logger.error("O pipeline de teste não pode continuar sem a imagem.")
            return

        # Verificar se o modelo ONNX existe no caminho esperado
        default_model_name = config_manager.get("default_models.face_detector", "yolo_default_onnx")
        model_config_key = f"models.face_detection.{default_model_name}.path"
        relative_model_path = config_manager.get(model_config_key)
        if relative_model_path:
            absolute_model_path = os.path.join(project_root, relative_model_path)
            if not os.path.exists(absolute_model_path):
                logger.warning(f"Arquivo do modelo ONNX ({default_model_name}) NÃO ENCONTRADO em: {absolute_model_path}")
                logger.warning(f"Verifique se o caminho '{relative_model_path}' em 'default_config.yaml' (chave: {model_config_key}) está correto e o arquivo existe.")
                logger.warning("A detecção de faces pode falhar se o modelo não for encontrado pelo FaceDetector.")
            else:
                logger.info(f"Modelo ONNX '{default_model_name}' encontrado em: {absolute_model_path}")
        else:
            logger.warning(f"Caminho para o modelo ONNX '{default_model_name}' não encontrado na configuração (chave: {model_config_key}).")

        # 5. Chamar o método de processamento do Orchestrator
        logger.info(f"Invocando orchestrator.process_static_image com: {absolute_image_path}")
        
        # Orchestrator.process_static_image agora não retorna um booleano de sucesso.
        # O sucesso ou falha é determinado pelos logs do Orchestrator.
        orchestrator.process_static_image(absolute_image_path)
        # Adicionar uma mensagem genérica de conclusão, já que os detalhes estarão nos logs do orchestrator.
        logger.info("Orchestrator.process_static_image call completed. Check logs for details.")

    except ConfigError as e:
        logger.critical(f"Erro de Configuração durante a execução principal: {e}", exc_info=True)
    except ImportError as e:
        logger.critical(f"Erro de importação - verifique se todos os módulos estão corretos e as dependências instaladas: {e}", exc_info=True)
    except Exception as e:
        logger.critical(f"Uma exceção não tratada ocorreu durante a execução do main_pipeline: {e}", exc_info=True)
    finally:
        logger.info("===================================================")
        logger.info("Fim da execução do pipeline VisioVox (teste).")
        logger.info("===================================================")


if __name__ == "__main__":
    setup_logging()
    run_main_pipeline() 