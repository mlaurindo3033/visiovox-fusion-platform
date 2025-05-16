import sys, os

# Adiciona src/ ao path para importar o pacote visiovox
# Isso pode não ser necessário se PYTHONPATH for definido externamente,
# mas é uma boa prática para scripts de teste autônomos.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..')) # Assume que o script está na raiz ou em uma subpasta como /scripts
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from visiovox.core.config_manager import ConfigManager
from visiovox.core.resource_manager import ResourceManager
from visiovox.core.logger_setup import setup_logging # Importar para configurar o logger

# Configurar logging para ver saídas detalhadas do ResourceManager e ModelDownloader
setup_logging()

if __name__ == '__main__':
    print("--- Testando download do arcface_w600k_r50 ---")
    cfg = ConfigManager()
    rm = ResourceManager(cfg)
    
    model_name = 'arcface_w600k_r50'
    model_type = 'face_recogniser' # Conforme definido no manifesto
    
    print(f"Tentando carregar o modelo: {model_name} (Tipo: {model_type})")
    try:
        model_path = rm.load_model(model_name, model_type)
        print(f"Caminho retornado por load_model: {model_path}")
        if model_path and os.path.exists(model_path):
            print(f"SUCESSO: Modelo {model_name} baixado/encontrado em: {os.path.abspath(model_path)}")
        elif model_path:
            print(f"AVISO: load_model retornou um caminho ({model_path}), mas ele NÃO EXISTE no sistema de arquivos.")
        else:
            print(f"FALHA: load_model não retornou um caminho para {model_name}.")
            
        # Verificar se o diretório esperado foi criado
        path_from_manifest = cfg.get(f'model_catalog.{model_name}.path_local_cache')
        if path_from_manifest:
            expected_dir = os.path.join(project_root, path_from_manifest.rpartition('/')[0])
            print(f"Verificando existência do diretório esperado: {expected_dir}")
            if os.path.isdir(expected_dir):
                print(f"Diretório {expected_dir} existe.")
                print(f"Conteúdo: {os.listdir(expected_dir)}")
            else:
                print(f"Diretório {expected_dir} NÃO existe.")
        else:
            print(f"Não foi possível obter 'path_local_cache' do manifesto para {model_name} (chave: model_catalog.{model_name}.path_local_cache).")
            
    except Exception as e:
        print(f"ERRO ao tentar carregar o modelo {model_name}: {e}")
        import traceback
        traceback.print_exc()
    
    # --- Adicionar inspeção do modelo aqui se o download foi bem-sucedido ---
    if 'model_path' in locals() and model_path and os.path.exists(model_path):
        print(f"\n--- Iniciando inspeção do modelo ONNX: {model_path} ---")
        try:
            import onnxruntime # Garantir que onnxruntime está importado neste escopo
            session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            
            print("\n--- Entradas do Modelo ---")
            inputs = session.get_inputs()
            if not inputs:
                print("Nenhuma entrada encontrada no modelo.")
            else:
                for i, model_input in enumerate(inputs):
                    print(f"Entrada #{i}:")
                    print(f"  Nome: {model_input.name}")
                    # Usar list comprehension para lidar com None ou str em shape dims
                    shape_str = ', '.join(['batch_size' if dim is None or (isinstance(dim, str) and not dim.isdigit()) else str(dim) for dim in model_input.shape])
                    print(f"  Shape: {model_input.shape} (Formato: {shape_str})")
                    print(f"  Tipo: {model_input.type}")
                    print("-" * 20)
            
            print("\n--- Saídas do Modelo ---")
            outputs = session.get_outputs()
            if not outputs:
                print("Nenhuma saída encontrada no modelo.")
            else:
                for i, model_output in enumerate(outputs):
                    print(f"Saída #{i}:")
                    print(f"  Nome: {model_output.name}")
                    shape_str = ', '.join(['batch_size' if dim is None or (isinstance(dim, str) and not dim.isdigit()) else str(dim) for dim in model_output.shape])
                    print(f"  Shape: {model_output.shape} (Formato: {shape_str})")
                    print(f"  Tipo: {model_output.type}")
                    print("-" * 20)
            print("--- Inspeção do modelo concluída ---")
        except Exception as inspect_e:
            print(f"ERRO durante a inspeção do modelo {model_path}: {inspect_e}")
            if 'traceback' not in locals(): import traceback # Garantir traceback importado
            traceback.print_exc()
    else:
        print(f"\n--- Inspeção do modelo pulada: o modelo {model_name} não foi baixado ou encontrado corretamente. ---")
        
    print("--- Teste de download e inspeção concluído ---") 