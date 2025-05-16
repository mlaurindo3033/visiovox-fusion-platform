# scripts/inspect_model_io.py
import argparse
import os
import sys

try:
    import onnxruntime
    import numpy as np # Import numpy para referência de tipos, se necessário no futuro
except ImportError:
    print("Por favor, instale a biblioteca onnxruntime: pip install onnxruntime")
    sys.exit(1)

def inspect_onnx_model(model_path: str):
    """
    Carrega um modelo ONNX e imprime informações sobre suas entradas e saídas.

    Args:
        model_path (str): Caminho para o arquivo .onnx do modelo.
    """
    if not os.path.exists(model_path):
        print(f"Erro: Arquivo do modelo não encontrado em '{model_path}'")
        return
    
    if not os.path.isfile(model_path):
        print(f"Erro: O caminho '{model_path}' não é um arquivo.")
        return

    try:
        print(f"Inspecionando o modelo ONNX em: {model_path}\n")
        
        # Tenta carregar o modelo com o provedor CPU para inspeção,
        # pois é mais universal e não depende de setup de GPU.
        # Para modelos que exigem GPU, esta parte pode precisar de ajuste ou
        # os provedores podem ser passados como argumento.
        providers = ['CPUExecutionProvider']
        session = onnxruntime.InferenceSession(model_path, providers=providers)
        
        print("--- Entradas do Modelo ---")
        inputs = session.get_inputs()
        if not inputs:
            print("Nenhuma entrada encontrada no modelo.")
        else:
            for i, model_input in enumerate(inputs):
                print(f"Entrada #{i}:")
                print(f"  Nome: {model_input.name}")
                print(f"  Shape: {model_input.shape} (Formato: {', '.join(['batch_size' if dim is None or (isinstance(dim, str) and not dim.isdigit()) else str(dim) for dim in model_input.shape])})")
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
                print(f"  Shape: {model_output.shape} (Formato: {', '.join(['batch_size' if dim is None or (isinstance(dim, str) and not dim.isdigit()) else str(dim) for dim in model_output.shape])})")
                print(f"  Tipo: {model_output.type}")
                print("-" * 20)

        # Opcional: Listar metadados do modelo, se disponíveis
        try:
            meta = session.get_modelmeta()
            if meta.custom_metadata_map:
                print("\n--- Metadados Customizados do Modelo ---")
                for key, value in meta.custom_metadata_map.items():
                    print(f"  {key}: {value}")
            # print(f"  Graph Name: {meta.graph_name}")
            # print(f"  Producer: {meta.producer_name}")
            # print(f"  Domain: {meta.domain}")
            # print(f"  Version: {meta.version}")
            # print(f"  Description: {meta.description}")
        except Exception as e:
            print(f"\nNão foi possível ler metadados adicionais: {e}")


    except onnxruntime.capi.onnxruntime_pybind11_state.InvalidGraph as e:
        print(f"Erro: O arquivo '{model_path}' não parece ser um modelo ONNX válido ou está corrompido.")
        print(f"Detalhes do erro: {e}")
    except Exception as e:
        print(f"Ocorreu um erro ao tentar carregar ou inspecionar o modelo '{model_path}': {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspeciona as entradas e saídas de um modelo ONNX.")
    parser.add_argument("model_path", type=str, help="Caminho para o arquivo .onnx do modelo.")
    
    args = parser.parse_args()
    
    inspect_onnx_model(args.model_path) 