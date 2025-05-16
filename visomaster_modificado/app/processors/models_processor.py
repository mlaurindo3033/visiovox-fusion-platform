import threading
import os
import subprocess as sp
import gc
import traceback
from typing import Dict, TYPE_CHECKING
from collections import OrderedDict

from packaging import version
import numpy as np
import onnxruntime
import torch
import onnx
from torchvision.transforms import v2
from PySide6 import QtCore
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ModuleNotFoundError:
    print("No TensorRT Found")
    TENSORRT_AVAILABLE = False

from app.processors.utils.engine_builder import onnx_to_trt as onnx2trt
from app.processors.utils.tensorrt_predictor import TensorRTPredictor
from app.processors.face_detectors import FaceDetectors
from app.processors.face_landmark_detectors import FaceLandmarkDetectors
from app.processors.face_masks import FaceMasks
from app.processors.face_restorers import FaceRestorers
from app.processors.face_swappers import FaceSwappers
from app.processors.frame_enhancers import FrameEnhancers
from app.processors.face_editors import FaceEditors
from app.processors.utils.dfm_model import DFMModel
from app.processors.models_data import models_list, arcface_mapping_model_dict, models_trt_list
from app.helpers.miscellaneous import is_file_exists
from app.helpers.downloader import download_file
from app.helpers.profiler import profile_func, measure_time, log_gpu_memory_usage
from app.helpers.memory_manager import memory_manager  # Importar o gerenciador de memória

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

onnxruntime.set_default_logger_severity(4)
onnxruntime.log_verbosity_level = -1
lock = threading.Lock()

class ModelsProcessor(QtCore.QObject):
    processing_complete = QtCore.Signal()
    model_loaded = QtCore.Signal()  # Signal emitted with Onnx InferenceSession
    
    # Novos sinais para progresso de carregamento
    start_loading_signal = QtCore.Signal(str)  # Sinal para iniciar carregamento (nome do modelo)
    progress_signal = QtCore.Signal(int)       # Sinal para atualizar progresso (porcentagem)
    completion_signal = QtCore.Signal(bool)    # Sinal para indicar conclusão (sucesso/falha)

    def __init__(self, main_window: 'MainWindow', device='cuda'):
        super().__init__()
        self.main_window = main_window
        self.provider_name = 'TensorRT'
        self.device = device
        self.model_lock = threading.RLock()  # Reentrant lock for model access
        self.trt_ep_options = {
            # 'trt_max_workspace_size': 3 << 30,  # Dimensione massima dello spazio di lavoro in bytes
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': "tensorrt-engines",
            'trt_timing_cache_enable': True,
            'trt_timing_cache_path': "tensorrt-engines",
            'trt_dump_ep_context_model': True,
            'trt_ep_context_file_path': "tensorrt-engines",
            'trt_layer_norm_fp32_fallback': True,
            'trt_builder_optimization_level': 5,
        }
        self.providers = [
            ('CUDAExecutionProvider'),
            ('CPUExecutionProvider')
        ]       
        self.nThreads = 2
        self.syncvec = torch.empty((1, 1), dtype=torch.float32, device=self.device)

        # Configurações de limites para os caches LRU
        self.onnx_cache_limit = 5  # Número máximo de modelos ONNX em cache
        self.trt_cache_limit = 3   # Número máximo de modelos TensorRT em cache
        self.dfm_cache_limit = 3   # Número máximo de modelos DFM em cache

        # Modelos essenciais que não serão removidos do cache
        self.essential_models = []  # Lista de nomes de modelos essenciais
        
        # Define modelos primários (sempre carregados) e secundários (lazy loading)
        self.primary_models = ['RetinaFace', 'Inswapper128ArcFace']  # Modelos principais sempre carregados
        self.secondary_models = []  # Será preenchido com outros modelos

        with measure_time("Inicialização de ModelsProcessor"):
            # Initialize models and models_path
            self.models: Dict[str, onnxruntime.InferenceSession] = {}
            self.models_path = {}
            self.models_data = {}
            for model_data in models_list:
                model_name, model_path = model_data['model_name'], model_data['local_path']
                self.models[model_name] = None #Model Instance
                self.models_path[model_name] = model_path
                self.models_data[model_name] = {'local_path': model_data['local_path'], 'hash': model_data['hash'], 'url': model_data.get('url')}
                # Adiciona à lista de secundários se não for primário
                if model_name not in self.primary_models:
                    self.secondary_models.append(model_name)

            # Inicializa os caches LRU como OrderedDict
            self.models_cache = OrderedDict()  # Cache LRU para modelos ONNX
            self.dfm_models: Dict[str, DFMModel] = {}
            self.dfm_models_cache = OrderedDict()  # Cache LRU para modelos DFM

            if TENSORRT_AVAILABLE:
                # Initialize models_trt and models_trt_path
                self.models_trt = {}
                self.models_trt_path = {}
                self.models_trt_cache = OrderedDict()  # Cache LRU para modelos TensorRT
                for model_data in models_trt_list:
                    model_name, model_path = model_data['model_name'], model_data['local_path']
                    self.models_trt[model_name] = None #Model Instance
                    self.models_trt_path[model_name] = model_path

            self.face_detectors = FaceDetectors(self)
            self.face_landmark_detectors = FaceLandmarkDetectors(self)
            self.face_masks = FaceMasks(self)
            self.face_restorers = FaceRestorers(self)
            self.face_swappers = FaceSwappers(self)
            self.frame_enhancers = FrameEnhancers(self)
            self.face_editors = FaceEditors(self)

            self.clip_session = []
            self.arcface_dst = np.array( [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
            self.FFHQ_kps = np.array([[ 192.98138, 239.94708 ], [ 318.90277, 240.1936 ], [ 256.63416, 314.01935 ], [ 201.26117, 371.41043 ], [ 313.08905, 371.15118 ] ])
            self.mean_lmk = []
            self.anchors  = []
            self.emap = []
            self.LandmarksSubsetIdxs = [
                0, 1, 4, 5, 6, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39,
                40, 46, 52, 53, 54, 55, 58, 61, 63, 65, 66, 67, 70, 78, 80,
                81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133,
                136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160,
                161, 162, 163, 168, 172, 173, 176, 178, 181, 185, 191, 195, 197, 234, 246,
                249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293, 295,
                296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 332, 334,
                336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382,
                384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454,
                466, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477
            ]

            self.normalize = v2.Normalize(mean = [ 0., 0., 0. ],
                                        std = [ 1/1.0, 1/1.0, 1/1.0 ])
            
            self.lp_mask_crop = self.face_editors.lp_mask_crop
            self.lp_lip_array = self.face_editors.lp_lip_array
            
            # Carrega modelos primários durante a inicialização
            for model_name in self.primary_models:
                if model_name in self.models_path:
                    self.load_model(model_name)

    def _unload_model(self, model_name):
        """Descarrega um modelo ONNX da memória de forma segura"""
        if model_name in self.models_cache:
            try:
                # Verifica se é um modelo essencial
                if model_name in self.primary_models:
                    return False  # Não descarregar modelos essenciais
                    
                # Remove do cache
                model_instance = self.models_cache.pop(model_name)
                # Limpa referências e libera memória
                del model_instance
                gc.collect()
                memory_manager.clear_cuda_cache()  # Usar o gerenciador de memória
                print(f"Modelo ONNX '{model_name}' removido do cache")
                return True
            except Exception as e:
                print(f"Erro ao descarregar modelo ONNX '{model_name}': {str(e)}")
        return False

    def _unload_trt_model(self, model_name):
        """Descarrega um modelo TensorRT da memória de forma segura"""
        if model_name in self.models_trt_cache:
            try:
                # Verifica se é um modelo essencial
                if model_name in self.primary_models:
                    return False  # Não descarregar modelos essenciais
                    
                # Remove do cache
                model_instance = self.models_trt_cache.pop(model_name)
                # Chama o método cleanup específico do TensorRTPredictor
                if hasattr(model_instance, 'cleanup'):
                    model_instance.cleanup()
                # Limpa referências e libera memória
                del model_instance
                gc.collect()
                memory_manager.clear_cuda_cache()  # Usar o gerenciador de memória
                print(f"Modelo TensorRT '{model_name}' removido do cache")
                return True
            except Exception as e:
                print(f"Erro ao descarregar modelo TensorRT '{model_name}': {str(e)}")
        return False

    def _unload_dfm_model(self, model_name):
        """Descarrega um modelo DFM da memória de forma segura"""
        if model_name in self.dfm_models_cache:
            try:
                # Verifica se é um modelo essencial
                if model_name in self.primary_models:
                    return False  # Não descarregar modelos essenciais
                    
                # Remove do cache
                model_instance = self.dfm_models_cache.pop(model_name)
                # Limpa referências e libera memória
                del model_instance
                gc.collect()
                memory_manager.clear_cuda_cache()  # Usar o gerenciador de memória
                print(f"Modelo DFM '{model_name}' removido do cache")
                return True
            except Exception as e:
                print(f"Erro ao descarregar modelo DFM '{model_name}': {str(e)}")
        return False

    def _manage_cache_size(self, cache, limit, unload_func):
        """Gerencia o tamanho de um cache, removendo entradas menos recentemente usadas se o limite for excedido"""
        # Se o cache estiver cheio
        if len(cache) >= limit:
            # Consulta o gerenciador de memória para obter modelos candidatos a serem descarregados
            models_to_unload = memory_manager.should_unload_models()
            
            # Filtra modelos no cache que podem ser descarregados
            for model_name in list(cache.keys()):
                if model_name in models_to_unload or (model_name not in self.primary_models and len(cache) > limit):
                    # Descarrega o modelo e remove-o do cache
                    unload_func(model_name)
                    # Sai se o cache estiver abaixo do limite
                    if len(cache) < limit:
                        break

    @profile_func(sort_by='cumulative')
    def load_model(self, model_name, session_options=None):
        """Carrega um modelo ONNX, usando cache LRU para gerenciar o uso de memória"""
        log_gpu_memory_usage(f"Antes de carregar modelo {model_name}")
        with self.model_lock:
            try:
                # Registra uso do modelo no gerenciador de memória
                memory_manager.register_model_usage(model_name)
                
                # Emite sinal para iniciar carregamento
                self.start_loading_signal.emit(f"Carregando modelo: {model_name}")
                self.progress_signal.emit(0)  # 0% de progresso
                
                # Verifica primeiro no cache (cache hit)
                if model_name in self.models_cache:
                    # Move o modelo para o final do OrderedDict (marca como mais recentemente usado)
                    self.models_cache.move_to_end(model_name)
                    model_instance = self.models_cache[model_name]
                    
                    # Emite sinais de carregamento concluído
                    self.progress_signal.emit(100)  # 100% progresso
                    self.completion_signal.emit(True)  # carregamento bem-sucedido
                    log_gpu_memory_usage(f"Cache hit para modelo {model_name}")
                    
                    return model_instance
                
                # Cache miss - precisa carregar o modelo
                self.main_window.model_loading_signal.emit()
                self.progress_signal.emit(30)  # Progresso de 30%
                
                # Verifica se é necessário liberar espaço no cache primeiro
                with measure_time(f"Gerenciamento de cache para modelo {model_name}"):
                    self._manage_cache_size(self.models_cache, self.onnx_cache_limit, self._unload_model)
                
                # Verifica uso de memória e limpa cache se necessário
                if memory_manager.get_gpu_memory_usage() > 0.7:  # Se mais de 70% da GPU está em uso
                    memory_manager.clear_cuda_cache()
                
                # Carrega o modelo
                self.progress_signal.emit(50)  # Progresso de 50%
                with measure_time(f"Carregamento efetivo do modelo {model_name}"):
                    if session_options is None:
                        model_instance = onnxruntime.InferenceSession(self.models_path[model_name], providers=self.providers)
                    else:
                        model_instance = onnxruntime.InferenceSession(self.models_path[model_name], sess_options=session_options, providers=self.providers)
                
                self.progress_signal.emit(90)  # Progresso de 90%
                
                # Adiciona o modelo recém-carregado ao cache
                self.models_cache[model_name] = model_instance
                # Atualiza o dicionário original para compatibilidade com código existente
                self.models[model_name] = model_instance
                
                # Emite sinais de conclusão
                self.progress_signal.emit(100)
                self.completion_signal.emit(True)
                self.model_loaded.emit()
                
                log_gpu_memory_usage(f"Após carregar modelo {model_name}")
                
                memory_manager.log_memory_stats()  # Registra estatísticas de memória
                
                return model_instance
            except Exception as e:
                print(traceback.format_exc())
                self.completion_signal.emit(False)  # falha no carregamento
                return None
                
    # Versão modificada da função run_inswapper usando pré-alocação de buffer
    def run_inswapper(self, image, embedding, output=None):
        """
        Executa o modelo Inswapper usando buffers pré-alocados
        
        Args:
            image: Tensor de imagem de destino
            embedding: Embedding facial
            output: Buffer pré-alocado opcional para saída. Se None, será criado
            
        Returns:
            Tensor de saída com o rosto trocado
        """
        if not self.models['Inswapper128']:
            self.models['Inswapper128'] = self.load_model('Inswapper128')
        
        # Se output não for fornecido, obter do pool de buffers ou criar novo
        using_provided_buffer = output is not None
        if output is None:
            output = memory_manager.get_buffer((1, 3, 128, 128), torch.float32, "inswapper_output")
        
        io_binding = self.models['Inswapper128'].io_binding()
        io_binding.bind_input(name='target', device_type=self.device, device_id=0, element_type=np.float32, shape=(1, 3, 128, 128), buffer_ptr=image.data_ptr())
        io_binding.bind_input(name='source', device_type=self.device, device_id=0, element_type=np.float32, shape=(1, 512), buffer_ptr=embedding.data_ptr())
        io_binding.bind_output(name='output', device_type=self.device, device_id=0, element_type=np.float32, shape=(1, 3, 128, 128), buffer_ptr=output.data_ptr())

        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device != "cpu":
            self.syncvec.cpu()
        self.models['Inswapper128'].run_with_iobinding(io_binding)
        
        # Se usamos um buffer do pool e não foi fornecido pelo chamador, liberá-lo
        if not using_provided_buffer:
            # Retornamos uma cópia para evitar modificações no buffer compartilhado
            result = output.clone()
            memory_manager.release_buffer(output, "inswapper_output")
            return result
        
        return output
                
    # Método para limpeza periódica usado durante processamento de longo prazo
    def periodic_cleanup(self):
        """
        Executa limpeza de memória e cache durante processamentos longos.
        Deve ser chamado periodicamente em operações de longo prazo como processamento de vídeo.
        """
        memory_manager.clear_cuda_cache()
        
        # Verificar se é necessário descarregar modelos secundários
        models_to_unload = memory_manager.should_unload_models()
        for model_name in models_to_unload:
            if model_name in self.secondary_models:
                # Se o modelo está em algum dos caches, removê-lo
                self._unload_model(model_name)
                self._unload_trt_model(model_name)
                self._unload_dfm_model(model_name)
        
        # Registrar estatísticas
        memory_manager.log_memory_stats()

    def delete_models(self):
        """Limpa todos os modelos ONNX do cache e libera memória"""
        # Limpa o cache LRU
        self.models_cache.clear()
        
        # Mantém o código original para garantir compatibilidade
        for model_name, model_instance in self.models.items():
            del model_instance
            self.models[model_name] = None
            
        self.clip_session = []
        gc.collect()
        torch.cuda.empty_cache()

    def delete_models_trt(self):
        """Limpa todos os modelos TensorRT do cache e libera memória"""
        # Limpa o cache LRU
        self.models_trt_cache.clear()
        
        if TENSORRT_AVAILABLE:
            for model_data in models_trt_list:
                model_name = model_data['model_name']
                if isinstance(self.models_trt[model_name], TensorRTPredictor):
                    # É uma instância de TensorRTPredictor
                    self.models_trt[model_name].cleanup()
                    del self.models_trt[model_name]
                    self.models_trt[model_name] = None #Model Instance
                    
        gc.collect()
        torch.cuda.empty_cache()

    def delete_models_dfm(self):
        """Limpa todos os modelos DFM do cache e libera memória"""
        # Limpa o cache LRU
        self.dfm_models_cache.clear()
        
        # Mantém o código original para garantir compatibilidade
        keys_to_remove = []
        for model_name, model_instance in self.dfm_models.items():
            del model_instance
            keys_to_remove.append(model_name)
        
        for model_name in keys_to_remove:
            self.dfm_models.pop(model_name)
        
        self.clip_session = []
        gc.collect()
        torch.cuda.empty_cache()

    def showModelLoadingProgressBar(self):
        """Método para compatibilidade com código existente"""
        # A exibição do diálogo de progresso é gerenciada pelos sinais
        # que emitimos nas funções de carregamento de modelos
        pass

    def hideModelLoadProgressBar(self):
        """Método para compatibilidade com código existente"""
        # A ocultação do diálogo de progresso é gerenciada pelos sinais
        # que emitimos nas funções de carregamento de modelos
        pass

    def switch_providers_priority(self, provider_name):
        match provider_name:
            case "TensorRT" | "TensorRT-Engine":
                providers = [
                                ('TensorrtExecutionProvider', self.trt_ep_options),
                                ('CUDAExecutionProvider'),
                                ('CPUExecutionProvider')
                            ]
                self.device = 'cuda'
                if version.parse(trt.__version__) < version.parse("10.2.0") and provider_name == "TensorRT-Engine":
                    print("TensorRT-Engine provider cannot be used when TensorRT version is lower than 10.2.0.")
                    provider_name = "TensorRT"

            case "CPU":
                providers = [
                                ('CPUExecutionProvider')
                            ]
                self.device = 'cpu'
            case "CUDA":
                providers = [
                                ('CUDAExecutionProvider'),
                                ('CPUExecutionProvider')
                            ]
                self.device = 'cuda'
            #case _:

        self.providers = providers
        self.provider_name = provider_name
        self.lp_mask_crop = self.lp_mask_crop.to(self.device)

        return self.provider_name

    def set_number_of_threads(self, value):
        self.nThreads = value
        self.delete_models_trt()

    def get_gpu_memory(self):
        command = "nvidia-smi --query-gpu=memory.total --format=csv"
        memory_total_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
        memory_total = [int(x.split()[0]) for i, x in enumerate(memory_total_info)]

        command = "nvidia-smi --query-gpu=memory.free --format=csv"
        memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
        memory_free = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]

        memory_used = memory_total[0] - memory_free[0]

        return memory_used, memory_total[0]
    
    def clear_gpu_memory(self):
        self.delete_models()
        self.delete_models_dfm()
        self.delete_models_trt()
        torch.cuda.empty_cache()


    def load_inswapper_iss_emap(self, model_name):
        with self.model_lock:
            if not self.models[model_name]:
                self.main_window.model_loading_signal.emit()
                graph = onnx.load(self.models_path[model_name]).graph
                self.emap = onnx.numpy_helper.to_array(graph.initializer[-1])
                self.main_window.model_loaded_signal.emit()

    def load_dfm_model(self, dfm_model):
        """
        Carrega um modelo DFM (DeepFaceLive) e retorna a instância do modelo.
        
        Args:
            dfm_model: Nome do modelo DFM a ser carregado
            
        Returns:
            Instância do modelo DFM carregado ou None se falhar
        """
        with self.model_lock:
            # Verificar se o modelo já está no cache
            if dfm_model in self.dfm_models_cache:
                # Mover para o final da fila LRU
                self.dfm_models_cache.move_to_end(dfm_model)
                return self.dfm_models_cache[dfm_model]
                
            # Emitir sinal de início de carregamento
            self.start_loading_signal.emit(f"Carregando modelo DFM: {dfm_model}")
            
            # Verificar limite de modelos carregados
            max_models_to_keep = getattr(self.main_window, 'control', {}).get('MaxDFMModelsSlider', 3)
            self._manage_cache_size(self.dfm_models_cache, max_models_to_keep, self._unload_dfm_model)
            
            try:
                # Importar a classe DFMModel do módulo utils
                from app.processors.utils.dfm_model import DFMModel
                
                # Obter o caminho do modelo a partir dos dados do main_window
                model_path = self.main_window.dfm_models_data.get(dfm_model)
                if not model_path:
                    raise ValueError(f"Caminho do modelo DFM '{dfm_model}' não encontrado")
                
                # Carregar o modelo
                model_instance = DFMModel(model_path, self.providers, self.device)
                
                # Adicionar ao cache
                self.dfm_models_cache[dfm_model] = model_instance
                
                # Emitir sinal de conclusão
                self.completion_signal.emit(True)
                return model_instance
            except Exception as e:
                print(f"Erro ao carregar modelo DFM '{dfm_model}': {str(e)}")
                print(traceback.format_exc())
                self.completion_signal.emit(False)
                return None

    def run_detect(self, img, detect_mode='RetinaFace', max_num=1, score=0.5, input_size=(512, 512), use_landmark_detection=False, landmark_detect_mode='203', landmark_score=0.5, from_points=False, rotation_angles=None):
        rotation_angles = rotation_angles or [0]
        return self.face_detectors.run_detect(img, detect_mode, max_num, score, input_size, use_landmark_detection, landmark_detect_mode, landmark_score, from_points, rotation_angles)
    
    def run_detect_landmark(self, img, bbox, det_kpss, detect_mode='203', score=0.5, from_points=False):
        return self.face_landmark_detectors.run_detect_landmark(img, bbox, det_kpss, detect_mode, score, from_points)

    def get_arcface_model(self, face_swapper_model): 
        if face_swapper_model in arcface_mapping_model_dict:
            return arcface_mapping_model_dict[face_swapper_model]
        else:
            raise ValueError(f"Face swapper model {face_swapper_model} not found.")

    def run_recognize_direct(self, img, kps, similarity_type='Opal', arcface_model='Inswapper128ArcFace'):
        return self.face_swappers.run_recognize_direct(img, kps, similarity_type, arcface_model)

    def calc_inswapper_latent(self, source_embedding):
        return self.face_swappers.calc_inswapper_latent(source_embedding)

    def run_inswapper(self, image, embedding, output):
        self.face_swappers.run_inswapper(image, embedding, output)

    def calc_swapper_latent_iss(self, source_embedding, version="A"):
        return self.face_swappers.calc_swapper_latent_iss(source_embedding, version)

    def run_iss_swapper(self, image, embedding, output, version="A"):
        self.face_swappers.run_iss_swapper(image, embedding, output, version)

    def calc_swapper_latent_simswap512(self, source_embedding):
        return self.face_swappers.calc_swapper_latent_simswap512(source_embedding)

    def run_swapper_simswap512(self, image, embedding, output):
        self.face_swappers.run_swapper_simswap512(image, embedding, output)

    def calc_swapper_latent_ghost(self, source_embedding):
        return self.face_swappers.calc_swapper_latent_ghost(source_embedding)

    def run_swapper_ghostface(self, image, embedding, output, swapper_model='GhostFace-v2'):
        self.face_swappers.run_swapper_ghostface(image, embedding, output, swapper_model)

    def calc_swapper_latent_cscs(self, source_embedding):
        return self.face_swappers.calc_swapper_latent_cscs(source_embedding)

    def run_swapper_cscs(self, image, embedding, output):
        self.face_swappers.run_swapper_cscs(image, embedding, output)

    def run_enhance_frame_tile_process(self, img, enhancer_type, tile_size=256, scale=1):
        return self.frame_enhancers.run_enhance_frame_tile_process(img, enhancer_type, tile_size, scale)

    def run_deoldify_artistic(self, image, output):
        return self.frame_enhancers.run_deoldify_artistic(image, output)

    def run_deoldify_stable(self, image, output):
        return self.frame_enhancers.run_deoldify_artistic(image, output)
    
    def run_deoldify_video(self, image, output):
        return self.frame_enhancers.run_deoldify_video(image, output)
    
    def run_ddcolor_artistic(self, image, output):
        return self.frame_enhancers.run_ddcolor_artistic(image, output)

    def run_ddcolor(self, tensor_gray_rgb, output_ab):
        return self.frame_enhancers.run_ddcolor(tensor_gray_rgb, output_ab)

    def run_occluder(self, image, output):
        self.face_masks.run_occluder(image, output)

    def run_dfl_xseg(self, image, output):
        self.face_masks.run_dfl_xseg(image, output)

    def run_faceparser(self, image, output):
        self.face_masks.run_faceparser(image, output)

    def run_CLIPs(self, img, CLIPText, CLIPAmount):
        return self.face_masks.run_CLIPs(img, CLIPText, CLIPAmount)
    
    def lp_motion_extractor(self, img, face_editor_type='Human-Face', **kwargs) -> dict:
        return self.face_editors.lp_motion_extractor(img, face_editor_type, **kwargs)

    def lp_appearance_feature_extractor(self, img, face_editor_type='Human-Face'):
        return self.face_editors.lp_appearance_feature_extractor(img, face_editor_type)

    def lp_retarget_eye(self, kp_source: torch.Tensor, eye_close_ratio: torch.Tensor, face_editor_type='Human-Face') -> torch.Tensor:
        return self.face_editors.lp_retarget_eye(kp_source, eye_close_ratio, face_editor_type)

    def lp_retarget_lip(self, kp_source: torch.Tensor, lip_close_ratio: torch.Tensor, face_editor_type='Human-Face') -> torch.Tensor:
        return self.face_editors.lp_retarget_lip(kp_source, lip_close_ratio, face_editor_type)

    def lp_stitch(self, kp_source: torch.Tensor, kp_driving: torch.Tensor, face_editor_type='Human-Face') -> torch.Tensor:
        return self.face_editors.lp_stitch(kp_source, kp_driving, face_editor_type)

    def lp_stitching(self, kp_source: torch.Tensor, kp_driving: torch.Tensor, face_editor_type='Human-Face') -> torch.Tensor:
        return self.face_editors.lp_stitching(kp_source, kp_driving, face_editor_type)

    def lp_warp_decode(self, feature_3d: torch.Tensor, kp_source: torch.Tensor, kp_driving: torch.Tensor, face_editor_type='Human-Face') -> torch.Tensor:
        return self.face_editors.lp_warp_decode(feature_3d, kp_source, kp_driving, face_editor_type)

    def findCosineDistance(self, vector1, vector2):
        vector1 = vector1.ravel()
        vector2 = vector2.ravel()
        cos_dist = 1 - np.dot(vector1, vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2)) # 2..0
        return 100-cos_dist*50

    def apply_facerestorer(self, swapped_face_upscaled, restorer_det_type, restorer_type, restorer_blend, fidelity_weight, detect_score):
        return self.face_restorers.apply_facerestorer(swapped_face_upscaled, restorer_det_type, restorer_type, restorer_blend, fidelity_weight, detect_score)

    def apply_occlusion(self, img, amount):
        return self.face_masks.apply_occlusion(img, amount)
    
    def apply_dfl_xseg(self, img, amount):
        return self.face_masks.apply_dfl_xseg(img, amount)
    
    def apply_face_parser(self, img, parameters):
        return self.face_masks.apply_face_parser(img, parameters)
    
    def apply_face_makeup(self, img, parameters):
        return self.face_editors.apply_face_makeup(img, parameters)
    
    def restore_mouth(self, img_orig, img_swap, kpss_orig, blend_alpha=0.5, feather_radius=10, size_factor=0.5, radius_factor_x=1.0, radius_factor_y=1.0, x_offset=0, y_offset=0):
        return self.face_masks.restore_mouth(img_orig, img_swap, kpss_orig, blend_alpha, feather_radius, size_factor, radius_factor_x, radius_factor_y, x_offset, y_offset)

    def restore_eyes(self, img_orig, img_swap, kpss_orig, blend_alpha=0.5, feather_radius=10, size_factor=3.5, radius_factor_x=1.0, radius_factor_y=1.0, x_offset=0, y_offset=0, eye_spacing_offset=0):
        return self.face_masks.restore_eyes(img_orig, img_swap, kpss_orig, blend_alpha, feather_radius, size_factor, radius_factor_x, radius_factor_y, x_offset, y_offset, eye_spacing_offset)

    def apply_fake_diff(self, swapped_face, original_face, DiffAmount):
        return self.face_masks.apply_fake_diff(swapped_face, original_face, DiffAmount)
