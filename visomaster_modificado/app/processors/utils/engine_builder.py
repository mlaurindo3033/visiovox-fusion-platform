# pylint: disable=no-member

import os
import sys
import logging
import platform
import ctypes
import time
import threading
import queue
from pathlib import Path

try:
    import tensorrt as trt
except ModuleNotFoundError:
    pass

logging.basicConfig(level=logging.INFO)
logging.getLogger("EngineBuilder").setLevel(logging.INFO)
log = logging.getLogger("EngineBuilder")

if 'trt' in globals():
    # Creazione di un'istanza globale di logger di TensorRT
    TRT_LOGGER = trt.Logger(trt.Logger.INFO) # pylint: disable=no-member
else:
    TRT_LOGGER = {}

# imported from https://github.com/warmshao/FasterLivePortrait/blob/master/scripts/onnx2trt.py
# adjusted to work with TensorRT 10.3.0
class EngineBuilder:
    """
    Parses an ONNX graph and builds a TensorRT engine from it.
    """

    def __init__(self, verbose=False, custom_plugin_path=None, builder_optimization_level=3, progress_callback=None):
        """
        :param verbose: If enabled, a higher verbosity level will be set on the TensorRT logger.
        :param custom_plugin_path: Path to the custom plugin library (DLL or SO).
        :param builder_optimization_level: Optimization level for TensorRT builder.
        :param progress_callback: Callback function to report progress (0-100) durante a conversão.
        """
        self.progress_callback = progress_callback
        
        if verbose:
            TRT_LOGGER.min_severity = trt.Logger.Severity.VERBOSE

        # Inicializa os plugins de TensorRT
        trt.init_libnvinfer_plugins(TRT_LOGGER, namespace="")

        # Construisce il builder di TensorRT e la configurazione usando lo stesso logger
        self.builder = trt.Builder(TRT_LOGGER)
        self.config = self.builder.create_builder_config()
        # Imposta il limite di memoria del pool di lavoro a 3 GB
        self.config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 3 * (2 ** 30))  # 3 GB

        # Imposta il livello di ottimizzazione del builder (se fornito)
        self.config.builder_optimization_level = builder_optimization_level

        # Crea un profilo di ottimizzazione, se necessario
        profile = self.builder.create_optimization_profile()
        self.config.add_optimization_profile(profile)

        self.batch_size = None
        self.network = None
        self.parser = None

        # Carica plugin personalizzati se specificato
        if custom_plugin_path is not None:
            if platform.system().lower() == 'linux':
                ctypes.CDLL(custom_plugin_path, mode=ctypes.RTLD_GLOBAL)
            else:
                ctypes.CDLL(custom_plugin_path, mode=ctypes.RTLD_GLOBAL, winmode=0)
                
        # Valores para controle de progresso
        self.start_time = time.time()
        self.progress_steps = [
            (0, "Inicializando"), 
            (3, "Carregando plugins"), 
            (5, "Configurando")
        ]
        
        # Fila de comunicação entre threads
        self.progress_queue = queue.Queue()
        self.stop_event = threading.Event()
        
        # Reportar progresso inicial
        self._report_progress(2, "Inicializando TensorRT")

    def _report_progress(self, progress, message=None):
        """Reporta o progresso da conversão para a callback, se disponível"""
        if self.progress_callback and callable(self.progress_callback):
            try:
                # Chama a callback com o valor atual de progresso (0-100)
                self.progress_callback(progress, message)
            except Exception as e:
                log.warning(f"Erro ao reportar progresso: {e}")
                # Continua mesmo se houver erro no callback

    def create_network(self, onnx_path):
        """
        Parse the ONNX graph and create the corresponding TensorRT network definition.
        :param onnx_path: The path to the ONNX graph to load.
        """
        self._report_progress(7, "Criando definição de rede")
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

        self.network = self.builder.create_network(network_flags)
        self.parser = trt.OnnxParser(self.network, TRT_LOGGER)

        onnx_path = os.path.realpath(onnx_path)
        self._report_progress(10, "Preparando para carregar arquivo ONNX")
        
        with open(onnx_path, "rb") as f:
            self._report_progress(12, "Analisando arquivo ONNX")
            if not self.parser.parse(f.read()):
                log.error("Failed to load ONNX file: %s", onnx_path)
                for error in range(self.parser.num_errors):
                    log.error(self.parser.get_error(error))
                sys.exit(1)
        
        self._report_progress(18, "Arquivo ONNX carregado com sucesso")
        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]

        log.info("Network Description")
        for net_input in inputs:
            self.batch_size = net_input.shape[0]
            log.info("Input '%s' with shape %s and dtype %s", net_input.name, net_input.shape, net_input.dtype)
        for net_output in outputs:
            log.info("Output %s' with shape %s and dtype %s", net_output.name, net_output.shape, net_output.dtype)
        
        self._report_progress(20, "Analisando camadas da rede")

    def create_engine(self, engine_path, precision):
        """
        Build the TensorRT engine and serialize it to disk.
        :param engine_path: The path where to serialize the engine to.
        :param precision: The datatype to use for the engine, either 'fp32', 'fp16' or 'int8'.
        """
        engine_path = os.path.realpath(engine_path)
        engine_dir = os.path.dirname(engine_path)
        os.makedirs(engine_dir, exist_ok=True)
        log.info("Building %s Engine in %s", precision, engine_path)
        self._report_progress(25, "Preparando configuração do motor TensorRT")

        # Forza TensorRT a rispettare i vincoli di precisione
        self.config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    
        if precision == "fp16":
            if not self.builder.platform_has_fast_fp16:
                log.warning("FP16 is not supported natively on this platform/device")
            else:
                self.config.set_flag(trt.BuilderFlag.FP16)
        
        self._report_progress(30, "Configurando precisão do motor")
        
        # Sistema avançado de monitoramento de progresso usando watchdog thread
        def progress_monitor():
            """Thread dedicada para monitorar o progresso e atualizar a UI mesmo quando a thread principal está ocupada"""
            current_progress = 30
            max_progress = 90
            
            # Progresso calculado: começa lento e acelera no meio
            progress_points = []
            
            # Gera pontos de progresso de 1% a 89%
            for i in range(1, 90):
                # Cria uma curva que começa mais lenta, acelera no meio e diminui no final
                if i < 30:
                    # Pontos iniciais mais espaçados (progresso mais lento)
                    delay = 0.7
                elif i < 60:
                    # Pontos do meio mais próximos (progresso mais rápido)
                    delay = 0.5
                else:
                    # Pontos finais mais espaçados novamente
                    delay = 0.7
                    
                progress_points.append((i, delay))
            
            # Loop principal do monitor de progresso
            try:
                for progress, delay in progress_points:
                    if self.stop_event.is_set():
                        break
                    
                    # Atualiza o progresso
                    message = f"Otimizando motor TensorRT: {progress}%"
                    if progress % 10 == 0:
                        if progress < 40:
                            message = f"Analisando rede neural: {progress}%"
                        elif progress < 60:
                            message = f"Otimizando camadas: {progress}%"
                        elif progress < 80:
                            message = f"Construindo motor: {progress}%"
                        else:
                            message = f"Finalizando construção: {progress}%"
                    
                    # Envia atualizações para a fila, não diretamente
                    self.progress_queue.put((progress, message))
                    
                    # Dorme um tempo variável entre atualizações
                    time.sleep(delay)
                    
                    # Se a conversão estiver demorando muito, acelera as atualizações
                    elapsed = time.time() - self.start_time
                    if elapsed > 180:  # > 3 minutos
                        delay *= 0.7  # 30% mais rápido
                    
            except Exception as e:
                log.error(f"Erro no monitor de progresso: {e}")
                
        # Thread separada para reportar o progresso da fila para o callback
        def progress_reporter():
            """Thread dedicada para enviar as atualizações de progresso da fila para o callback"""
            try:
                while not self.stop_event.is_set():
                    try:
                        # Tenta obter um item da fila com timeout para não bloquear para sempre
                        progress, message = self.progress_queue.get(timeout=0.1)
                        self._report_progress(progress, message)
                        self.progress_queue.task_done()
                    except queue.Empty:
                        # Se a fila estiver vazia, apenas continua
                        continue
            except Exception as e:
                log.error(f"Erro no reporter de progresso: {e}")
        
        # Inicia as threads de monitoramento e reporte
        monitor_thread = threading.Thread(target=progress_monitor, daemon=True)
        reporter_thread = threading.Thread(target=progress_reporter, daemon=True)
        
        try:
            # Costruzione del motore serializzato
            log.info("Iniciando a construção do motor TensorRT (pode demorar vários minutos)...")
            self._report_progress(30, "Iniciando otimização e construção (processo longo)")
            
            # Inicia as threads
            monitor_thread.start()
            reporter_thread.start()
            
            # Marca o tempo de início para cálculos precisos
            build_start_time = time.time()
            
            # Construção propriamente dita (operação intensiva)
            serialized_engine = self.builder.build_serialized_network(self.network, self.config)
            
            # Calcula o tempo real de construção
            build_time = time.time() - build_start_time
            
            # Sinaliza para as threads pararem
            self.stop_event.set()
            
            # Garante que as threads terminem
            monitor_thread.join(timeout=1.0)
            reporter_thread.join(timeout=1.0)
            
            # Limpa a fila
            while not self.progress_queue.empty():
                try:
                    self.progress_queue.get_nowait()
                    self.progress_queue.task_done()
                except queue.Empty:
                    break
            
            # Atualiza para 90% após conclusão
            self._report_progress(90, f"Construção completada em {build_time:.1f}s, serializando motor")

            # Verifica che il motore sia stato serializzato correttamente
            if serialized_engine is None:
                self._report_progress(0, "Falha na construção do motor")
                raise RuntimeError("Errore nella costruzione del motore TensorRT!")

            # Scrittura del motore serializzato su disco
            with open(engine_path, "wb") as f:
                log.info("Serializing engine to file: %s", engine_path)
                f.write(serialized_engine)
            
            self._report_progress(95, "Motor serializado em arquivo")
            self._report_progress(100, f"Processo completo em {build_time:.1f}s")
            
        except Exception as e:
            # Sinaliza para as threads pararem em caso de erro
            self.stop_event.set()
            if monitor_thread.is_alive():
                monitor_thread.join(timeout=1.0)
            if reporter_thread.is_alive():
                reporter_thread.join(timeout=1.0)
                
            self._report_progress(0, f"Erro: {str(e)}")
            raise

def change_extension(file_path, new_extension, version=None):
    """
    Change the extension of the file path and optionally prepend a version.
    """
    # Remove leading '.' from the new extension if present
    new_extension = new_extension.lstrip('.')

    # Create the new file path with the version before the extension, if provided
    if version:
        new_file_path = Path(file_path).with_suffix(f'.{version}.{new_extension}')
    else:
        new_file_path = Path(file_path).with_suffix(f'.{new_extension}')

    return str(new_file_path)

def onnx_to_trt(onnx_model_path, trt_model_path=None, precision="fp16", custom_plugin_path=None, verbose=False, progress_callback=None):
    # The precision mode to build in, either 'fp32', 'fp16' or 'int8', default: 'fp16'"

    if trt_model_path is None:
        trt_version = trt.__version__
        trt_model_path = change_extension(onnx_model_path, "trt", version=trt_version)
    
    # Cria o builder com a callback de progresso
    builder = EngineBuilder(verbose=verbose, 
                           custom_plugin_path=custom_plugin_path, 
                           progress_callback=progress_callback)

    builder.create_network(onnx_model_path)
    builder.create_engine(trt_model_path, precision)
