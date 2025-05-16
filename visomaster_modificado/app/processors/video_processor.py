import threading
import queue
from typing import TYPE_CHECKING, Dict, Tuple
import time
import subprocess
from pathlib import Path
import os
import gc
from functools import partial

import cv2
import numpy as np
import torch
import pyvirtualcam

from PySide6.QtCore import QObject, QTimer, Signal, Slot
from PySide6.QtGui import QPixmap
from app.processors.workers.frame_worker import FrameWorker
from app.ui.widgets.actions import graphics_view_actions
from app.ui.widgets.actions import common_actions as common_widget_actions

from app.ui.widgets.actions import video_control_actions
from app.ui.widgets.actions import layout_actions
import app.helpers.miscellaneous as misc_helpers

from app.helpers.memory_manager import memory_manager  # Importar o gerenciador de memória

import sys
import glob
import datetime
import math
import json
import logging
import hashlib

try:
    import av
    USE_PYAV = True
except ModuleNotFoundError:
    print("No PyAV Found, using OpenCV for Frame Extraction")
    USE_PYAV = False

from app.helpers.miscellaneous import get_hash_from_filename, read_frame
from app.helpers.profiler import profile_func, measure_time, log_gpu_memory_usage

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

# Constantes para gerenciamento de memória
MEMORY_CLEANUP_INTERVAL = 20  # Limpar memória a cada 20 frames
MEMORY_USAGE_THRESHOLD = 0.7  # Limpar memória se o uso estiver acima de 70%

CACHE_BASE_DIR = "cache_frames"
DEFAULT_CACHE_FORMAT = 'webp'
DEFAULT_CACHE_QUALITY = 90

def get_cache_dir_for_video(media_path):
    """Gera um diretório de cache único para o vídeo com base no hash do caminho."""
    video_hash = hashlib.md5(media_path.encode('utf-8')).hexdigest()
    return os.path.join(CACHE_BASE_DIR, video_hash)

def save_frame_to_cache(frame, frame_number, media_path, format='png', quality=95, metadata=None):
    cache_dir = get_cache_dir_for_video(media_path)
    os.makedirs(cache_dir, exist_ok=True)
    if format == 'png':
        path = os.path.join(cache_dir, f"frame_{frame_number:05d}.png")
        cv2.imwrite(path, frame)
    elif format == 'jpg':
        path = os.path.join(cache_dir, f"frame_{frame_number:05d}.jpg")
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    elif format == 'webp':
        path = os.path.join(cache_dir, f"frame_{frame_number:05d}.webp")
        cv2.imwrite(path, frame, [cv2.IMWRITE_WEBP_QUALITY, quality])
    # Salvar metadados se fornecidos
    if metadata is not None:
        meta_path = os.path.join(cache_dir, 'cache_metadata.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

def load_frame_from_cache(frame_number, media_path, format='png'):
    cache_dir = get_cache_dir_for_video(media_path)
    if format == 'png':
        path = os.path.join(cache_dir, f"frame_{frame_number:05d}.png")
    elif format == 'jpg':
        path = os.path.join(cache_dir, f"frame_{frame_number:05d}.jpg")
    elif format == 'webp':
        path = os.path.join(cache_dir, f"frame_{frame_number:05d}.webp")
    else:
        return None
    if os.path.exists(path):
        return cv2.imread(path)
    return None

def save_cache_metadata(media_path, metadata):
    cache_dir = get_cache_dir_for_video(media_path)
    os.makedirs(cache_dir, exist_ok=True)
    meta_path = os.path.join(cache_dir, 'cache_metadata.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def load_cache_metadata(media_path):
    cache_dir = get_cache_dir_for_video(media_path)
    meta_path = os.path.join(cache_dir, 'cache_metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def clear_cache_for_video(media_path):
    cache_dir = get_cache_dir_for_video(media_path)
    if os.path.exists(cache_dir):
        for f in os.listdir(cache_dir):
            os.remove(os.path.join(cache_dir, f))
        os.rmdir(cache_dir)

# Função auxiliar para gerar metadados básicos do cache
def generate_cache_metadata(media_path, params=None):
    return {
        'media_hash': hashlib.md5(media_path.encode('utf-8')).hexdigest(),
        'date': datetime.datetime.now().isoformat(),
        'parameters': params or {},
    }

class VideoProcessor(QObject):
    frame_processed_signal = Signal(int, QPixmap, np.ndarray)
    webcam_frame_processed_signal = Signal(QPixmap, np.ndarray)
    single_frame_processed_signal = Signal(int, QPixmap, np.ndarray)
    processing_complete = Signal(float, float)  # Emits elapsed time and FPS
    single_frame_processed = Signal(np.ndarray, float, bool)  # Emits processed frame, frame number and status
    video_processing_started = Signal()
    video_processing_stopped = Signal()
    processing_progress = Signal(float, int, int)  # Emits progress (0.0-1.0), current frame, total frames
    save_media_progress = Signal(int, int)  # Emits frame received, total frames
    def __init__(self, main_window: 'MainWindow', num_threads=2):
        super(VideoProcessor, self).__init__()
        self.main_window = main_window
        self.num_threads = num_threads
        self.frame_queue = queue.Queue(maxsize=num_threads)
        self.workers = []
        self.file_type = None
        self.media_path = None
        self.current_frame = None
        self.video_writer = None
        self.recording = False
        self.processing = False
        
        # Contadores para gerenciamento de memória
        self.frame_count_since_cleanup = 0
        self.last_memory_cleanup_time = time.time()
        
        # Buffer pools para reutilização
        self.frame_buffer_pool = []  # Pool de buffers para frames
        self.max_buffer_pool_size = 10  # Tamanho máximo do pool
        
        # Initialize variables for processing
        self.media_capture = None
        self.webcam_thread = None
        self.stop_event = threading.Event()
        self.webcam_stop_event = threading.Event()
        self.enable_webcam_flag = False
        self.webcam_path = 0  # Default webcam
        self.current_frame_number = 0
        self.max_frame_number = 0
        self.fps = 0.0
        self.frame_width = 0
        self.frame_height = 0
        self.frame_count = 0
        self.next_frame_to_display = 0  # For thread-safe UI updates
        self.display_frame_lock = threading.Lock()
        self.total_frames_processed = 0
        self.output_filename = ""
        
        # Dicionário para armazenar os frames a serem exibidos
        self.frames_to_display = {}
        self.webcam_frames_to_display = queue.Queue()
        
        # Virtual camera
        self.virtual_camera = None
        self.virtual_cam_enabled = False
        self.virtual_cam_backend = 'auto'  # 'auto', 'pyvirtualcam', 'obs'
        self.enable_obs_virtualcam = False
        self.enable_virtualcam_flag = False
        self.ffmpeg_process = None
        
        # Timers para melhorar a reprodução de vídeo
        self.frame_read_timer = QTimer()
        self.frame_read_timer.timeout.connect(self.process_next_frame_from_timer)
        
        self.frame_display_timer = QTimer()
        self.frame_display_timer.timeout.connect(self.update_video_display)
        
        self.gpu_memory_update_timer = QTimer()
        self.gpu_memory_update_timer.timeout.connect(self.update_gpu_memory_stats)
        
        # Timer para limpeza periódica de memória
        self.memory_cleanup_timer = QTimer()
        self.memory_cleanup_timer.timeout.connect(self.periodic_memory_cleanup)
        self.memory_cleanup_timer.setInterval(30000)  # 30 segundos
        
        # Variáveis para estatísticas
        self.start_time = 0
        self.play_start_time = 0
        self.frames_displayed = 0
        self.current_fps = 0

        self.frame_processed_signal.connect(self.store_frame_to_display)
        self.webcam_frame_processed_signal.connect(self.store_webcam_frame_to_display)
        self.single_frame_processed_signal.connect(self.display_current_frame)

    def periodic_memory_cleanup(self):
        """Executa limpeza periódica de memória durante processamento de vídeo"""
        if self.processing:
            # Chamar o método de limpeza do models_processor
            self.main_window.models_processor.periodic_cleanup()
            
            # Registrar estatísticas
            memory_manager.log_memory_stats()
    
    def get_frame_buffer(self, height, width, channels=3, dtype=np.uint8):
        """
        Obtém um buffer pré-alocado para frames ou cria um novo se necessário
        
        Args:
            height: Altura do frame
            width: Largura do frame
            channels: Número de canais (padrão: 3 para RGB)
            dtype: Tipo de dados (padrão: np.uint8)
            
        Returns:
            Buffer pré-alocado
        """
        # Usa o gerenciador de memória para obter um buffer
        shape = (height, width, channels)
        return memory_manager.get_buffer(shape, torch.from_numpy(np.zeros(1, dtype=dtype)).dtype, "frame_buffer")
    
    def release_frame_buffer(self, buffer):
        """
        Devolve um buffer ao pool
        
        Args:
            buffer: Buffer a ser liberado
        """
        if buffer is not None:
            memory_manager.release_buffer(buffer, "frame_buffer")

    Slot(int, QPixmap, np.ndarray)
    def store_frame_to_display(self, frame_number, pixmap, frame):
        # Salvar frame processado no cache persistente (WebP por padrão)
        if self.media_path:
            # Gerar metadados básicos (pode expandir com mais parâmetros se desejar)
            metadata = generate_cache_metadata(self.media_path)
            save_frame_to_cache(frame, frame_number, self.media_path, format=DEFAULT_CACHE_FORMAT, quality=DEFAULT_CACHE_QUALITY, metadata=metadata if frame_number == 0 else None)
        self.frames_to_display[frame_number] = (pixmap, frame)
        # Verificar se é necessário limpeza de memória
        self.frame_count_since_cleanup += 1
        if self.frame_count_since_cleanup >= MEMORY_CLEANUP_INTERVAL:
            self.check_memory_usage()

    # Use a queue to store the webcam frames, since the order of frames is not that important (Unless there are too many threads)
    Slot(QPixmap, np.ndarray)
    def store_webcam_frame_to_display(self, pixmap, frame):
        # print("Called store_webcam_frame_to_display()")
        self.webcam_frames_to_display.put((pixmap, frame))
        
        # Verificar periódico de memória
        self.frame_count_since_cleanup += 1
        if self.frame_count_since_cleanup >= MEMORY_CLEANUP_INTERVAL:
            self.check_memory_usage()

    def check_memory_usage(self):
        """Verifica se o uso de memória está alto e executa limpeza se necessário"""
        self.frame_count_since_cleanup = 0
        
        # Usar o gerenciador de memória para verificar o uso de GPU
        usage = memory_manager.get_gpu_memory_usage()
        
        # Se o uso estiver acima do limite ou passou muito tempo desde a última limpeza
        current_time = time.time()
        time_since_last_cleanup = current_time - self.last_memory_cleanup_time
        
        if usage > MEMORY_USAGE_THRESHOLD or time_since_last_cleanup > 60:  # 60 segundos
            self.main_window.models_processor.periodic_cleanup()
            self.last_memory_cleanup_time = current_time
            memory_manager.log_memory_stats()

    Slot(int, QPixmap, np.ndarray)
    def display_current_frame(self, frame_number, pixmap, frame):
        # Tenta carregar do cache persistente antes de exibir (WebP por padrão)
        cached = self.load_frame_from_persistent_cache(frame_number)
        if cached is not None:
            frame = cached
        if self.main_window.loading_new_media:
            graphics_view_actions.update_graphics_view(self.main_window, pixmap, frame_number, reset_fit=True)
            self.main_window.loading_new_media = False
        else:
            graphics_view_actions.update_graphics_view(self.main_window, pixmap, frame_number,)
        self.current_frame = frame
        # Limpar cache CUDA usando o gerenciador de memória
        memory_manager.clear_cuda_cache(force=False)  # Limpa apenas se necessário
        #Set GPU Memory Progressbar
        common_widget_actions.update_gpu_memory_progressbar(self.main_window)

    def display_next_frame(self):
        if not self.processing or (self.next_frame_to_display > self.max_frame_number):
            self.stop_processing()
        if self.next_frame_to_display not in self.frames_to_display:
            return
        else:
            pixmap, frame = self.frames_to_display.pop(self.next_frame_to_display)
            self.current_frame = frame

            # Check and send the frame to virtualcam, if the option is selected
            self.send_frame_to_virtualcam(frame)

            if self.recording:
                self.recording_sp.stdin.write(frame.tobytes())
            # Update the widget values using parameters if it is not recording (The updation of actual parameters is already done inside the FrameWorker, this step is to make the changes appear in the widgets)
            if not self.recording:
                video_control_actions.update_widget_values_from_markers(self.main_window, self.next_frame_to_display)
            graphics_view_actions.update_graphics_view(self.main_window, pixmap, self.next_frame_to_display)
            self.threads.pop(self.next_frame_to_display)
            self.next_frame_to_display += 1

    def display_next_webcam_frame(self):
        # print("Called display_next_webcam_frame()")
        if not self.processing:
            self.stop_processing()
        if self.webcam_frames_to_display.empty():
            # print("No Webcam frame found to display")
            return
        else:
            pixmap, frame = self.webcam_frames_to_display.get()
            self.current_frame = frame
            self.send_frame_to_virtualcam(frame)
            graphics_view_actions.update_graphics_view(self.main_window, pixmap, 0)

    def send_frame_to_virtualcam(self, frame: np.ndarray):
        if self.main_window.control['SendVirtCamFramesEnableToggle'] and self.virtcam:
            # Check if the dimensions of the frame matches that of the Virtcam object
            # If it doesn't match, reinstantiate the Virtcam object with new dimensions
            height, width, _ = frame.shape
            if self.virtcam.height!=height or self.virtcam.width!=width:
                self.enable_virtualcam()
            try:
                self.virtcam.send(frame)
                self.virtcam.sleep_until_next_frame()
            except Exception as e:
                print(e)

    def set_number_of_threads(self, value):
        """Atualiza o número de threads para processamento de frames.
        
        Args:
            value: Novo número de threads desejado
        """
        was_processing = self.processing
        self.stop_processing()  # Para o processamento atual e limpa recursos
        
        # Atualiza o número de threads no models_processor
        self.main_window.models_processor.set_number_of_threads(value)
        
        # Atualiza configurações locais
        self.num_threads = value
        self.frame_queue = queue.Queue(maxsize=self.num_threads)
        
        print(f"Número máximo de threads definido como {value}")
        
        # Se estava processando antes, reinicia o processamento com as novas configurações
        if was_processing:
            self.process_video()

    @profile_func(sort_by='cumulative')
    def process_video(self, start_frame=0, stop_frame=None, increment=1, flush=True, swapper=None, editor=None):
        start_time = time.time()
        
        print("VideoProcessor: Checking video")
        if self.media_capture is None:
            print("No video source set!")
            return
            
        # Reset stop event
        self.stop_event.clear()
        self.processing = True
        self.video_processing_started.emit()

        # Reset counters
        frames_processed = 0
        self.total_frames_processed = 0
        self.frames_displayed = 0
        
        # Reiniciar contadores de gerenciamento de memória
        self.frame_count_since_cleanup = 0
        self.last_memory_cleanup_time = time.time()

        # Get the path to output folder (usually it is the workspace folder inside user data folder)
        output_filename = self.main_window.control.get('OutputMediaFolder', "processed_output")
        root_path = Path(output_filename)
        
        # Create the output directory if it doesn't exist
        output_dir = root_path
        os.makedirs(output_dir, exist_ok=True)

        if stop_frame is None:
            stop_frame = self.max_frame_number

        # Initialize video writer if not already
        if self.recording:
            self.output_filename = os.path.join(output_dir, f"VisoMaster_Output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.output_filename, fourcc, self.fps, (self.frame_width, self.frame_height))
        
        total_frames_to_process = (stop_frame - start_frame) / increment
        skipped_frames = 0
        
        # Set video position to the start frame
        self.set_video_position(start_frame)
        self.current_frame_number = start_frame
        
        # Configurar tempos e intervalos para sincronização
        frame_time = 1.0 / self.fps if self.fps > 0 else 0.033  # ~30fps default
        
        # Ajustar o intervalo com base na capacidade de processamento
        # Quanto mais complexo o processamento, maior deve ser o intervalo
        processing_factor = 1.5  # Fator de ajuste para dar tempo ao processamento
        frame_interval_ms = int(frame_time * 1000 * processing_factor)  # Ajuste para dar tempo ao processamento
        
        # Definir um intervalo mínimo para garantir que o processamento tenha tempo suficiente
        min_interval_ms = 40  # 25 fps máximo
        frame_interval_ms = max(frame_interval_ms, min_interval_ms)
        
        print(f"Video FPS: {self.fps:.2f}, Frame interval: {frame_interval_ms}ms")
        
        # Buffer de frames processados para garantir uma reprodução suave
        self.frames_to_display.clear()
        self.next_frame_to_display = start_frame
        
        # Iniciar nosso timer de leitura de frames para um FPS estável
        self.frame_read_timer.setInterval(frame_interval_ms)
        self.frame_read_timer.start()
        
        # Iniciar timer para atualização do display 
        # Usar intervalo ligeiramente menor para garantir que exibição acompanhe o processamento
        display_interval_ms = int(frame_interval_ms * 0.9)
        self.frame_display_timer.setInterval(display_interval_ms)
        self.frame_display_timer.start()
        
        # Configurar timer para atualizar a barra de progresso da GPU
        self.gpu_memory_update_timer.setInterval(3000)  # Atualiza a cada 3 segundos
        self.gpu_memory_update_timer.start()
        
        # Iniciar timer de limpeza de memória
        self.memory_cleanup_timer.start()
        
        # Iniciar variáveis de temporização
        self.start_time = time.perf_counter()
        self.play_start_time = float(self.current_frame_number / self.fps)
        
        # Emitir sinal de início para atualização na UI
        self.video_processing_started.emit()
        
        # Não precisamos processar os frames aqui, pois os timers vão cuidar disso
        return

    def set_video_position(self, frame_position):
        """Define a posição do vídeo para um frame específico"""
        if self.media_capture and self.file_type == 'video':
            self.media_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
            self.current_frame_number = frame_position
    
    @profile_func(sort_by='cumulative')    
    def process_frame_swap(self, frame, frame_num=0):
        """Process a single frame for face swapping"""
        with measure_time("Face swap processing"):
            # Usar buffer pré-alocado em vez de cópia
            try:
                # Obter buffer do gerenciador de memória com as mesmas dimensões do frame
                processed_frame = memory_manager.get_buffer(frame.shape, 
                                                        torch.from_numpy(np.zeros(1, dtype=frame.dtype)).dtype,
                                                        "processed_frame_buffer")
                # Copiar o frame para o buffer
                np.copyto(processed_frame, frame)
            except Exception as e:
                # Fallback para cópia tradicional
                print(f"Erro ao alocar buffer para frame: {str(e)}")
                processed_frame = frame.copy()
            
            # Verificar se existem faces de entrada para usar na troca
            if not self.main_window.input_faces:
                # Apenas exibir a mensagem e retornar o frame original em vez de bloquear o processamento
                print("Nenhuma face de entrada disponível para troca. Exibindo frame original.")
                # Liberar buffer se não for usado
                if hasattr(processed_frame, '_is_from_pool') and processed_frame._is_from_pool:
                    memory_manager.release_buffer(processed_frame, "processed_frame_buffer")
                return frame.copy()
            
            # Get the parameters for face swapping
            models_processor = self.main_window.models_processor

            # Get the face detection parameters - usando valores padrão seguros se current_widget_parameters for None
            # Primeiro verificar se current_widget_parameters existe
            if not hasattr(self.main_window, 'current_widget_parameters') or self.main_window.current_widget_parameters is None:
                # Valores padrão seguros
                detect_mode = 'RetinaFace'
                score = 0.5  # 50%
                max_faces = 1
                input_size = 640
                use_landmark_detection = True
                landmark_detect_mode = '5'
                landmark_score = 0.5  # 50%
            else:
                # Obter valores dos parâmetros com fallbacks seguros
                detect_mode = self.main_window.current_widget_parameters.get('FaceDetectMode', 'RetinaFace')
                score = self.main_window.current_widget_parameters.get('FaceDetectScore', 50) / 100.0
                max_faces = self.main_window.current_widget_parameters.get('MaxFacesSlider', 1)
                input_size = self.main_window.current_widget_parameters.get('FaceDetectSizeSlider', 640)
                use_landmark_detection = self.main_window.current_widget_parameters.get('UseLandmarkDetectToggle', True)
                landmark_detect_mode = self.main_window.current_widget_parameters.get('FaceLandmarkDetectMode', '5')
                landmark_score = self.main_window.current_widget_parameters.get('FaceLandmarkScore', 50) / 100.0
            
            target_input_size = (input_size, input_size)
            
            # Detect faces in the frame
            with measure_time("Face detection"):
                try:
                    # Verificar o formato do frame e convertê-lo para tensor se necessário
                    # Usar um contexto de buffer para garantir liberação automática
                    with memory_manager.buffer_context((*frame.shape[:2], 3), 
                                                    torch.from_numpy(np.zeros(1, dtype=np.uint8)).dtype,
                                                    "detection_frame_buffer") as frame_buffer:
                        # Copiar o frame para o buffer
                        np.copyto(frame_buffer, processed_frame)
                        
                        # Converter para tensor
                        if isinstance(frame_buffer, np.ndarray):
                            # Converter de numpy para tensor do PyTorch
                            frame_tensor = torch.from_numpy(frame_buffer.astype('uint8')).to(models_processor.device)
                            # Se o formato é HWC (altura, largura, canais), converter para CHW (canais, altura, largura)
                            if frame_tensor.shape[-1] == 3:  # Se a última dimensão tem tamanho 3, é um formato HWC
                                frame_tensor = frame_tensor.permute(2, 0, 1)  # Converter para CHW
                        else:
                            # Já é um tensor, apenas garantir que está no dispositivo correto
                            frame_tensor = frame_buffer.to(models_processor.device)
                    
                    # Agora use o tensor para detecção
                    bboxes, kpss_5, kpss = models_processor.run_detect(
                        frame_tensor, 
                        detect_mode=detect_mode, 
                        max_num=max_faces, 
                        score=score, 
                        input_size=target_input_size,
                        use_landmark_detection=use_landmark_detection,
                        landmark_detect_mode=landmark_detect_mode,
                        landmark_score=landmark_score
                    )
                    
                    # Criar uma lista de dicionários com os dados de detecção
                    result = []
                    if len(kpss_5) > 0:
                        for i in range(len(kpss_5)):
                            face_data = {
                                'id': i,
                                'bbox': bboxes[i] if i < len(bboxes) else None,
                                'det_score': score,
                                'kps': kpss_5[i] if i < len(kpss_5) else None,
                                'lmk_score': landmark_score,
                                'world_kps': kpss[i] if i < len(kpss) else None
                            }
                            result.append(face_data)
                    
                except Exception as e:
                    print(f"Erro na detecção de face: {str(e)}")
                    # Liberar buffer antes de retornar
                    if hasattr(processed_frame, '_is_from_pool') and processed_frame._is_from_pool:
                        memory_manager.release_buffer(processed_frame, "processed_frame_buffer")
                    return frame.copy()
            
            # If no face is detected, return the original frame
            if not result:
                print("Nenhuma face detectada no frame. Exibindo frame original.")
                # Liberar buffer antes de retornar
                if hasattr(processed_frame, '_is_from_pool') and processed_frame._is_from_pool:
                    memory_manager.release_buffer(processed_frame, "processed_frame_buffer")
                return frame.copy()
            
            # Verificar se existem parâmetros definidos
            if not self.main_window.parameters:
                print("Nenhum parâmetro definido para faces detectadas. Execute 'Find Faces' primeiro.")
                # Liberar buffer antes de retornar
                if hasattr(processed_frame, '_is_from_pool') and processed_frame._is_from_pool:
                    memory_manager.release_buffer(processed_frame, "processed_frame_buffer")
                return frame.copy()
            
            # Flag para acompanhar se pelo menos uma face foi processada
            any_face_processed = False
            
            # Prepara listas para processamento em lote
            batch_images = []
            batch_embeddings = []
            face_locations = []  # Lista para armazenar informações de localização das faces
            face_modes = []  # Lista para armazenar os modos de face swapper
            
            # Registrar uso de modelos no gerenciador de memória
            memory_manager.register_model_usage('Inswapper128ArcFace')
            
            # Verificar periodicamente a memória durante processamento intensivo
            if frame_num % 10 == 0:  # A cada 10 frames
                self.check_memory_usage()
            
            # Coletando todas as faces a serem processadas em lote
            for target_face in result:
                with measure_time("Preparing faces for batch processing"):
                    bbox = target_face['bbox']
                    det_score = target_face['det_score']
                    
                    # Obter todos os keypoints disponíveis - usar o formato 5 pontos para reconhecimento facial padrão
                    kps = target_face['kps']  # 5 pontos faciais
                    
                    # Verificar se existem parâmetros para esta face específica
                    face_id = target_face['id']
                    if face_id >= len(self.main_window.parameters):
                        print(f"Nenhum parâmetro definido para face {face_id}. Pulando.")
                        continue
                    
                    face_params = self.main_window.parameters[face_id]
                    
                    # Verificar se a troca está habilitada para esta face
                    if not face_params.get('SwapEnabledToggle', True):
                        continue
                        
                    # Obter a entrada selecionada pelo usuário
                    input_index = face_params.get('SourceFaceDropdown', 0)
                    if input_index >= len(self.main_window.input_faces):
                        print(f"Índice de face de entrada inválido: {input_index}")
                        continue
                    
                    input_face = self.main_window.input_faces[input_index]
                    input_embedding = input_face.get('embedding')
                    
                    if input_embedding is None:
                        print("Embedding de face de entrada não disponível. Pulando.")
                        continue
                    
                    # Obter modo de processamento
                    face_swapper_mode = face_params.get('FaceSwapperMode', 'Inswapper')
                    
                    try:
                        # Preparar a face para swap - pré-processar para o formato correto
                        swap_image = self.prepare_face_for_swap(processed_frame, kps)
                        
                        # Adicionar à lista para processamento em lote
                        batch_images.append(swap_image)
                        batch_embeddings.append(input_embedding)
                        face_locations.append((bbox, kps))
                        face_modes.append(face_swapper_mode)
                        
                    except Exception as e:
                        print(f"Erro ao preparar face para troca: {str(e)}")
                        continue
            
            # Processar o lote de faces usando CUDA streams para paralelização (se disponível)
            if batch_images and batch_embeddings:
                try:
                    with measure_time("Batch face swap processing"):
                        # Se for Inswapper128 e houver mais de um rosto, usar batch real
                        if all(m == 'Inswapper' for m in face_modes) and len(batch_images) > 1:
                            swapped_faces = models_processor.face_swappers.run_inswapper_batch(batch_images, batch_embeddings)
                        # Se for SimSwap512 e houver mais de um rosto, usar batch real
                        elif all(m == 'SimSwap' for m in face_modes) and len(batch_images) > 1:
                            swapped_faces = models_processor.face_swappers.run_swapper_simswap512_batch(batch_images, batch_embeddings)
                        # Se for GhostFace e houver mais de um rosto, usar batch real
                        elif all(m.startswith('GhostFace') for m in face_modes) and len(batch_images) > 1:
                            swapped_faces = models_processor.face_swappers.run_swapper_ghostface_batch(batch_images, batch_embeddings, swapper_model=face_modes[0])
                        # Se for CSCS e houver mais de um rosto, usar batch real
                        elif all(m == 'CSCS' for m in face_modes) and len(batch_images) > 1:
                            swapped_faces = models_processor.face_swappers.run_swapper_cscs_batch(batch_images, batch_embeddings)
                        else:
                            # Usar o processador de modelos para processar faces em paralelo (streams)
                            swapped_faces = models_processor.face_swappers.process_faces_with_streams(
                                batch_images, batch_embeddings, face_modes[0], num_streams=self.num_threads)
                        # Integrar as faces trocadas de volta no frame
                        for i, swapped_face in enumerate(swapped_faces):
                            if swapped_face is not None:
                                bbox, kps = face_locations[i]
                                processed_frame = self.integrate_face_to_frame(processed_frame, swapped_face, bbox, kps)
                                any_face_processed = True
                except Exception as e:
                    print(f"Erro no processamento em lote: {str(e)}")
            
            # Se nenhuma face foi processada, apenas retornar o frame original
            if not any_face_processed:
                # Liberar buffer antes de retornar
                if hasattr(processed_frame, '_is_from_pool') and processed_frame._is_from_pool:
                    memory_manager.release_buffer(processed_frame, "processed_frame_buffer")
                return frame.copy()
            
            # Criar uma cópia do resultado para retornar, e liberar o buffer
            result_frame = processed_frame.copy()
            
            # Liberar buffer antes de retornar
            if hasattr(processed_frame, '_is_from_pool') and processed_frame._is_from_pool:
                memory_manager.release_buffer(processed_frame, "processed_frame_buffer")
                
            # Limpar cache periodicamente durante processamento de vídeo
            if frame_num % 50 == 0:  # A cada 50 frames
                models_processor.periodic_cleanup()
                
            return result_frame

    def process_next_frame(self):
        """Read the next frame and add it to the queue for processing."""

        if self.current_frame_number > self.max_frame_number:
            # print("Stopping frame_read_timer as all frames have been read!")
            self.frame_read_timer.stop()
            return

        if self.frame_queue.qsize() >= self.num_threads:
            # print(f"Queue is full ({self.frame_queue.qsize()} frames). Throttling frame reading.")
            return

        if self.file_type == 'video' and self.media_capture:
            ret, frame = misc_helpers.read_frame(self.media_capture, preview_mode = not self.recording)
            if ret:
                frame = frame[..., ::-1]  # Convert BGR to RGB
                # print(f"Enqueuing frame {self.current_frame_number}")
                self.frame_queue.put(self.current_frame_number)
                self.start_frame_worker(self.current_frame_number, frame)
                self.current_frame_number += 1
            else:
                print("Cannot read frame!", self.current_frame_number)
                self.stop_processing()
                self.main_window.display_messagebox_signal.emit('Error Reading Frame', f'Error Reading Frame {self.current_frame_number}.\n Stopped Processing...!', self.main_window)

    def start_frame_worker(self, frame_number, frame, is_single_frame=False):
        """Start a FrameWorker to process the given frame."""
        # Garantir que swapfacesButton esteja ativado durante a reprodução para manter o faceswap
        swap_faces_enabled = False
        if hasattr(self.main_window, 'swapfacesButton'):
            swap_faces_enabled = self.main_window.swapfacesButton.isChecked()
        
        # Se estamos reproduzindo (não é single_frame) e não foi solicitado o faceswap explicitamente,
        # garantir que seja ativado para a reprodução contínua se tivermos faces de entrada disponíveis
        if not is_single_frame and not swap_faces_enabled and hasattr(self.main_window, 'input_faces') and self.main_window.input_faces:
            # Ativar temporariamente o swap para o processamento deste frame
            temp_swap_enabled = True 
        else:
            temp_swap_enabled = False
        
        try:
            # Se necessário, ativar temporariamente o flag de swap faces
            if temp_swap_enabled:
                self.main_window.swapfacesButton.blockSignals(True)
                self.main_window.swapfacesButton.setChecked(True)
                self.main_window.swapfacesButton.blockSignals(False)
            
            # Criar e iniciar o worker
            worker = FrameWorker(frame, self.main_window, frame_number, self.frame_queue, is_single_frame)
            self.workers.append(worker)
            
            # Executar o worker
            if is_single_frame:
                worker.run()
            else:
                worker.start()
            
        finally:
            # Restaurar o estado original do botão se foi alterado temporariamente
            if temp_swap_enabled:
                self.main_window.swapfacesButton.blockSignals(True)
                self.main_window.swapfacesButton.setChecked(swap_faces_enabled)
                self.main_window.swapfacesButton.blockSignals(False)

    def process_current_frame(self):

        # print("\nCalled process_current_frame()",self.current_frame_number)
        # self.main_window.processed_frames.clear()

        self.next_frame_to_display = self.current_frame_number
        if self.file_type == 'video' and self.media_capture:
            ret, frame = misc_helpers.read_frame(self.media_capture, preview_mode=False)
            if ret:
                frame = frame[..., ::-1]  # Convert BGR to RGB
                # print(f"Enqueuing frame {self.current_frame_number}")
                self.frame_queue.put(self.current_frame_number)
                self.start_frame_worker(self.current_frame_number, frame, is_single_frame=True)
                
                self.media_capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number)
            else:
                print("Cannot read frame!", self.current_frame_number)
                self.main_window.display_messagebox_signal.emit('Error Reading Frame', f'Error Reading Frame {self.current_frame_number}.', self.main_window)

        # """Process a single image frame directly without queuing."""
        elif self.file_type == 'image':
            frame = misc_helpers.read_image_file(self.media_path)
            if frame is not None:

                frame = frame[..., ::-1]  # Convert BGR to RGB
                self.frame_queue.put(self.current_frame_number)
                # print("Processing current frame as image.")
                self.start_frame_worker(self.current_frame_number, frame, is_single_frame=True)
            else:
                print("Error: Unable to read image file.")

        # Handle webcam capture
        elif self.file_type == 'webcam':
            ret, frame = misc_helpers.read_frame(self.media_capture, preview_mode = False)
            if ret:
                frame = frame[..., ::-1]  # Convert BGR to RGB
                # print(f"Enqueuing frame {self.current_frame_number}")
                self.frame_queue.put(self.current_frame_number)
                self.start_frame_worker(self.current_frame_number, frame, is_single_frame=True)
            else:
                print("Unable to read Webcam frame!")
        self.join_and_clear_threads()

    def process_next_webcam_frame(self):
        # print("Called process_next_webcam_frame()")

        if self.frame_queue.qsize() >= self.num_threads:
            # print(f"Queue is full ({self.frame_queue.qsize()} frames). Throttling frame reading.")
            return
        if self.file_type == 'webcam' and self.media_capture:
            ret, frame = misc_helpers.read_frame(self.media_capture, preview_mode = False)
            if ret:
                frame = frame[..., ::-1]  # Convert BGR to RGB
                # print(f"Enqueuing frame {self.current_frame_number}")
                self.frame_queue.put(self.current_frame_number)
                self.start_frame_worker(self.current_frame_number, frame)

    # @misc_helpers.benchmark
    def stop_processing(self):
        """Para o processamento do vídeo"""
        # Definir flag para interromper o processamento
        self.stop_event.set()
        
        # Parar todos os timers
        self.stop_timers()
        
        # Verificar se estava processando antes
        was_processing = self.processing
        self.processing = False
        
        # Finalizar e liberar recursos
        self.join_and_clear_threads()
        
        # Limpar dicionários e filas com verificação de existência
        if hasattr(self, 'frames_to_display'):
            self.frames_to_display.clear()
        if hasattr(self, 'webcam_frames_to_display') and hasattr(self.webcam_frames_to_display, 'queue'):
            self.webcam_frames_to_display.queue.clear()
        
        # Finalizar gravação se estiver acontecendo
        if self.recording and self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            self.recording = False
            print(f"Vídeo salvo em: {self.output_filename}")
        
        # Liberar memória CUDA explicitamente para evitar vazamentos
        if torch.cuda.is_available():
            try:
                # Limpar cache do PyTorch
                torch.cuda.empty_cache()
                
                # Coletar lixo do Python para liberar objetos não referenciados
                gc.collect()
                
                # Forçar a sincronização de todos os streams CUDA
                torch.cuda.synchronize()
                
                # Registrar uso de memória para depuração
                if hasattr(self.main_window, 'models_processor') and hasattr(self.main_window.models_processor, 'print_gpu_memory_usage'):
                    self.main_window.models_processor.print_gpu_memory_usage("Após stop_processing")
            except Exception as e:
                print(f"Erro ao liberar memória CUDA: {str(e)}")
        
        # Emitir sinais de conclusão
        if was_processing:
            elapsed = time.perf_counter() - self.start_time if hasattr(self, 'start_time') and self.start_time > 0 else 0
            fps = self.frames_displayed / elapsed if elapsed > 0 else 0
            print(f"\nProcessamento concluído em {elapsed:.2f} segundos")
            print(f"FPS médio: {fps:.2f}")
            print(f"Frames processados: {self.frames_displayed}")
            
            self.video_processing_stopped.emit()
            self.processing_complete.emit(elapsed, fps)
            
        return was_processing

    def join_and_clear_threads(self):
        """Finaliza e limpa todos os workers em execução"""
        # Verificar se há workers ativos
        if hasattr(self, 'workers') and self.workers:
            for worker in self.workers:
                if hasattr(worker, 'join') and callable(worker.join) and hasattr(worker, 'is_alive'):
                    if worker.is_alive():
                        worker.join()
            self.workers.clear()
        # Para compatibilidade com código antigo
        elif hasattr(self, 'threads') and self.threads:
            for _, thread in self.threads.items():
                if hasattr(thread, 'is_alive') and thread.is_alive():
                    thread.join()
            self.threads.clear()
    
    def create_ffmpeg_subprocess(self):
        # Use Dimensions of the last processed frame as it could be different from the original frame due to restorers and frame enhancers 
        frame_height, frame_width, _ = self.current_frame.shape

        self.temp_file = r'temp_output.mp4'
        if Path(self.temp_file).is_file():
            os.remove(self.temp_file)

        args = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-f", "rawvideo",             # Specify raw video input
            "-pix_fmt", "bgr24",          # Pixel format of input frames
            "-s", f"{frame_width}x{frame_height}",  # Frame resolution
            "-r", str(self.fps),          # Frame rate
            "-i", "pipe:",                # Input from stdin
            "-vf", f"pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuvj420p",  # Padding and format conversion            
            "-c:v", "libx264",            # H.264 codec
            "-crf", "18",                 # Quality setting
            self.temp_file                # Output file
        ]

        self.recording_sp = subprocess.Popen(args, stdin=subprocess.PIPE)

    def enable_virtualcam(self, backend=False):
        #Check if capture contains any cv2 stream or is it an empty list
        if self.media_capture:
            if isinstance(self.current_frame, np.ndarray):
                frame_height, frame_width, _ = self.current_frame.shape
            else:
                frame_height = int(self.media_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frame_width = int(self.media_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.disable_virtualcam()
            try:
                backend = backend or self.main_window.control['VirtCamBackendSelection']
                # self.virtcam = pyvirtualcam.Camera(width=vid_width, height=vid_height, fps=int(self.fps), backend='unitycapture', device='Unity Video Capture')
                self.virtcam = pyvirtualcam.Camera(width=frame_width, height=frame_height, fps=int(self.fps), backend=backend, fmt=pyvirtualcam.PixelFormat.BGR)

            except Exception as e:
                print(e)

    def disable_virtualcam(self):
        if hasattr(self, 'virtcam') and self.virtcam:
            self.virtcam.close()
            self.virtcam = None

    def process_next_frame_from_timer(self):
        """Processa o próximo frame quando o timer dispara"""
        if not self.processing or self.stop_event.is_set():
            self.stop_timers()
            return
        
        # Verificar se alcançamos o fim do vídeo
        if self.current_frame_number >= self.max_frame_number:
            print("Fim do vídeo alcançado")
            self.stop_timers()
            self.stop_processing()
            return

        # Verificar se a fila está muito cheia
        if self.frame_queue.qsize() >= self.num_threads:
            # Não adicionar mais frames à fila se ela estiver cheia
            # Vamos aguardar o próximo ciclo
            return

        # Obter o próximo frame do vídeo
        status, frame = read_frame(self.media_capture, preview_mode=False)
        if not status or frame is None:
            print(f"Fim do vídeo ou erro ao ler o frame na posição {self.current_frame_number}")
            self.stop_timers()
            self.stop_processing()
            return
        
        try:
            # Importante: converter BGR para RGB para processamento
            frame_rgb = frame[..., ::-1].copy()  # BGR para RGB
            
            # Usar o sistema de workers para processar o frame de forma consistente
            # Este é o segredo: usar o mesmo mecanismo que é usado quando o vídeo está parado
            frame_number_snapshot = self.current_frame_number  # Snapshot do número atual
            
            # Adicionar à fila de processamento
            self.frame_queue.put(frame_number_snapshot)
            
            # IMPORTANTE: Usar o mesmo worker thread que é usado quando o vídeo está parado
            # Isso garante que o processamento seja idêntico
            self.start_frame_worker(frame_number_snapshot, frame_rgb)
            
            # Avançar para o próximo frame
            self.current_frame_number += 1
            
        except Exception as e:
            print(f"Erro ao processar frame {self.current_frame_number}: {str(e)}")
            import traceback
            traceback.print_exc()
            # Avançar para o próximo frame mesmo em caso de erro
            self.current_frame_number += 1

    def update_video_display(self):
        """Atualiza o display com o próximo frame a ser exibido"""
        if not self.processing:
            return
        
        # Verificar se há frames para mostrar
        if len(self.frames_to_display) == 0:
            return
        
        # Encontrar o próximo frame a ser exibido
        # Se não houver o próximo frame específico, mostrar o mais próximo disponível
        available_frames = sorted(self.frames_to_display.keys())
        
        # Se o next_frame_to_display não estiver disponível, encontrar o frame mais próximo
        if self.next_frame_to_display not in self.frames_to_display:
            # Encontrar o frame disponível mais próximo
            for frame_num in available_frames:
                if frame_num >= self.next_frame_to_display:
                    self.next_frame_to_display = frame_num
                    break
            else:
                # Se não houver frames futuros, manter o valor atual
                return
        
        # Obter o frame para exibição
        pixmap, frame = self.frames_to_display.pop(self.next_frame_to_display)
        self.current_frame = frame
        
        # Enviar o frame para a câmera virtual se necessário
        if hasattr(self, 'virtcam') and self.virtcam:
            self.send_frame_to_virtualcam(frame)
        
        # Gravar o frame se estiver gravando
        if self.recording and self.video_writer is not None:
            self.video_writer.write(frame)
        
        # Atualizar a interface com o frame atual
        graphics_view_actions.update_graphics_view(self.main_window, pixmap, self.next_frame_to_display)
        
        # Atualizar slider de vídeo sem disparar eventos
        if hasattr(self.main_window, 'videoSeekSlider'):
            self.main_window.videoSeekSlider.blockSignals(True)
            self.main_window.videoSeekSlider.setValue(self.next_frame_to_display)
            self.main_window.videoSeekSlider.blockSignals(False)
        
        # Atualizar rótulo de tempo se existir
        if hasattr(self.main_window, 'lblMediaTime'):
            current_time = float(self.next_frame_to_display / self.fps) if self.fps > 0 else 0
            duration = float(self.max_frame_number / self.fps) if self.fps > 0 else 0
            current_time_str = time.strftime('%M:%S', time.gmtime(current_time))
            duration_str = time.strftime('%M:%S', time.gmtime(duration))
            self.main_window.lblMediaTime.setText(f"{current_time_str} / {duration_str}")
        
        # Avançar para o próximo frame
        self.next_frame_to_display += 1
        
        # Calcular e atualizar FPS atual
        elapsed = time.perf_counter() - self.start_time
        if elapsed > 0:
            current_fps = self.frames_displayed / elapsed
            if hasattr(self.main_window, 'lblFPS'):
                self.main_window.lblFPS.setText(f"FPS: {current_fps:.1f}")

    def update_gpu_memory_stats(self):
        """Atualiza as estatísticas de uso de memória GPU"""
        if not self.processing:
            return
        
        try:
            # Registrar uso de memória GPU
            log_gpu_memory_usage(f"Durante o processamento de vídeo (frame {self.current_frame_number})")
            
            # Atualizar interface com estatísticas de GPU (se existir)
            common_widget_actions.update_gpu_memory_progressbar(self.main_window)
        except Exception as e:
            print(f"Erro ao atualizar estatísticas de GPU: {str(e)}")

    def stop_timers(self):
        """Para todos os timers em execução"""
        if hasattr(self, 'frame_read_timer') and self.frame_read_timer.isActive():
            self.frame_read_timer.stop()
        if hasattr(self, 'frame_display_timer') and self.frame_display_timer.isActive():
            self.frame_display_timer.stop()
        if hasattr(self, 'gpu_memory_update_timer') and self.gpu_memory_update_timer.isActive():
            self.gpu_memory_update_timer.stop()
        if hasattr(self, 'memory_cleanup_timer') and self.memory_cleanup_timer.isActive():
            self.memory_cleanup_timer.stop()

    def prepare_face_for_swap(self, frame, kps):
        """
        Prepara uma face para swap, recortando e transformando-a conforme necessário.
        
        Args:
            frame: Frame completo contendo a face
            kps: Keypoints da face
            
        Returns:
            Tensor da face preparada para swap
        """
        try:
            # Usar buffer pré-alocado para o tensor da face
            with memory_manager.buffer_context((3, 128, 128), torch.float32, "face_swap_buffer") as face_buffer:
                # Converter frame para tensor se for numpy array
                if isinstance(frame, np.ndarray):
                    # Usar buffer temporário para conversão, evitando allocação na GPU
                    with memory_manager.buffer_context((*frame.shape[:2], 3), 
                                                       torch.from_numpy(np.zeros(1, dtype=np.uint8)).dtype,
                                                       "frame_tensor_buffer") as frame_buffer:
                        # Copiar dados para o buffer
                        np.copyto(frame_buffer, frame)
                        # Converter para tensor
                        frame_tensor = torch.from_numpy(frame_buffer.astype('uint8')).to(self.main_window.models_processor.device)
                        if frame_tensor.shape[-1] == 3:  # HWC format
                            frame_tensor = frame_tensor.permute(2, 0, 1)  # Convert to CHW
                else:
                    # Já é tensor, garantir que está no dispositivo correto
                    frame_tensor = frame.to(self.main_window.models_processor.device)
                
                # Processar a face com o modelo ArcFace para obter uma face alinhada de 128x128
                # Usando o buffer pré-alocado como destino
                models_processor = self.main_window.models_processor
                try:
                    # Obter modelo ArcFace
                    arcface_model = 'Inswapper128ArcFace'  # Usar o modelo padrão
                    # Transformar a face para formato adequado usando o buffer pré-alocado
                    transformed_face = models_processor.face_swappers.recognize(
                        arcface_model, frame_tensor, kps, 'Opal', output_tensor=face_buffer
                    )
                    # Se transformação falhou, tentar com modelo default
                    if transformed_face is None:
                        transformed_face = models_processor.face_swappers.recognize(
                            'ResNet34ArcFace', frame_tensor, kps, 'ArcFace', output_tensor=face_buffer
                        )
                    
                    return transformed_face
                except Exception as e:
                    print(f"Erro no preparo da face: {str(e)}")
                    # Em caso de erro, retornar um tensor do frame recortado diretamente
                    return frame_tensor
        except Exception as e:
            print(f"Erro ao alocar buffer para prepare_face_for_swap: {str(e)}")
            # Fallback para método original sem buffer pré-alocado
            if isinstance(frame, np.ndarray):
                frame_tensor = torch.from_numpy(frame).to(self.main_window.models_processor.device)
                if frame_tensor.shape[-1] == 3:  # HWC format
                    frame_tensor = frame_tensor.permute(2, 0, 1)  # Convert to CHW
            else:
                frame_tensor = frame.clone().to(self.main_window.models_processor.device)
            
            return frame_tensor
        
    def integrate_face_to_frame(self, frame, face_result, bbox, kps):
        """
        Reintegra uma face processada de volta ao frame original.
        
        Args:
            frame: Frame original
            face_result: Face processada
            bbox: Bounding box da face
            kps: Keypoints da face
            
        Returns:
            Frame com a face substituída
        """
        # Cálculo de margens para um ajuste melhor
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Garantir que as coordenadas estejam dentro dos limites do frame
        height, width = frame.shape[:2] if isinstance(frame, np.ndarray) else frame.shape[1:]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        
        # Redimensionar a face processada para o tamanho da bbox
        face_height, face_width = y2 - y1, x2 - x1
        
        try:
            # Usar buffer pré-alocado para face redimensionada
            face_buffer_key = f"face_resize_buffer_{face_height}x{face_width}"
            
            # Converter face_result para numpy se for tensor
            if isinstance(face_result, torch.Tensor):
                if face_result.dim() == 4:  # Se formato [1, C, H, W]
                    face_result = face_result.squeeze(0)
                if face_result.shape[0] == 3:  # Se formato [C, H, W]
                    face_result = face_result.permute(1, 2, 0)  # Para [H, W, C]
                face_result = face_result.cpu().numpy()
            
            # Usar buffer pré-alocado para face redimensionada se possível
            with memory_manager.buffer_context((face_height, face_width, 3), 
                                             torch.from_numpy(np.zeros(1, dtype=np.uint8)).dtype,
                                             face_buffer_key) as face_resize_buffer:
                # Redimensionar para o buffer
                cv2.resize(face_result, (face_width, face_height), dst=face_resize_buffer)
                
                # Evitar cópia desnecessária do frame se possível
                # Se o frame já é um array numpy e não está sendo usado em outro lugar
                if isinstance(frame, np.ndarray):
                    # Substituir a região diretamente no frame
                    frame[y1:y2, x1:x2] = face_resize_buffer
                    return frame
                else:
                    # Preparar cópia do frame para modificação
                    with memory_manager.buffer_context(frame.shape, 
                                                    frame.dtype, 
                                                    "result_frame_buffer") as result_buffer:
                        # Copiar o frame para o buffer
                        np.copyto(result_buffer, frame)
                        
                        # Substituir região da face
                        if isinstance(result_buffer, np.ndarray):
                            result_buffer[y1:y2, x1:x2] = face_resize_buffer
                        else:
                            # Se o frame for um tensor
                            if result_buffer.shape[0] == 3:  # CHW format
                                # Converter face_resized para CHW
                                face_tensor = torch.from_numpy(face_resize_buffer).to(result_buffer.device)
                                if face_tensor.shape[-1] == 3:  # Se for HWC
                                    face_tensor = face_tensor.permute(2, 0, 1)  # Para CHW
                                result_buffer[:, y1:y2, x1:x2] = face_tensor
                            else:
                                # Se for HWC
                                face_tensor = torch.from_numpy(face_resize_buffer).to(result_buffer.device)
                                result_buffer[y1:y2, x1:x2] = face_tensor
                                
                        return result_buffer
                        
        except Exception as e:
            print(f"Erro ao usar buffer pré-alocado para integrate_face_to_frame: {str(e)}")
            # Fallback para método original em caso de erro
            
            # Se face_result for um tensor, converter para numpy
            if isinstance(face_result, torch.Tensor):
                if face_result.dim() == 4:  # Se formato [1, C, H, W]
                    face_result = face_result.squeeze(0)
                if face_result.shape[0] == 3:  # Se formato [C, H, W]
                    face_result = face_result.permute(1, 2, 0)  # Para [H, W, C]
                face_result = face_result.cpu().numpy()
                
            # Redimensionar para o tamanho da região da face
            face_resized = cv2.resize(face_result, (face_width, face_height))
            
            # Preparar cópia do frame para modificação
            result_frame = frame.copy()
            
            # Substituir região da face
            if isinstance(result_frame, np.ndarray):
                result_frame[y1:y2, x1:x2] = face_resized
            else:
                # Se o frame for um tensor
                if result_frame.shape[0] == 3:  # CHW format
                    # Converter face_resized para CHW
                    face_tensor = torch.from_numpy(face_resized).to(result_frame.device)
                    if face_tensor.shape[-1] == 3:  # Se for HWC
                        face_tensor = face_tensor.permute(2, 0, 1)  # Para CHW
                    result_frame[:, y1:y2, x1:x2] = face_tensor
                else:
                    # Se for HWC
                    face_tensor = torch.from_numpy(face_resized).to(result_frame.device)
                    result_frame[y1:y2, x1:x2] = face_tensor
                    
            return result_frame

    def load_frame_from_persistent_cache(self, frame_number):
        if self.media_path:
            return load_frame_from_cache(frame_number, self.media_path, format=DEFAULT_CACHE_FORMAT)
        return None