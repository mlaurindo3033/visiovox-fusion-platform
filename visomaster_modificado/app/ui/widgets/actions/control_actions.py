from typing import TYPE_CHECKING
import torch
from PySide6 import QtWidgets 

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow
from app.ui.widgets.actions import common_actions as common_widget_actions

#'''
#    Define functions here that has to be executed when value of a control widget (In the settings tab) is changed.
#    The first two parameters should be the MainWindow object and the new value of the control 
#'''

def change_execution_provider(main_window: 'MainWindow', new_provider):
    main_window.video_processor.stop_processing()
    main_window.models_processor.switch_providers_priority(new_provider)
    main_window.models_processor.clear_gpu_memory()
    common_widget_actions.update_gpu_memory_progressbar(main_window)

def change_threads_number(main_window: 'MainWindow', new_threads_number):
    main_window.video_processor.set_number_of_threads(new_threads_number)
    torch.cuda.empty_cache()
    common_widget_actions.update_gpu_memory_progressbar(main_window)

def set_video_playback_fps(main_window: 'MainWindow', set_video_fps=False):
    # print("Called set_video_playback_fps()")
    if set_video_fps and main_window.video_processor.media_capture:
        main_window.parameter_widgets['VideoPlaybackCustomFpsSlider'].set_value(main_window.video_processor.fps)

def toggle_virtualcam(main_window: 'MainWindow', toggle_value=False):
    video_processor = main_window.video_processor
    if toggle_value:
        video_processor.enable_virtualcam()
    else:
        video_processor.disable_virtualcam()

def enable_virtualcam(main_window: 'MainWindow', backend):
    print('backend', backend)
    main_window.video_processor.enable_virtualcam(backend=backend)

def run_benchmark_threads(main_window):
    import time
    from PySide6.QtWidgets import QMessageBox
    threads_to_test = [2, 4, 8, 12, 16]
    results = []
    video_processor = main_window.video_processor
    original_threads = video_processor.num_threads
    test_frames = 30  # Número de frames para testar

    # Verifica se há vídeo carregado
    if not hasattr(video_processor, 'media_capture') or video_processor.media_capture is None:
        QMessageBox.warning(main_window, 'Benchmark Threads', 'Nenhum vídeo carregado. Por favor, carregue um vídeo antes de rodar o benchmark.')
        return

    # Salva a posição original do vídeo
    original_pos = video_processor.media_capture.get(1) if video_processor.media_capture else 0

    for n in threads_to_test:
        video_processor.set_number_of_threads(n)
        # Volta para o início do vídeo
        video_processor.media_capture.set(1, 0)
        frames_processed = 0
        start = time.time()
        while frames_processed < test_frames:
            ret, frame = video_processor.media_capture.read()
            if not ret:
                break  # Fim do vídeo
            # Simula processamento real (pode ser substituído por um passo do pipeline)
            _ = frame.copy()
            frames_processed += 1
        elapsed = time.time() - start
        fps = frames_processed / elapsed if elapsed > 0 else 0
        results.append((n, round(fps, 2)))
    # Restaura a posição original do vídeo
    if video_processor.media_capture:
        video_processor.media_capture.set(1, original_pos)
    video_processor.set_number_of_threads(original_threads)
    msg = '\n'.join([f'{n} threads: {fps} FPS' for n, fps in results])
    best = max(results, key=lambda x: x[1])
    msg += f"\n\nMelhor configuração: {best[0]} threads ({best[1]} FPS)"
    QMessageBox.information(main_window, 'Benchmark Threads', f'Resultados do benchmark (real):\n{msg}')