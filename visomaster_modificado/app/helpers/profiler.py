import os
import time
import cProfile
import pstats
import io
import functools
import threading
import datetime
from contextlib import contextmanager
import gc
import torch
import numpy as np
import psutil

# Configurações do perfilador
PROFILE_DIR = "profile_results"
LOG_FILE = os.path.join(PROFILE_DIR, "performance_log.txt")
ENABLE_PROFILING = True  # Pode ser alterado para desabilitar o perfilamento em produção

# Garantir que o diretório de perfis exista
os.makedirs(PROFILE_DIR, exist_ok=True)

def get_memory_usage():
    """Obtém informações de uso de memória RAM e VRAM"""
    # Memória RAM
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / (1024 * 1024)  # MB
    
    # Memória VRAM (se disponível)
    vram_usage = 0
    vram_total = 0
    try:
        if torch.cuda.is_available():
            vram_usage = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)  # MB
    except:
        pass
        
    return {
        "ram_usage_mb": ram_usage,
        "vram_usage_mb": vram_usage,
        "vram_total_mb": vram_total
    }

@contextmanager
def measure_time(name, log_to_file=True):
    """Medidor de tempo para código em bloco usando with"""
    if not ENABLE_PROFILING:
        yield
        return
        
    start_time = time.time()
    mem_before = get_memory_usage()
    
    try:
        yield
    finally:
        end_time = time.time()
        elapsed = end_time - start_time
        mem_after = get_memory_usage()
        
        mem_diff = {
            "ram": mem_after["ram_usage_mb"] - mem_before["ram_usage_mb"],
            "vram": mem_after["vram_usage_mb"] - mem_before["vram_usage_mb"]
        }
        
        message = (f"PERF: {name} - Tempo: {elapsed:.4f}s | "
                  f"RAM delta: {mem_diff['ram']:.2f}MB | "
                  f"VRAM delta: {mem_diff['vram']:.2f}MB")
        
        print(message)
        
        if log_to_file:
            with open(LOG_FILE, "a") as f:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] {message}\n")

def profile_func(output_file=None, lines_to_print=20, sort_by='cumulative'):
    """Decorador para perfilar funções individuais usando cProfile"""
    def inner(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not ENABLE_PROFILING:
                return func(*args, **kwargs)
                
            # Nome do arquivo de saída baseado no nome da função se não for fornecido
            nonlocal output_file
            if output_file is None:
                output_file = os.path.join(PROFILE_DIR, f"{func.__name__}_profile.txt")
                
            # Executa a função com cProfile
            profiler = cProfile.Profile()
            try:
                profiler.enable()
                result = func(*args, **kwargs)
                profiler.disable()
                
                # Formata e salva os resultados
                s = io.StringIO()
                ps = pstats.Stats(profiler, stream=s).sort_stats(sort_by)
                ps.print_stats(lines_to_print)
                
                with open(output_file, 'w') as f:
                    f.write(s.getvalue())
                    
                # Opcional: imprimir na tela também
                # print(f"Perfil para {func.__name__} salvo em {output_file}")
                
                return result
            finally:
                # Garantir que o profiler seja desabilitado mesmo em caso de exceção
                if profiler is not None:
                    profiler.disable()
        return wrapper
    return inner

def start_memory_monitor(interval=5.0):
    """Inicia um thread para monitorar o uso de memória em intervalos regulares"""
    stop_event = threading.Event()
    
    def monitor_memory():
        while not stop_event.is_set():
            mem_info = get_memory_usage()
            message = (f"MONITOR: RAM: {mem_info['ram_usage_mb']:.2f}MB | "
                      f"VRAM: {mem_info['vram_usage_mb']:.2f}MB / {mem_info['vram_total_mb']:.2f}MB")
            
            print(message)
            with open(LOG_FILE, "a") as f:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] {message}\n")
                
            # Forçar coleta de lixo para ver se há vazamentos de memória
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            stop_event.wait(interval)
    
    monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
    monitor_thread.start()
    
    return stop_event  # O evento pode ser usado para interromper o monitoramento

def analyze_call_frequencies(log_file=LOG_FILE):
    """Analisa o arquivo de log para identificar funções chamadas com mais frequência"""
    if not os.path.exists(log_file):
        return "Arquivo de log não encontrado."
        
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # Extrai nomes de funções e contagens
    func_counts = {}
    for line in lines:
        if "PERF:" in line:
            parts = line.split(" - ")[0].split("PERF: ")
            if len(parts) > 1:
                func_name = parts[1].strip()
                func_counts[func_name] = func_counts.get(func_name, 0) + 1
    
    # Ordena por frequência
    sorted_funcs = sorted(func_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Cria relatório
    report = "Análise de Frequência de Chamadas:\n"
    report += "-" * 40 + "\n"
    for func, count in sorted_funcs[:20]:  # Top 20
        report += f"{func}: {count} chamadas\n"
    
    return report

def profile_section(app, section_name):
    """Perfila uma seção inteira da aplicação (use com 'with')"""
    profiler = cProfile.Profile()
    output_file = os.path.join(PROFILE_DIR, f"{section_name}_profile.txt")
    
    def start():
        if ENABLE_PROFILING:
            profiler.enable()
        return profiler
    
    def stop():
        if ENABLE_PROFILING and profiler is not None:
            profiler.disable()
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            ps.print_stats(30)  # Top 30 funções
            
            with open(output_file, 'w') as f:
                f.write(s.getvalue())
            
            print(f"Perfil para {section_name} salvo em {output_file}")
    
    return start, stop

def log_gpu_memory_usage(message=""):
    """Registra o uso atual de memória GPU"""
    if not ENABLE_PROFILING:
        return
        
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        reserved = torch.cuda.memory_reserved() / (1024 * 1024)  # MB
        
        log_msg = f"GPU Memory [{message}] - Allocated: {allocated:.2f}MB, Reserved: {reserved:.2f}MB"
        print(log_msg)
        
        with open(LOG_FILE, "a") as f:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {log_msg}\n")

def clear_profiling_data():
    """Limpa dados de profiling anteriores (use com cuidado)"""
    for filename in os.listdir(PROFILE_DIR):
        if filename.endswith('_profile.txt'):
            try:
                os.remove(os.path.join(PROFILE_DIR, filename))
            except:
                pass
                
    # Recria o arquivo de log
    with open(LOG_FILE, 'w') as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] Profiling data cleared\n") 