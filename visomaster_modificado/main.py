import os
import sys
import argparse
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon
from PySide6.QtCore import Qt

from app.ui import main_ui
from PySide6 import QtWidgets 
import qdarktheme
from app.ui.core.proxy_style import ProxyStyle

# Adicionamos a opção de perfilamento
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='VisoMaster - Face Swap and Edit Application')
    parser.add_argument('--profile', action='store_true', help='Enable performance profiling')
    parser.add_argument('--profile-output', type=str, default='profile_results/main_profile.txt',
                        help='Output file for profiling results')
    parser.add_argument('--memory-monitor', action='store_true', 
                        help='Enable memory monitoring (logs memory usage periodically)')
    args = parser.parse_args()

    # Configure high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("visomaster.ico"))
    
    # Iniciar monitoramento de memória se solicitado
    memory_monitor_thread = None
    if args.memory_monitor:
        from app.helpers.profiler import start_memory_monitor, clear_profiling_data
        clear_profiling_data()  # Limpa dados de perfilamento antigos
        print("Iniciando monitoramento de memória...")
        memory_monitor_thread = start_memory_monitor(interval=5.0)
    
    # Aplicar perfilamento se solicitado
    if args.profile:
        import cProfile
        import pstats
        import io
        from app.helpers.profiler import clear_profiling_data
        
        print(f"Perfilamento ativado. Resultados serão salvos em: {args.profile_output}")
        clear_profiling_data()  # Limpa dados de perfilamento antigos
        
        # Criar diretório de saída se não existir
        os.makedirs(os.path.dirname(args.profile_output), exist_ok=True)
        
        # Iniciar perfilamento
        profiler = cProfile.Profile()
        profiler.enable()
        
        # Iniciar a aplicação
        main_window = main_ui.MainWindow()
        main_window.show()
        exit_code = app.exec()
        
        # Parar perfilamento e salvar resultados
        profiler.disable()
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(100)  # Mostrar as top 100 funções
        
        with open(args.profile_output, 'w') as f:
            f.write(s.getvalue())
        
        print(f"Perfilamento concluído. Resultados salvos em: {args.profile_output}")
        return exit_code
    else:
        # Execução normal sem perfilamento
        main_window = main_ui.MainWindow()
        main_window.show()
        return app.exec()

if __name__ == '__main__':
    sys.exit(main())