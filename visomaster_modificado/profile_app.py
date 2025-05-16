import sys
import os
import argparse
import cProfile
import pstats
import io
from datetime import datetime

# Adicionar o diretório atual ao path para importações
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executa o VisoMaster com perfilamento de desempenho")
    parser.add_argument("--output", "-o", type=str, default=None, 
                       help="Arquivo de saída para os resultados do perfilamento")
    parser.add_argument("--function", "-f", type=str, default=None,
                       help="Função específica a perfilar (deixe em branco para perfilar a aplicação inteira)")
    parser.add_argument("--memory-only", "-m", action="store_true",
                       help="Apenas monitora o uso de memória sem perfilamento de CPU")
    parser.add_argument("--duration", "-d", type=int, default=60,
                       help="Duração do perfilamento em segundos (para monitores)")
    
    args = parser.parse_args()
    
    # Configurar saída para o profiler
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"profile_results/visomaster_profile_{timestamp}.txt"
    
    # Importar apenas quando necessário para não afetar o tempo de inicialização
    from app.helpers import profiler
    
    # Se for apenas monitoramento de memória
    if args.memory_only:
        print(f"Iniciando monitoramento de memória por {args.duration} segundos...")
        from main import main
        
        # Iniciar monitoramento de memória
        stop_event = profiler.start_memory_monitor(interval=2.0)
        
        # Executar a aplicação
        main()
        
        # O monitoramento continuará até o programa ser encerrado
        
    # Se for perfilamento de função específica
    elif args.function:
        print(f"Perfilando função: {args.function}")
        # Este modo requer modificação manual do código para decorar
        # a função específica com @profiler.profile_func()
        from main import main
        main()
        
    # Perfilamento completo da aplicação
    else:
        print("Iniciando perfilamento completo da aplicação...")
        profiler.clear_profiling_data()
        
        # Usar cProfile para perfilar a execução completa
        profile = cProfile.Profile()
        profile.enable()
        
        # Importar e executar o ponto de entrada da aplicação
        from main import main
        main()
        
        profile.disable()
        
        # Salvar resultados
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        s = io.StringIO()
        ps = pstats.Stats(profile, stream=s).sort_stats('cumulative')
        ps.print_stats(100)  # Mostra as top 100 funções
        
        with open(args.output, 'w') as f:
            f.write(s.getvalue())
        
        print(f"Perfil salvo em: {args.output}") 