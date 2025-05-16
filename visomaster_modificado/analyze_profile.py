import os
import sys
import argparse
import pstats
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

PROFILE_DIR = "profile_results"
LOG_FILE = os.path.join(PROFILE_DIR, "performance_log.txt")

def analyze_cprofile_file(profile_file, top_n=20):
    """Analisa e exibe os resultados de um arquivo de perfilamento cProfile"""
    if not os.path.exists(profile_file):
        print(f"Arquivo {profile_file} não encontrado.")
        return None
    
    # Carregar estatísticas
    p = pstats.Stats(profile_file)
    
    # Ordenar por tempo cumulativo
    p.sort_stats('cumulative')
    
    # Criar um dicionário com os dados das principais funções
    func_stats = {}
    for func, (cc, nc, tt, ct, callers) in p.stats.items():
        if nc > 0:  # Filtrar apenas funções que foram chamadas
            module_name = func[0].split("\\")[-1] if len(func) > 0 else "unknown"
            func_name = func[2] if len(func) > 2 else "unknown"
            full_name = f"{module_name}:{func_name}"
            
            func_stats[full_name] = {
                'calls': nc,
                'total_time': tt,
                'cumulative_time': ct,
                'time_per_call': tt / nc if nc > 0 else 0
            }
    
    # Ordenar pelo tempo cumulativo e pegar o top_n
    sorted_funcs = sorted(func_stats.items(), key=lambda x: x[1]['cumulative_time'], reverse=True)[:top_n]
    
    return sorted_funcs

def analyze_time_log(log_file=LOG_FILE, top_n=20):
    """Analisa o arquivo de log de tempo para encontrar operações mais demoradas"""
    if not os.path.exists(log_file):
        print(f"Arquivo de log {log_file} não encontrado.")
        return None
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # Extrai informações de tempo das linhas de log
    time_data = {}
    for line in lines:
        if "PERF:" in line:
            try:
                operation = line.split("PERF:")[1].split(" - ")[0].strip()
                time_str = line.split("Tempo:")[1].split("s")[0].strip()
                time_value = float(time_str)
                
                if operation not in time_data:
                    time_data[operation] = {
                        'times': [],
                        'ram_deltas': [],
                        'vram_deltas': []
                    }
                
                time_data[operation]['times'].append(time_value)
                
                # Extrair deltas de RAM e VRAM se presentes
                if "RAM delta:" in line and "VRAM delta:" in line:
                    ram_delta = float(line.split("RAM delta:")[1].split("MB")[0].strip())
                    vram_delta = float(line.split("VRAM delta:")[1].split("MB")[0].strip())
                    time_data[operation]['ram_deltas'].append(ram_delta)
                    time_data[operation]['vram_deltas'].append(vram_delta)
            except:
                continue
    
    # Calcular estatísticas para cada operação
    operation_stats = {}
    for op, data in time_data.items():
        times = data['times']
        if not times:
            continue
            
        operation_stats[op] = {
            'count': len(times),
            'avg_time': sum(times) / len(times),
            'min_time': min(times),
            'max_time': max(times),
            'total_time': sum(times)
        }
        
        # Adicionar estatísticas de RAM/VRAM se disponíveis
        if data['ram_deltas']:
            operation_stats[op]['avg_ram_delta'] = sum(data['ram_deltas']) / len(data['ram_deltas'])
            operation_stats[op]['avg_vram_delta'] = sum(data['vram_deltas']) / len(data['vram_deltas'])
    
    # Ordenar pelo tempo total e pegar o top_n
    sorted_ops = sorted(operation_stats.items(), key=lambda x: x[1]['total_time'], reverse=True)[:top_n]
    
    return sorted_ops

def analyze_memory_usage(log_file=LOG_FILE):
    """Analisa o uso de memória ao longo do tempo com base nos logs"""
    if not os.path.exists(log_file):
        print(f"Arquivo de log {log_file} não encontrado.")
        return None
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # Extrair informações de monitoramento de memória
    timestamps = []
    ram_usage = []
    vram_usage = []
    vram_total = []
    
    for line in lines:
        if "MONITOR:" in line:
            try:
                # Extrair timestamp
                timestamp_str = line.split("[")[1].split("]")[0]
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                
                # Extrair valores de RAM e VRAM
                ram_value = float(line.split("RAM:")[1].split("MB")[0].strip())
                vram_value = float(line.split("VRAM:")[1].split("/")[0].strip())
                vram_total_value = float(line.split("/")[1].split("MB")[0].strip())
                
                timestamps.append(timestamp)
                ram_usage.append(ram_value)
                vram_usage.append(vram_value)
                vram_total.append(vram_total_value)
            except:
                continue
        elif "GPU Memory" in line:
            try:
                # Extrair timestamp
                timestamp_str = line.split("[")[1].split("]")[0]
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                
                # Extrair valores de VRAM
                allocated = float(line.split("Allocated:")[1].split("MB")[0].strip())
                reserved = float(line.split("Reserved:")[1].split("MB")[0].strip())
                
                # Adicionar ao gráfico com um marcador diferente
                timestamps.append(timestamp)
                vram_usage.append(allocated)
                vram_total.append(reserved)
            except:
                continue
    
    return timestamps, ram_usage, vram_usage, vram_total

def plot_memory_usage(timestamps, ram_usage, vram_usage, vram_total):
    """Plota o uso de memória ao longo do tempo"""
    if not timestamps:
        print("Sem dados de memória para plotar.")
        return
    
    # Converter timestamps para valores numéricos
    time_values = [(t - timestamps[0]).total_seconds() for t in timestamps]
    
    plt.figure(figsize=(12, 8))
    
    # Plotar RAM
    plt.subplot(2, 1, 1)
    plt.plot(time_values, ram_usage, 'b-', linewidth=2, label='RAM Usage (MB)')
    plt.title('RAM Usage Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('RAM (MB)')
    plt.grid(True)
    plt.legend()
    
    # Plotar VRAM
    plt.subplot(2, 1, 2)
    plt.plot(time_values, vram_usage, 'r-', linewidth=2, label='VRAM Usage (MB)')
    if vram_total:
        plt.plot(time_values, vram_total, 'g--', linewidth=1, label='VRAM Total (MB)')
    plt.title('VRAM Usage Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('VRAM (MB)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(PROFILE_DIR, 'memory_usage.png'))
    plt.close()
    
    print(f"Gráfico de uso de memória salvo em {os.path.join(PROFILE_DIR, 'memory_usage.png')}")

def plot_top_operations(sorted_ops):
    """Plota as operações mais demoradas"""
    if not sorted_ops:
        print("Sem dados de operações para plotar.")
        return
    
    operations = [op[0] if len(op[0]) < 30 else op[0][:27] + "..." for op in sorted_ops]
    total_times = [op[1]['total_time'] for op in sorted_ops]
    avg_times = [op[1]['avg_time'] for op in sorted_ops]
    
    # Ordenar por tempo total
    idx = np.argsort(total_times)
    operations = [operations[i] for i in idx]
    total_times = [total_times[i] for i in idx]
    avg_times = [avg_times[i] for i in idx]
    
    # Plotar gráfico de barras
    fig, ax1 = plt.subplots(figsize=(12, 10))
    
    y_pos = np.arange(len(operations))
    
    ax1.barh(y_pos, total_times, align='center', alpha=0.7, color='b', label='Total Time (s)')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(operations)
    ax1.invert_yaxis()  # Labels de baixo para cima
    ax1.set_xlabel('Time (seconds)')
    ax1.set_title('Top Operations by Total Time')
    
    # Adicionar eixo secundário para tempo médio
    ax2 = ax1.twiny()
    ax2.barh(y_pos, avg_times, align='center', alpha=0.4, color='r', label='Avg Time (s)')
    ax2.set_xlabel('Average Time per Call (seconds)')
    
    # Adicionar legendas
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PROFILE_DIR, 'top_operations.png'))
    plt.close()
    
    print(f"Gráfico de operações mais demoradas salvo em {os.path.join(PROFILE_DIR, 'top_operations.png')}")

def create_html_report(profile_file, log_file=LOG_FILE):
    """Cria um relatório HTML com os resultados da análise"""
    # Analisar dados
    cprofile_data = analyze_cprofile_file(profile_file)
    time_log_data = analyze_time_log(log_file)
    memory_data = analyze_memory_usage(log_file)
    
    # Gerar gráficos
    if memory_data and memory_data[0]:
        plot_memory_usage(*memory_data)
    
    if time_log_data:
        plot_top_operations(time_log_data)
    
    # Gerar relatório HTML
    html_file = os.path.join(PROFILE_DIR, 'profile_report.html')
    
    with open(html_file, 'w') as f:
        f.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>VisoMaster Performance Analysis</title>
            <style>
                body { 
                    font-family: Arial, sans-serif; 
                    margin: 20px; 
                    background-color: #f5f5f5;
                }
                h1, h2, h3 { 
                    color: #333;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }
                th, td {
                    padding: 8px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }
                th {
                    background-color: #4CAF50;
                    color: white;
                }
                tr:hover {
                    background-color: #f5f5f5;
                }
                .graph {
                    margin: 20px 0;
                    text-align: center;
                }
                .graph img {
                    max-width: 100%;
                    box-shadow: 0 0 5px rgba(0,0,0,0.2);
                }
                .summary {
                    background-color: #e9f7ef;
                    padding: 15px;
                    margin-bottom: 20px;
                    border-left: 5px solid #4CAF50;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>VisoMaster Performance Analysis</h1>
                <p>Report generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
                
                <div class="summary">
                    <h2>Summary of Findings</h2>
                    <p>This report provides an analysis of the performance bottlenecks in the VisoMaster application.</p>
                </div>
                
                <h2>Memory Usage</h2>
        """)
        
        # Adicionar gráficos de memória
        if os.path.exists(os.path.join(PROFILE_DIR, 'memory_usage.png')):
            f.write("""
                <div class="graph">
                    <img src="memory_usage.png" alt="Memory Usage Graph">
                </div>
            """)
        
        # Adicionar gráficos de operações
        f.write("""
                <h2>Top Time-Consuming Operations</h2>
        """)
        
        if os.path.exists(os.path.join(PROFILE_DIR, 'top_operations.png')):
            f.write("""
                <div class="graph">
                    <img src="top_operations.png" alt="Top Operations Graph">
                </div>
            """)
        
        # Tabela de funções mais demoradas (cProfile)
        f.write("""
                <h2>Top Functions (cProfile)</h2>
                <table>
                    <tr>
                        <th>Function</th>
                        <th>Calls</th>
                        <th>Total Time (s)</th>
                        <th>Cumulative Time (s)</th>
                        <th>Time per Call (s)</th>
                    </tr>
        """)
        
        if cprofile_data:
            for func_name, stats in cprofile_data:
                f.write(f"""
                    <tr>
                        <td>{func_name}</td>
                        <td>{stats['calls']}</td>
                        <td>{stats['total_time']:.4f}</td>
                        <td>{stats['cumulative_time']:.4f}</td>
                        <td>{stats['time_per_call']:.6f}</td>
                    </tr>
                """)
        
        f.write("""
                </table>
                
                <h2>Top Operations (Time Logs)</h2>
                <table>
                    <tr>
                        <th>Operation</th>
                        <th>Count</th>
                        <th>Total Time (s)</th>
                        <th>Avg Time (s)</th>
                        <th>Min Time (s)</th>
                        <th>Max Time (s)</th>
                    </tr>
        """)
        
        if time_log_data:
            for op_name, stats in time_log_data:
                f.write(f"""
                    <tr>
                        <td>{op_name}</td>
                        <td>{stats['count']}</td>
                        <td>{stats['total_time']:.4f}</td>
                        <td>{stats['avg_time']:.4f}</td>
                        <td>{stats['min_time']:.4f}</td>
                        <td>{stats['max_time']:.4f}</td>
                    </tr>
                """)
        
        f.write("""
                </table>
                
                <h2>Recommendations</h2>
                <ul>
                    <li>Focus optimization efforts on the top time-consuming functions</li>
                    <li>Consider parallelizing operations where appropriate</li>
                    <li>Monitor memory usage to identify potential leaks</li>
                    <li>Optimize model loading and unloading to reduce VRAM usage</li>
                </ul>
            </div>
        </body>
        </html>
        """)
    
    print(f"Relatório HTML gerado em: {html_file}")
    return html_file

def main():
    parser = argparse.ArgumentParser(description='Analyze VisoMaster performance profile data')
    parser.add_argument('--profile', type=str, default=None, 
                        help='cProfile output file to analyze (default: most recent in profile_results)')
    parser.add_argument('--log', type=str, default=LOG_FILE, 
                        help='Time measurement log file (default: profile_results/performance_log.txt)')
    parser.add_argument('--top', type=int, default=20, 
                        help='Number of top functions/operations to display')
    
    args = parser.parse_args()
    
    # Se nenhum arquivo de perfil for especificado, use o mais recente
    if args.profile is None:
        profile_files = glob.glob(os.path.join(PROFILE_DIR, "*_profile.txt"))
        if not profile_files:
            print("Nenhum arquivo de perfil encontrado em", PROFILE_DIR)
            return
        
        # Ordenar por data de modificação (mais recente primeiro)
        profile_files.sort(key=os.path.getmtime, reverse=True)
        args.profile = profile_files[0]
    
    # Criar diretório de saída
    os.makedirs(PROFILE_DIR, exist_ok=True)
    
    # Criar relatório HTML
    report_file = create_html_report(args.profile, args.log)
    
    # Abrir o relatório no navegador padrão
    if os.path.exists(report_file):
        import webbrowser
        webbrowser.open('file://' + os.path.realpath(report_file))

if __name__ == "__main__":
    main() 