import json

def collect_keys_from_dynamic_widgets(main_window):
    keys = set()
    if hasattr(main_window, 'dynamic_widgets'):
        for widget in main_window.dynamic_widgets:
            key = getattr(widget, 'translation_key', None)
            if key:
                keys.add(key)
    return sorted(keys)

if __name__ == '__main__':
    # Exemplo de uso: rode este script dentro do contexto do seu app
    # e passe a instância do main_window
    # from app.ui.main_ui import MainWindow
    # main_window = MainWindow()
    # keys = collect_keys_from_dynamic_widgets(main_window)
    # print(json.dumps(keys, ensure_ascii=False, indent=2))
    print('Este script deve ser usado dentro do contexto do app, importando e chamando collect_keys_from_dynamic_widgets(main_window)') 