# cleanup_proper.py
import os
import subprocess

def proper_cleanup():
    print("🧹 FAZENDO LIMPEZA COMPLETA E CORRETA...")
    
    # 1. Parar Ollama
    print("⏹️  Parando Ollama...")
    os.system("taskkill /f /im ollama.exe 2>nul")
    
    # 2. Listar e remover todos os modelos
    print("🗑️  Removendo modelos...")
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=30)
        lines = result.stdout.strip().split('\n')
        
        # Pular cabeçalho e processar cada modelo
        for line in lines[1:]:  # Pular "NAME" header
            if line.strip():
                model_name = line.split()[0]  # Primeira coluna é o nome
                if model_name and model_name != "NAME":
                    print(f"   Removendo: {model_name}")
                    os.system(f'ollama rm "{model_name}" >nul 2>&1')
    except:
        print("   ⚠️  Não foi possível listar modelos (Ollama pode estar parado)")
    
    print("✅ Limpeza concluída!")

proper_cleanup()