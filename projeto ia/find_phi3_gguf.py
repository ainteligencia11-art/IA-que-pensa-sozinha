# find_phi3_gguf.py
import os
import struct

def find_phi3_gguf():
    """Encontra o arquivo GGUF do Phi-3 mini"""
    blobs_dir = r"C:\Users\igor.cicale\.ollama\models\blobs"
    
    print("🔍 Procurando arquivo GGUF do Phi-3 mini...")
    
    gguf_files = []
    for root, dirs, files in os.walk(blobs_dir):
        for file in files:
            if file.startswith("sha256-"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'rb') as f:
                        magic = f.read(4)
                        if magic == b'GGUF':
                            file_size = os.path.getsize(file_path)
                            # Phi-3 mini deve ter ~2.3 GB
                            if 2.0 <= file_size / (1024**3) <= 2.5:
                                gguf_files.append((file_path, file_size))
                                print(f"✅ Possível Phi-3 mini: {file}")
                                print(f"   Tamanho: {file_size / (1024**3):.2f} GB")
                except:
                    pass
    
    return gguf_files

files = find_phi3_gguf()
if files:
    print(f"\n🎯 Encontrado: {len(files)} arquivo(s)")
    for path, size in files:
        print(f"📁 {os.path.basename(path)}")
        print(f"📏 {size/(1024**3):.2f} GB")
else:
    print("❌ Nenhum arquivo GGUF encontrado")