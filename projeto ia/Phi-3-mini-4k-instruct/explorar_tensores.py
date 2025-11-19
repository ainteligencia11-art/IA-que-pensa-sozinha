import json
import os
from safetensors import safe_open

# Carregar o índice dos tensores
with open('model.safetensors.index.json', 'r') as f:
    index = json.load(f)

print("📊 ESTRUTURA DE TENSORES:")
print(f"Total de tensores: {len(index['weight_map'])}")

# Ver distribuição entre os arquivos
file_distribution = {}
for tensor_name, file_path in index['weight_map'].items():
    file_distribution[file_path] = file_distribution.get(file_path, 0) + 1

for file_path, count in file_distribution.items():
    print(f"📁 {file_path}: {count} tensores")

# Amostra de alguns tensores importantes
print("\n🔍 TENSORES PRINCIPAIS:")
sample_tensors = [
    name for name in index['weight_map'].keys() 
    if any(key in name for key in ['embed', 'lm_head', 'layers.0', 'layers.31'])
][:10]

for tensor in sample_tensors:
    print(f"  - {tensor}")

# Vamos também ver o tamanho dos arquivos de tensores
print("\n📏 TAMANHO DOS ARQUIVOS:")
for file_path in file_distribution.keys():
    file_size = os.path.getsize(file_path) / (1024**3)  # em GB
    print(f"  {file_path}: {file_size:.2f} GB")