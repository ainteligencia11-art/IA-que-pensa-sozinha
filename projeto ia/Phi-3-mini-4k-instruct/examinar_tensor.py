import json
from safetensors import safe_open
import torch

# Carregar o índice
with open('model.safetensors.index.json', 'r') as f:
    index = json.load(f)

# Escolher um tensor pequeno para testar
tensor_name = "model.layers.0.input_layernorm.weight"
file_path = index['weight_map'][tensor_name]

print(f"📁 Carregando tensor '{tensor_name}' do arquivo '{file_path}'")

# Carregar o tensor
with safe_open(file_path, framework="pt") as f:
    tensor = f.get_tensor(tensor_name)

print(f"📊 Forma do tensor: {tensor.shape}")
print(f"📊 Tipo de dados: {tensor.dtype}")
print(f"📊 Valores (amostra): {tensor[:5] if len(tensor) > 5 else tensor}")

# Estatísticas básicas
print(f"📈 Estatísticas:")
print(f"   Mínimo: {tensor.min().item()}")
print(f"   Máximo: {tensor.max().item()}")
print(f"   Média: {tensor.float().mean().item()}")
print(f"   Desvio padrão: {tensor.float().std().item()}")