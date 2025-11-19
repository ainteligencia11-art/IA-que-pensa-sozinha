import json

with open('config.json', 'r') as f:
    config = json.load(f)

print("🔧 CONFIGURAÇÃO DO PHI-3:")
print(f"Arquitetura: {config.get('architectures', ['N/A'])[0]}")
print(f"Hidden layers: {config.get('num_hidden_layers', 'N/A')}")
print(f"Attention heads: {config.get('num_attention_heads', 'N/A')}")
print(f"Hidden size: {config.get('hidden_size', 'N/A')}")
print(f"Vocab size: {config.get('vocab_size', 'N/A')}")