import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Carregar o tokenizer
tokenizer = AutoTokenizer.from_pretrained("./Phi-3-mini-4k-instruct")

# Carregar a configuração do modelo
from transformers import Phi3Config
config = Phi3Config.from_pretrained("./Phi-3-mini-4k-instruct")
print("Configuração do modelo:")
print(config)

# Tentar carregar o modelo com baixo uso de memória
model = AutoModelForCausalLM.from_pretrained(
    "./Phi-3-mini-4k-instruct",
    torch_dtype=torch.float16,  # Usar meia precisão para economizar memória
    device_map="auto",          # Usar CPU (já que não temos GPU poderosa)
    trust_remote_code=True      # Necessário para modelos Phi-3
)

# Inspecionar os tensores
print("\n🔍 Estrutura do modelo (primeiros 10 tensores):")
for name, param in model.named_parameters():
    print(f"{name}: {param.shape} {param.dtype}")
    if 'weight' in name:
        print(f"   Exemplo de valores: {param.data[0][:5]}")
    break  # Só mostra os primeiros 10 para não sobrecarregar

# Verificar o número total de parâmetros
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal de parâmetros: {total_params:,}")