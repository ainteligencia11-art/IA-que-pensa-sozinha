from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Caminho local do modelo
modelo_dir = r"C:\Users\igor.cicale\OneDrive - EcoPower Energia Solar\Área de Trabalho\projeto ia\Phi-3-mini-4k-instruct"

# Carrega tokenizer e modelo
print("🔄 Carregando modelo Phi-3-mini-4k-instruct...")
tokenizer = AutoTokenizer.from_pretrained(modelo_dir)
model = AutoModelForCausalLM.from_pretrained(modelo_dir, torch_dtype=torch.bfloat16, device_map="auto")

print("✅ Modelo carregado. Pronto para conversar.")

while True:
    entrada = input("\n🗣️ Você: ")
    if entrada.lower() in ["sair", "exit", "quit"]:
        print("👋 Encerrando conversa.")
        break

    inputs = tokenizer(entrada, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
    resposta = tokenizer.decode(output[0], skip_special_tokens=True)

    print(f"🤖 Phi-3: {resposta}")
