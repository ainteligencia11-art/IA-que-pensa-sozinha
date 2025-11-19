from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, time

modelo_dir = r"C:\Users\igor.cicale\OneDrive - EcoPower Energia Solar\Área de Trabalho\projeto ia\Phi-3-mini-4k-instruct"

print("🔄 Carregando modelo Phi-3-mini-4k-instruct...")

tokenizer = AutoTokenizer.from_pretrained(modelo_dir)

# Detecta GPU automaticamente
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    modelo_dir,
    device_map=None,
    torch_dtype=dtype
).to(device)

print(f"✅ Modelo carregado no dispositivo: {device.upper()}")

while True:
    entrada = input("\n🗣️ Você: ")
    if entrada.lower() in ["sair", "exit", "quit"]:
        print("👋 Encerrando conversa.")
        break

    inputs = tokenizer(entrada, return_tensors="pt").to(device)

    print("🤔 Gerando resposta...")
    t0 = time.time()

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.8,
            top_p=0.9
        )

    tempo = time.time() - t0
    resposta = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"🤖 Phi-3 ({tempo:.1f}s): {resposta}")

