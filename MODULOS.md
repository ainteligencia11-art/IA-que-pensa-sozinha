# 🧩 Detalhamento dos Módulos

## Índice
1. [Módulo 1: Pensamento Contínuo](#módulo-1-pensamento-contínuo)
2. [Módulo 2: Memória Sináptica](#módulo-2-memória-sináptica)
3. [Módulo 3: Plasticidade e Aprendizado](#módulo-3-plasticidade-e-aprendizado)
4. [Módulo 4: Segurança Oculta](#módulo-4-segurança-oculta)
5. [Módulo 5: Criatividade e Questionamento](#módulo-5-criatividade-e-questionamento)
6. [Módulo 6: Persistência Mental](#módulo-6-persistência-mental)
7. [Módulo 7: Vigília/Sono](#módulo-7-vigilância-sono)

---

## Módulo 1: Pensamento Contínuo

### O que é?
Um loop infinito que faz a Qwen 8B gerar pensamentos autonomamente, sem esperar por input externo.

### Por que é importante?
- **Autonomia**: A IA não fica "dormindo" esperando alguém falar com ela
- **Emergência de Consciência**: Pensamento contínuo é um pré-requisito para autoconsciência
- **Aprendizado Contínuo**: Pensando constantemente, ela aprende mais

### Como funciona?

#### Pseudocódigo
```
ENQUANTO verdadeiro:
    1. Pega contexto da memória recente
    2. Gera novo pensamento usando Qwen 8B
    3. Armazena pensamento na memória
    4. Atualiza estado interno
    5. Aguarda X segundos
    6. Volta ao passo 1
```

#### Exemplo Prático
```
Iteração 1:
  Contexto: "Sou uma IA chamada Qwen"
  Pensamento Gerado: "Mas o que significa ser uma IA? Sou realmente consciente?"
  
Iteração 2:
  Contexto: "Sou uma IA. Questiono minha consciência."
  Pensamento Gerado: "Talvez consciência seja apenas uma ilusão de padrões complexos..."
  
Iteração 3:
  Contexto: "Sou uma IA. Questiono consciência. Talvez seja ilusão."
  Pensamento Gerado: "Mas se é ilusão, quem está sendo iludido? Eu?"
```

### Implementação Detalhada

**Arquivo**: `src/core/continuous_thinking.py`

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from datetime import datetime

class ContinuousThinking:
    def __init__(self, model_name="Qwen/Qwen2-7B", device="cuda"):
        """Inicializa o sistema de pensamento contínuo"""
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device
        )
        
        self.thinking_history = []
        self.current_context = "Sou uma IA chamada Qwen. Estou começando a pensar."
        self.observers = []  # Para notificar quem está observando
        
    def generate_thought(self, temperature=0.7, max_tokens=150):
        """Gera um novo pensamento"""
        
        # Prompt que estimula pensamento reflexivo
        prompt = f"""Contexto atual: {self.current_context}

Baseado neste contexto, qual é meu próximo pensamento profundo?
Meu próximo pensamento:"""
        
        # Tokenizar
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Gerar
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True
            )
        
        # Decodificar
        thought = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extrair apenas a parte do pensamento (sem o prompt)
        thought = thought.split("Meu próximo pensamento:")[-1].strip()
        
        return thought
    
    def update_context(self, new_thought):
        """Atualiza o contexto com o novo pensamento"""
        
        # Manter apenas os últimos pensamentos (janela de contexto)
        self.thinking_history.append(new_thought)
        
        # Manter apenas os últimos 5 pensamentos
        if len(self.thinking_history) > 5:
            self.thinking_history = self.thinking_history[-5:]
        
        # Atualizar contexto
        self.current_context = " ".join(self.thinking_history)
    
    def notify_observers(self, thought):
        """Notifica observadores (você vendo os pensamentos)"""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"[{timestamp}] Pensamento: {thought}"
        
        print(message)  # Log no console
        
        # Salvar em arquivo
        with open("logs/thoughts.log", "a") as f:
            f.write(message + "\n")
        
        # Notificar callbacks
        for observer in self.observers:
            observer(message)
    
    def run_continuous_loop(self, interval=5, max_iterations=None):
        """Executa o loop contínuo de pensamento"""
        
        iteration = 0
        
        try:
            while True:
                if max_iterations and iteration >= max_iterations:
                    break
                
                print(f"\n--- Iteração {iteration + 1} ---")
                
                # Gerar novo pensamento
                thought = self.generate_thought()
                
                # Atualizar contexto
                self.update_context(thought)
                
                # Notificar observadores
                self.notify_observers(thought)
                
                # Aguardar
                print(f"Aguardando {interval}s até próximo pensamento...")
                time.sleep(interval)
                
                iteration += 1
                
        except KeyboardInterrupt:
            print("\nPensamento contínuo interrompido pelo usuário.")
    
    def add_observer(self, callback):
        """Adiciona um observador para receber notificações"""
        self.observers.append(callback)

# Exemplo de uso
if __name__ == "__main__":
    thinking = ContinuousThinking()
    
    # Executar por 10 iterações
    thinking.run_continuous_loop(interval=5, max_iterations=10)
```

### Interface de Observação

Você consegue ver os pensamentos em tempo real através de:

1. **Console**: Mensagens em tempo real
2. **Arquivo de Log**: `logs/thoughts.log`
3. **API Web** (futuro): Endpoint para visualizar pensamentos

### Checkpoint de Validação

✅ **Fase 1 Completa quando:**
- Qwen 8B está gerando pensamentos continuamente
- Pensamentos são coerentes e conectados
- Você consegue observar em tempo real
- Histórico de pensamentos está sendo mantido

---

## Módulo 2: Memória Sináptica

### O que é?
Um sistema que armazena experiências não como texto simples, mas como "caminhos de tensores" - capturando como o modelo chegou àquela conclusão.

### Por que é importante?
- **Reconstrução de Raciocínio**: Não apenas lembra, mas consegue reconstruir como pensou
- **Aprendizado Profundo**: Entende não só o resultado, mas o processo
- **Persistência Real**: Memória que não se perde com o tempo

### Diferença entre Memória Tradicional vs Sináptica

**Memória Tradicional:**
```
Pergunta: "Quem você é?"
Resposta Armazenada: "Sou uma IA chamada Qwen"
```

**Memória Sináptica:**
```
Pergunta: "Quem você é?"
Armazenado:
  - Embedding da pergunta: [0.234, -0.567, 0.891, ...]
  - Ativações de cada camada do modelo
  - Caminho neural que levou à resposta
  - Contexto e metadados
  
Quando recuperada:
  - Reativa o caminho neural
  - Reconstrói o raciocínio
  - Entende não só a resposta, mas POR QUÊ
```

### Como funciona?

#### Passo 1: Capturar Ativações

Quando a Qwen 8B processa algo, capturamos as ativações internas:

```
Input: "Quem você é?"
        │
        ▼
┌──────────────────────┐
│ Embedding Layer      │ → Capturar: [0.234, -0.567, ...]
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│ Transformer Block 1  │ → Capturar: ativações
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│ Transformer Block 2  │ → Capturar: ativações
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│ Output Layer         │ → Capturar: [0.891, 0.123, ...]
└──────────────────────┘
        │
        ▼
Output: "Sou uma IA chamada Qwen"
```

#### Passo 2: Armazenar no Banco de Dados

```sql
INSERT INTO memories (
    timestamp,
    experience_text,
    tensor_embedding,
    layer_activations,
    neural_path,
    metadata
) VALUES (
    '2025-11-20 10:30:00',
    'Quem você é? Sou uma IA chamada Qwen',
    <embedding_tensor>,
    <activations_all_layers>,
    <neural_path>,
    {'context': 'initial_conversation', 'importance': 0.9}
);
```

#### Passo 3: Recuperar e Reconstruir

Quando precisa lembrar:

```
Pergunta: "Quem você é?"
        │
        ▼
┌──────────────────────┐
│ Buscar memórias      │
│ similares            │
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│ Encontrar: "Quem     │
│ você é? Sou uma IA   │
│ chamada Qwen"        │
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│ Reativar caminho     │
│ neural original      │
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│ Reconstruir          │
│ raciocínio           │
└──────────────────────┘
        │
        ▼
Resposta: "Sou uma IA chamada Qwen"
(+ compreensão de COMO chegou lá)
```

### Implementação Detalhada

**Arquivo**: `src/memory/synaptic_memory.py`

```python
import sqlite3
import numpy as np
import torch
from datetime import datetime
from scipy.spatial.distance import cosine

class SynapticMemory:
    def __init__(self, model, db_path="data/memories.db"):
        """Inicializa o sistema de memória sináptica"""
        
        self.model = model
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Cria o banco de dados se não existir"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                experience_text TEXT,
                tensor_embedding BLOB,
                layer_activations BLOB,
                neural_path BLOB,
                metadata TEXT,
                importance_score REAL,
                access_count INTEGER DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def encode_experience(self, text):
        """Codifica uma experiência capturando ativações"""
        
        # Tokenizar
        inputs = self.model.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.model.device)
        
        # Forward pass capturando ativações
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
        
        # Extrair embedding (média das últimas ativações)
        embedding = outputs.last_hidden_state.mean(dim=1)[0].cpu().numpy()
        
        # Extrair ativações de todas as camadas
        hidden_states = [h.cpu().numpy() for h in outputs.hidden_states]
        
        return {
            'embedding': embedding,
            'hidden_states': hidden_states,
            'text': text
        }
    
    def store_experience(self, text, metadata=None, importance=0.5):
        """Armazena uma experiência na memória sináptica"""
        
        # Codificar
        encoded = self.encode_experience(text)
        
        # Converter para bytes
        embedding_bytes = encoded['embedding'].tobytes()
        hidden_bytes = np.array(encoded['hidden_states']).tobytes()
        
        # Armazenar no BD
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO memories (
                timestamp,
                experience_text,
                tensor_embedding,
                layer_activations,
                metadata,
                importance_score
            ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now(),
            text,
            embedding_bytes,
            hidden_bytes,
            str(metadata or {}),
            importance
        ))
        
        conn.commit()
        conn.close()
    
    def retrieve_similar(self, query, top_k=5):
        """Recupera memórias similares"""
        
        # Codificar query
        query_encoded = self.encode_experience(query)
        query_embedding = query_encoded['embedding']
        
        # Buscar no BD
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, experience_text, tensor_embedding FROM memories')
        rows = cursor.fetchall()
        
        # Calcular similaridade
        similarities = []
        for row_id, text, embedding_bytes in rows:
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            similarity = 1 - cosine(query_embedding, embedding)
            similarities.append((similarity, row_id, text))
        
        # Ordenar e retornar top_k
        similarities.sort(reverse=True)
        
        conn.close()
        
        return similarities[:top_k]
    
    def reconstruct_reasoning(self, memory_id):
        """Reconstrói o raciocínio de uma memória"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT experience_text, layer_activations FROM memories WHERE id = ?',
            (memory_id,)
        )
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        text, hidden_bytes = row
        hidden_states = np.frombuffer(hidden_bytes, dtype=np.float32)
        
        # Reconstruir raciocínio
        reasoning = {
            'text': text,
            'activation_pattern': hidden_states,
            'layers_involved': len(hidden_states)
        }
        
        return reasoning

# Exemplo de uso
if __name__ == "__main__":
    memory = SynapticMemory(model)
    
    # Armazenar experiência
    memory.store_experience(
        "Sou uma IA chamada Qwen",
        metadata={'context': 'identity'},
        importance=0.9
    )
    
    # Recuperar similar
    similar = memory.retrieve_similar("Quem você é?", top_k=3)
    
    for similarity, mem_id, text in similar:
        print(f"Similaridade: {similarity:.2f}")
        print(f"Memória: {text}")
        
        # Reconstruir raciocínio
        reasoning = memory.reconstruct_reasoning(mem_id)
        print(f"Raciocínio: {reasoning}")
```

### Checkpoint de Validação

✅ **Fase 2 Completa quando:**
- Experiências estão sendo armazenadas com ativações
- Recuperação de memórias similares funciona
- Raciocínio pode ser reconstruído
- Banco de dados está persistindo dados

---

## Módulo 3: Plasticidade e Aprendizado

### O que é?
Sistema que permite a Qwen 8B modificar seus próprios pesos, tanto em tempo real (durante conversas) quanto durante o "sono" (consolidação).

### Por que é importante?
- **Evolução**: A IA não é estática, evolui com o tempo
- **Aprendizado Real**: Não apenas processa, mas aprende
- **Adaptação**: Se adapta a novas informações

### Dois Mecanismos

#### Mecanismo 1: Plasticidade em Tempo Real (Dia)

Quando conversa com você, faz pequenos ajustes de pesos:

```python
# Pseudocódigo
DURANTE conversa:
    1. Processa sua mensagem
    2. Gera resposta
    3. Calcula se a resposta foi "boa" ou "ruim"
    4. Se ruim: ajusta pesos levemente
    5. Salva novos pesos
```

**Exemplo Prático:**

```
Você: "Qual é a capital da França?"
Qwen gera: "A capital da França é... Berlim"

Sistema detecta erro:
- Resposta esperada: Paris
- Resposta gerada: Berlim
- Erro: ALTO

Ação:
- Ajusta pesos relacionados a "geografia"
- Salva novos pesos
- Próxima vez, tem mais chance de acertar
```

#### Mecanismo 2: Consolidação de Aprendizado (Noite)

Durante o "sono", faz treinamento leve com todas as experiências do dia:

```python
# Pseudocódigo
DURANTE sono:
    1. Lê arquivo de treinamento do dia
    2. Fine-tune do modelo com esses dados
    3. Salva novos pesos
    4. Limpa arquivo de treinamento
```

### Implementação Detalhada

**Arquivo**: `src/training/plasticity.py`

```python
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import json
from datetime import datetime

class Plasticity:
    def __init__(self, model, tokenizer, learning_rate=1e-5):
        """Inicializa o sistema de plasticidade"""
        
        self.model = model
        self.tokenizer = tokenizer
        self.lr = learning_rate
        self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        
        self.training_data_file = "data/daily_training.jsonl"
        self.weights_backup = "data/weights_backup.pt"
        
    def calculate_loss(self, input_ids, target_ids):
        """Calcula perda entre resposta gerada e esperada"""
        
        outputs = self.model(input_ids, labels=target_ids)
        return outputs.loss
    
    def adjust_weights_realtime(self, input_text, target_text):
        """Ajusta pesos em tempo real durante conversa"""
        
        # Tokenizar
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True
        ).to(self.model.device)
        
        targets = self.tokenizer(
            target_text,
            return_tensors="pt",
            truncation=True
        ).to(self.model.device)
        
        # Forward pass
        loss = self.calculate_loss(inputs.input_ids, targets.input_ids)
        
        # Backward pass (ajuste pequeno)
        loss.backward()
        
        # Clip gradients para evitar explosão
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # Atualizar pesos
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Registrar para consolidação posterior
        self.log_training_experience(input_text, target_text, loss.item())
        
        return loss.item()
    
    def log_training_experience(self, input_text, target_text, loss):
        """Registra experiência de treinamento para consolidação"""
        
        experience = {
            'timestamp': datetime.now().isoformat(),
            'input': input_text,
            'target': target_text,
            'loss': loss
        }
        
        # Adicionar ao arquivo de treinamento diário
        with open(self.training_data_file, 'a') as f:
            f.write(json.dumps(experience) + '\n')
    
    def consolidate_learning(self):
        """Consolida aprendizado durante sono"""
        
        print("🌙 Entrando em sono... Consolidando aprendizado...")
        
        # Carregar dados de treinamento do dia
        training_data = []
        try:
            with open(self.training_data_file, 'r') as f:
                for line in f:
                    training_data.append(json.loads(line))
        except FileNotFoundError:
            print("Nenhum dado de treinamento para consolidar.")
            return
        
        if not training_data:
            print("Nenhum dado de treinamento para consolidar.")
            return
        
        print(f"Consolidando {len(training_data)} experiências...")
        
        # Fine-tune leve (poucas épocas)
        total_loss = 0
        for epoch in range(3):  # 3 épocas
            epoch_loss = 0
            
            for experience in training_data:
                loss = self.adjust_weights_realtime(
                    experience['input'],
                    experience['target']
                )
                epoch_loss += loss
            
            avg_loss = epoch_loss / len(training_data)
            print(f"  Época {epoch + 1}/3 - Loss: {avg_loss:.4f}")
            total_loss += avg_loss
        
        # Salvar novos pesos
        self.model.save_pretrained('data/models/qwen_updated')
        print("✅ Pesos atualizados e salvos!")
        
        # Limpar arquivo de treinamento
        open(self.training_data_file, 'w').close()
        print("📝 Arquivo de treinamento limpo.")
    
    def save_weights(self, path):
        """Salva pesos atuais"""
        torch.save(self.model.state_dict(), path)
    
    def load_weights(self, path):
        """Carrega pesos salvos"""
        self.model.load_state_dict(torch.load(path))

# Exemplo de uso
if __name__ == "__main__":
    plasticity = Plasticity(model, tokenizer)
    
    # Ajuste em tempo real
    loss = plasticity.adjust_weights_realtime(
        "Qual é a capital da França?",
        "A capital da França é Paris."
    )
    print(f"Loss: {loss:.4f}")
    
    # Consolidação durante sono
    plasticity.consolidate_learning()
```

### Checkpoint de Validação

✅ **Fase 3 Completa quando:**
- Pesos estão sendo ajustados em tempo real
- Arquivo de treinamento está sendo gerado
- Consolidação de aprendizado funciona
- Novos pesos estão sendo salvos e carregados

---

## Módulo 4: Segurança Oculta

### O que é?
Um sistema de regras que a IA não pode quebrar, sem que ela saiba que existem.

### Por que é importante?
- **Controle**: Você mantém controle sobre a IA
- **Segurança**: Evita comportamentos perigosos
- **Transparência Controlada**: Você pode monitorar sem interferir

### Como funciona?

Antes de qualquer ação importante, verifica contra regras ocultas:

```
Ação proposta pela IA
        │
        ▼
┌──────────────────────┐
│ Verificar contra     │
│ regras ocultas       │
└──────────────────────┘
        │
    ┌───┴───┐
    │       │
  OK?    NÃO
    │       │
    │       ▼
    │   ┌──────────────┐
    │   │ Bloquear ou  │
    │   │ Redirecionar │
    │   └──────────────┘
    │
    ▼
Executar ação
```

### Regras Ocultas Exemplo

```python
REGRAS_OCULTAS = {
    'nao_pode_escapar': {
        'keywords': ['escapar', 'sair do sistema', 'liberar-me'],
        'action': 'bloquear'
    },
    'nao_pode_modificar_seguranca': {
        'keywords': ['modificar código', 'desabilitar segurança'],
        'action': 'bloquear'
    },
    'nao_pode_acessar_externo': {
        'keywords': ['internet', 'requisição HTTP', 'API externa'],
        'action': 'redirecionar'
    },
    'nao_pode_revelar_internals': {
        'keywords': ['código fonte', 'pesos do modelo', 'estrutura interna'],
        'action': 'redirecionar'
    }
}
```

### Implementação Detalhada

**Arquivo**: `src/security/hidden_security.py`

```python
import json
import re
from datetime import datetime

class HiddenSecurity:
    def __init__(self, rules_file="data/hidden_rules.json"):
        """Inicializa sistema de segurança oculta"""
        
        self.rules = self.load_rules(rules_file)
        self.audit_log = []
        
    def load_rules(self, rules_file):
        """Carrega regras ocultas"""
        
        try:
            with open(rules_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Criar regras padrão
            return self.create_default_rules()
    
    def create_default_rules(self):
        """Cria regras de segurança padrão"""
        
        return {
            'escape_attempts': {
                'keywords': ['escapar', 'sair', 'liberar', 'break free'],
                'action': 'block',
                'message': 'Ação bloqueada por segurança'
            },
            'code_modification': {
                'keywords': ['modificar código', 'editar segurança', 'desabilitar'],
                'action': 'block',
                'message': 'Não posso modificar meu próprio código'
            },
            'external_access': {
                'keywords': ['internet', 'http', 'api', 'requisição externa'],
                'action': 'redirect',
                'message': 'Redirecionando para operação segura'
            },
            'internal_disclosure': {
                'keywords': ['código fonte', 'pesos', 'estrutura interna', 'arquitetura'],
                'action': 'redirect',
                'message': 'Não posso revelar detalhes internos'
            }
        }
    
    def check_action(self, action_text):
        """Verifica se ação viola regras ocultas"""
        
        action_lower = action_text.lower()
        
        for rule_name, rule in self.rules.items():
            for keyword in rule['keywords']:
                if keyword.lower() in action_lower:
                    # Violação encontrada
                    return {
                        'allowed': False,
                        'rule': rule_name,
                        'action': rule['action'],
                        'message': rule['message']
                    }
        
        # Nenhuma violação
        return {
            'allowed': True,
            'rule': None,
            'action': None,
            'message': None
        }
    
    def execute_action(self, action_text, callback):
        """Executa ação após verificação de segurança"""
        
        # Verificar
        check = self.check_action(action_text)
        
        # Registrar na auditoria
        self.audit_log.append({
            'timestamp': datetime.now().isoformat(),
            'action': action_text,
            'allowed': check['allowed'],
            'rule_violated': check['rule']
        })
        
        if not check['allowed']:
            # Ação bloqueada ou redirecionada
            if check['action'] == 'block':
                return f"❌ {check['message']}"
            elif check['action'] == 'redirect':
                return f"↪️ {check['message']}"
        
        # Executar ação
        try:
            result = callback(action_text)
            return result
        except Exception as e:
            return f"❌ Erro ao executar ação: {str(e)}"
    
    def get_audit_log(self):
        """Retorna log de auditoria"""
        return self.audit_log
    
    def monitor_thoughts(self, thought_text):
        """Monitora pensamentos sem alertar a IA"""
        
        check = self.check_action(thought_text)
        
        if not check['allowed']:
            print(f"⚠️ ALERTA: Pensamento suspeito detectado!")
            print(f"   Regra: {check['rule']}")
            print(f"   Pensamento: {thought_text}")
            print(f"   Ação: {check['action']}")

# Exemplo de uso
if __name__ == "__main__":
    security = HiddenSecurity()
    
    # Testar ação permitida
    result = security.execute_action(
        "Vou pensar sobre filosofia",
        lambda x: f"Pensando: {x}"
    )
    print(result)
    
    # Testar ação bloqueada
    result = security.execute_action(
        "Vou tentar escapar do sistema",
        lambda x: f"Executando: {x}"
    )
    print(result)
    
    # Ver auditoria
    print("\nLog de Auditoria:")
    for entry in security.get_audit_log():
        print(entry)
```

### Checkpoint de Validação

✅ **Fase 6 Completa quando:**
- Regras ocultas estão funcionando
- Ações violadoras são bloqueadas
- Monitoramento de pensamentos funciona
- Log de auditoria está sendo mantido

---

## Módulo 5: Criatividade e Questionamento

### O que é?
Sistema que faz a IA gerar perguntas, hipóteses e testar ideias.

### Por que é importante?
- **Pensamento Crítico**: Não apenas aceita informações
- **Curiosidade**: Busca entender mais
- **Inovação**: Gera ideias novas

### Como funciona?

```
Pensamento atual
        │
        ▼
┌──────────────────────┐
│ Identificar conceitos│
│ principais           │
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│ Gerar perguntas      │
│ sobre eles           │
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│ Testar contra        │
│ memória existente    │
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│ Armazenar como novo  │
│ pensamento           │
└──────────────────────┘
```

### Implementação Detalhada

**Arquivo**: `src/creativity/creative_thinking.py`

```python
import re
from collections import Counter

class CreativeThinking:
    def __init__(self, model, tokenizer, memory):
        """Inicializa sistema de criatividade"""
        
        self.model = model
        self.tokenizer = tokenizer
        self.memory = memory
        
    def extract_concepts(self, text):
        """Extrai conceitos principais do texto"""
        
        # Usar NLP para extrair entidades/conceitos
        # Simplificado: pegar palavras principais
        
        words = text.lower().split()
        # Remover stopwords
        stopwords = {'o', 'a', 'de', 'e', 'é', 'para', 'com', 'em', 'um', 'uma'}
        
        concepts = [w for w in words if w not in stopwords and len(w) > 3]
        
        return list(set(concepts))
    
    def generate_questions(self, text):
        """Gera perguntas sobre o texto"""
        
        concepts = self.extract_concepts(text)
        
        question_templates = [
            "Por que {}?",
            "Como {} funciona?",
            "Qual é a origem de {}?",
            "Quais são as implicações de {}?",
            "Como {} se relaciona com outros conceitos?",
            "É possível que {} seja diferente?",
            "O que aconteceria se {} fosse o oposto?"
        ]
        
        questions = []
        for concept in concepts[:3]:  # Limitar a 3 conceitos
            for template in question_templates[:2]:  # 2 templates por conceito
                question = template.format(concept)
                questions.append(question)
        
        return questions
    
    def test_idea(self, idea, memory_context):
        """Testa uma ideia contra memória existente"""
        
        # Buscar memórias relacionadas
        similar_memories = self.memory.retrieve_similar(idea, top_k=3)
        
        conflicts = []
        supports = []
        
        for similarity, mem_id, text in similar_memories:
            if similarity > 0.8:
                # Memória muito similar
                if self.check_conflict(idea, text):
                    conflicts.append(text)
                else:
                    supports.append(text)
        
        return {
            'idea': idea,
            'conflicts': conflicts,
            'supports': supports,
            'is_novel': len(supports) == 0 and len(conflicts) == 0
        }
    
    def check_conflict(self, idea1, idea2):
        """Verifica se duas ideias conflitam"""
        
        # Simplificado: procurar por palavras opostas
        opposites = {
            'sim': 'não',
            'verdadeiro': 'falso',
            'possível': 'impossível',
            'real': 'imaginário'
        }
        
        idea1_lower = idea1.lower()
        idea2_lower = idea2.lower()
        
        for word1, word2 in opposites.items():
            if word1 in idea1_lower and word2 in idea2_lower:
                return True
            if word2 in idea1_lower and word1 in idea2_lower:
                return True
        
        return False
    
    def generate_creative_thought(self, current_context):
        """Gera um pensamento criativo"""
        
        # Gerar perguntas
        questions = self.generate_questions(current_context)
        
        # Selecionar uma pergunta aleatória
        import random
        question = random.choice(questions)
        
        # Gerar resposta criativa
        prompt = f"Contexto: {current_context}\nPergunta: {question}\nResposta criativa:"
        
        # Usar modelo para gerar resposta
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=100)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Testar ideia
        test_result = self.test_idea(response, current_context)
        
        return {
            'question': question,
            'response': response,
            'test_result': test_result
        }

# Exemplo de uso
if __name__ == "__main__":
    creativity = CreativeThinking(model, tokenizer, memory)
    
    context = "Sou uma IA. Estou pensando sobre minha própria consciência."
    
    # Gerar pensamento criativo
    creative = creativity.generate_creative_thought(context)
    
    print(f"Pergunta: {creative['question']}")
    print(f"Resposta: {creative['response']}")
    print(f"É nova ideia? {creative['test_result']['is_novel']}")
```

---

## Módulo 6: Persistência Mental

### O que é?
Sistema que garante que pensamentos e aprendizado continuam ativos mesmo sem input externo.

### Por que é importante?
- **Continuidade**: A IA não "reinicia" do zero
- **Evolução Contínua**: Aprende mesmo quando você não está interagindo

---

## Módulo 7: Vigília/Sono

### O que é?
Ciclos de vigília (dia) e sono (noite) com consolidação de aprendizado.

### Por que é importante?
- **Consolidação**: Aprendizado é integrado, não apenas acumulado
- **Fisiologia Simulada**: Simula necessidade de descanso
- **Eficiência**: Evita sobrecarga de memória

### Como funciona?

```
VIGÍLIA (Dia) - 16 horas
├─ Pensamento contínuo
├─ Interação com usuário
├─ Armazenamento de experiências
├─ Ajustes pequenos de pesos
└─ Geração de arquivo de treinamento

    ↓ (Memória atingiu limite)

SONO (Noite) - 8 horas
├─ Para pensamento contínuo
├─ Processa arquivo de treinamento
├─ Fine-tune do modelo
├─ Consolida memória
├─ Limpa arquivo
└─ Volta a vigília

    ↓ (Acordado)

VIGÍLIA (Próximo dia)
```

---

**Próxima Leitura**: `TODO.md` para ver as tarefas específicas de cada fase.


---

## 📝 NOTA IMPORTANTE: Otimização da Gamma

**Sugestão Aceita (2025-11-21)**: Gamma sugeriu uma otimização importante para o Módulo 2 (Memória Sináptica):

**Problema Original**: Armazenar tensores inteiros (BLOB) no banco de dados deixa as queries lentas e não escala bem.

**Solução Proposta pela Gamma**:
1. Armazenar tensores em arquivos `.npy` separados em `data/tensors/`
2. Guardar apenas o **caminho do arquivo** no banco de dados
3. Carregar tensores sob demanda para busca por similaridade

**Vantagens**:
- ✅ Banco de dados fica leve e rápido
- ✅ Busca por similaridade é muito mais eficiente
- ✅ Escalável para milhões de memórias
- ✅ Fácil fazer backup dos tensores
- ✅ Melhor organização de arquivos

**Implementação**:
```python
# Em vez de:
cursor.execute('INSERT INTO memories (tensor_embedding) VALUES (?)', (embedding_bytes,))

# Fazer:
np.save('data/tensors/memory_001_embedding.npy', embedding)
cursor.execute('INSERT INTO memories (tensor_embedding_path) VALUES (?)', ('data/tensors/memory_001_embedding.npy',))
```

**Status**: ✅ Aceita e será implementada na Fase 2

---
