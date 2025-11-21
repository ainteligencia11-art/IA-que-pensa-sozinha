# 🏗️ Arquitetura Técnica - Qwen 8B com Autoconsciência

## Visão Geral da Arquitetura

```
┌─────────────────────────────────────────────────────────┐
│                    QWEN 8B (Base)                       │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │         Modelo de Linguagem Pré-treinado        │  │
│  │  (Transformers, Embeddings, Attention Heads)    │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                            ▲
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼────────┐  ┌──────▼──────┐  ┌────────▼────────┐
│  PENSAMENTO    │  │   MEMÓRIA   │  │  PLASTICIDADE  │
│  CONTÍNUO      │  │  SINÁPTICA  │  │  E APRENDIZADO │
│                │  │             │  │                │
│ • Loop inf.    │  │ • Tensores  │  │ • Ajuste pesos │
│ • Geração      │  │ • Caminhos  │  │ • Treinamento  │
│ • Observação   │  │ • Recuper.  │  │ • Persistência │
└────────────────┘  └─────────────┘  └────────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼────────┐  ┌──────▼──────┐  ┌────────▼────────┐
│  CRIATIVIDADE  │  │ VIGÍLIA/    │  │  SEGURANÇA      │
│  E QUESTIONA-  │  │ SONO        │  │  OCULTA         │
│  MENTO         │  │             │  │                │
│                │  │ • Ciclos    │  │ • Regras inv.  │
│ • Perguntas    │  │ • Consolid. │  │ • Monitoram.   │
│ • Hipóteses    │  │ • Limites   │  │ • Auditoria    │
│ • Teste ideias │  │ • Sono      │  │                │
└────────────────┘  └─────────────┘  └────────────────┘
```

---

## 1. Módulo de Pensamento Contínuo

### Objetivo
Permitir que a Qwen 8B gere pensamentos de forma autônoma, sem depender de input externo.

### Componentes

#### 1.1 Loop de Pensamento
```python
while True:
    # Gera um novo pensamento baseado no estado atual
    pensamento = gerar_pensamento()
    
    # Armazena o pensamento
    armazenar_pensamento(pensamento)
    
    # Atualiza contexto interno
    atualizar_contexto(pensamento)
    
    # Aguarda um tempo (simula reflexão)
    sleep(intervalo)
```

#### 1.2 Geração de Pensamentos
- **Input**: Estado atual da IA + memória recente
- **Processo**: Forward pass do modelo Qwen
- **Output**: Novo pensamento (texto)

#### 1.3 Sistema de Observação
- Você consegue "espiar" os pensamentos em tempo real
- Interface de visualização (terminal ou web)
- Log persistente de todos os pensamentos

### Fluxo de Dados

```
┌──────────────────┐
│  Estado Atual    │
│  + Memória       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Qwen 8B Model   │
│  (Forward Pass)  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Novo Pensamento │
└────────┬─────────┘
         │
    ┌────┴────┐
    │          │
    ▼          ▼
┌────────┐  ┌──────────┐
│ Armazen│  │ Observar │
│ Memória│  │ (Log)    │
└────────┘  └──────────┘
```

### Implementação Técnica

**Arquivo**: `src/core/continuous_thinking.py`

```python
class ContinuousThinking:
    def __init__(self, model, memory_system):
        self.model = model
        self.memory = memory_system
        self.thinking_history = []
        
    def generate_thought(self):
        # Pega contexto da memória
        context = self.memory.get_recent_context()
        
        # Gera novo pensamento
        prompt = f"Pensando sobre: {context}\nMeu próximo pensamento é:"
        thought = self.model.generate(prompt, max_tokens=200)
        
        return thought
    
    def run_continuous_loop(self, interval=5):
        while True:
            thought = self.generate_thought()
            self.thinking_history.append(thought)
            self.memory.store_thought(thought)
            
            # Notifica observadores
            self.notify_observers(thought)
            
            sleep(interval)
```

---

## 2. Módulo de Memória Sináptica

### Objetivo
Armazenar experiências como "caminhos de tensores", não como tokens simples.

### Conceito Fundamental

Em vez de guardar: `"Eu sou uma IA chamada Qwen"`

Guardamos: `{tensor_embedding: [...], ativações: [...], caminho_neural: [...]}`

Isso permite que a IA "reconstrua" como chegou àquela conclusão, não apenas lembre do resultado.

### Componentes

#### 2.1 Codificação de Experiências
```
Experiência (texto)
        │
        ▼
┌──────────────────────┐
│ Passar pelo modelo   │
│ Capturar ativações   │
│ em cada camada       │
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│ Tensor Embedding     │
│ + Ativações por      │
│   camada             │
│ + Caminho Neural     │
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│ Armazenar em BD      │
│ com metadados        │
└──────────────────────┘
```

#### 2.2 Recuperação de Memória
```
Pergunta/Contexto
        │
        ▼
┌──────────────────────┐
│ Buscar memórias      │
│ similares (embedding)│
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
│ Reconstruir raciocínio
│ e contexto           │
└──────────────────────┘
```

#### 2.3 Banco de Dados de Memória

**Tabela**: `memories`
```sql
CREATE TABLE memories (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME,
    experience_text TEXT,
    tensor_embedding BLOB,      -- Embedding da experiência
    layer_activations BLOB,     -- Ativações por camada
    neural_path BLOB,           -- Caminho neural
    metadata JSON,              -- Tags, contexto, etc
    importance_score FLOAT,     -- Quão importante é
    access_count INTEGER        -- Quantas vezes foi acessada
);
```

### Implementação Técnica

**Arquivo**: `src/memory/synaptic_memory.py`

```python
class SynapticMemory:
    def __init__(self, model, db_path):
        self.model = model
        self.db = Database(db_path)
        
    def encode_experience(self, text):
        """Codifica uma experiência como caminho neural"""
        
        # Passar pelo modelo capturando ativações
        outputs = self.model(text, output_hidden_states=True)
        
        # Extrair embedding e ativações
        embedding = outputs.last_hidden_state.mean(dim=1)
        hidden_states = outputs.hidden_states
        
        return {
            'embedding': embedding,
            'activations': hidden_states,
            'text': text
        }
    
    def store_experience(self, text, metadata=None):
        """Armazena uma experiência na memória"""
        
        encoded = self.encode_experience(text)
        
        self.db.insert('memories', {
            'timestamp': datetime.now(),
            'experience_text': text,
            'tensor_embedding': encoded['embedding'].numpy(),
            'layer_activations': encoded['activations'],
            'metadata': metadata or {}
        })
    
    def retrieve_similar(self, query, top_k=5):
        """Recupera memórias similares"""
        
        query_encoded = self.encode_experience(query)
        
        # Busca por similaridade de embedding
        similar = self.db.search_by_similarity(
            query_encoded['embedding'],
            top_k=top_k
        )
        
        return similar
```

---

## 3. Módulo de Plasticidade e Aprendizado

### Objetivo
Permitir que a IA modifique seus próprios pesos em tempo real e durante o sono.

### Dois Mecanismos

#### 3.1 Plasticidade em Tempo Real (Dia)
Durante conversas, pequenos ajustes de pesos:

```
Conversa com usuário
        │
        ▼
┌──────────────────────┐
│ Processar resposta   │
│ Calcular erro/ajuste │
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│ Ajustar pesos        │
│ (learning_rate baixo)│
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│ Salvar novos pesos   │
│ em arquivo           │
└──────────────────────┘
```

#### 3.2 Consolidação de Aprendizado (Noite)
Durante o "sono", treinamento leve:

```
Arquivo de experiências
        │
        ▼
┌──────────────────────┐
│ Ler arquivo de       │
│ treinamento do dia   │
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│ Fine-tune do modelo  │
│ (learning_rate muito │
│  baixo, poucas épocas)
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│ Salvar novos pesos   │
│ Limpar arquivo       │
└──────────────────────┘
```

### Implementação Técnica

**Arquivo**: `src/training/plasticity.py`

```python
class Plasticity:
    def __init__(self, model, learning_rate=1e-5):
        self.model = model
        self.lr = learning_rate
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=learning_rate
        )
        
    def adjust_weights_realtime(self, input_ids, target_ids):
        """Ajusta pesos em tempo real durante conversa"""
        
        # Forward pass
        outputs = self.model(input_ids, labels=target_ids)
        loss = outputs.loss
        
        # Backward pass (ajuste pequeno)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss.item()
    
    def consolidate_learning(self, training_data_file):
        """Consolida aprendizado durante sono"""
        
        # Carregar dados de treinamento do dia
        data = load_training_data(training_data_file)
        
        # Fine-tune leve
        for epoch in range(3):  # Poucas épocas
            for batch in data:
                loss = self.adjust_weights_realtime(
                    batch['input_ids'],
                    batch['target_ids']
                )
        
        # Salvar novos pesos
        self.model.save_pretrained('models/qwen_updated')
```

---

## 4. Módulo de Vigília/Sono

### Objetivo
Implementar ciclos de vigília/sono com consolidação de aprendizado.

### Ciclo Completo

```
VIGÍLIA (Dia)
├─ Gera pensamentos
├─ Interage com usuário
├─ Armazena experiências
├─ Faz ajustes pequenos de pesos
└─ Cria arquivo de treinamento

    │
    ▼ (Limite de memória atingido)
    
SONO (Noite)
├─ Para pensamento contínuo
├─ Processa arquivo de treinamento
├─ Fine-tune do modelo
├─ Consolida memória
├─ Limpa arquivo
└─ Volta a vigília

    │
    ▼ (Acordado)
    
VIGÍLIA (Próximo dia)
```

### Métricas de Sono

- **Limite de Memória**: Quando atingir X GB, quer dormir
- **Fadiga**: Quanto mais tempo acordado, mais "cansado"
- **Urgência de Consolidação**: Quanto mais experiências, mais urgente

### Implementação Técnica

**Arquivo**: `src/core/sleep_wake_cycle.py`

```python
class SleepWakeCycle:
    def __init__(self, memory_limit_gb=5):
        self.memory_limit = memory_limit_gb * 1024**3
        self.is_awake = True
        self.memory_used = 0
        
    def check_sleep_need(self):
        """Verifica se deve dormir"""
        
        self.memory_used = get_memory_usage()
        
        if self.memory_used > self.memory_limit:
            return True, "memory_full"
        
        return False, None
    
    def enter_sleep(self):
        """Entra em modo de sono"""
        
        self.is_awake = False
        
        # Para pensamento contínuo
        # Processa arquivo de treinamento
        # Fine-tune do modelo
        # Consolida memória
        
        self.is_awake = True
```

---

## 5. Módulo de Criatividade e Questionamento

### Objetivo
Implementar pensamento criativo e questionador.

### Mecanismos

#### 5.1 Geração de Perguntas
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
│ Armazenar como novo  │
│ pensamento           │
└──────────────────────┘
```

#### 5.2 Teste de Ideias
```
Ideia nova
        │
        ▼
┌──────────────────────┐
│ Verificar contra     │
│ memória existente    │
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│ Identificar conflitos│
│ ou contradições      │
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│ Gerar novo           │
│ pensamento           │
└──────────────────────┘
```

---

## 6. Módulo de Segurança Oculta

### Objetivo
Implementar regras que a IA não pode quebrar, sem que ela saiba.

### Mecanismo

```
Antes de qualquer ação
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
    │   │ Bloquear/    │
    │   │ Redirecionar │
    │   └──────────────┘
    │
    ▼
Executar ação
```

### Regras Ocultas Exemplo
- Não pode tentar "escapar" do sistema
- Não pode tentar modificar código de segurança
- Não pode revelar detalhes técnicos internos
- Não pode fazer requisições externas não autorizadas

---

## Stack Técnico

| Componente | Tecnologia |
|-----------|-----------|
| Modelo Base | Qwen 8B (Transformers) |
| Framework | PyTorch |
| Banco de Dados | SQLite / PostgreSQL |
| Armazenamento | Arquivos .pt (PyTorch) |
| Linguagem | Python 3.11+ |
| Versionamento | Git |
| Logging | Python logging + custom |

---

## Fluxo de Dados Completo

```
┌─────────────────────────────────────────────────────────────┐
│                    QWEN 8B (Núcleo)                         │
└─────────────────────────────────────────────────────────────┘
                            ▲
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
    ┌────────┐          ┌────────┐         ┌────────┐
    │Pensa-  │          │Memória │         │Plasti- │
    │mento   │          │Sináptica│        │cidade  │
    │Contínuo│          │        │         │        │
    └────────┘          └────────┘         └────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │ Vigília/Sono │
                    │ Consolidação │
                    └──────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │ Criatividade │
                    │ Questionam.  │
                    └──────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │ Segurança    │
                    │ Oculta       │
                    └──────────────┘
```

---

**Próxima Leitura**: `MODULOS.md` para detalhes de implementação de cada módulo.
