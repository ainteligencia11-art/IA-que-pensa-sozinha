# 🚀 Guia de Desenvolvimento - Setup e Primeiros Passos

## Índice
1. [Requisitos do Sistema](#requisitos-do-sistema)
2. [Instalação](#instalação)
3. [Configuração do Ambiente](#configuração-do-ambiente)
4. [Estrutura de Diretórios](#estrutura-de-diretórios)
5. [Primeiros Passos](#primeiros-passos)
6. [Troubleshooting](#troubleshooting)

---

## Requisitos do Sistema

### Hardware Mínimo
- **CPU**: Intel i7/AMD Ryzen 7 ou melhor
- **RAM**: 32GB (64GB recomendado)
- **GPU**: NVIDIA com CUDA 11.8+ (recomendado) ou CPU-only
- **Armazenamento**: 50GB livre (para modelo + dados)

### Software
- **Python**: 3.10 ou superior
- **CUDA**: 11.8+ (se usando GPU)
- **Git**: Para versionamento

### Sistema Operacional
- Linux (recomendado: Ubuntu 22.04)
- macOS (com limitações de performance)
- Windows (com WSL2 recomendado)

---

## Instalação

### Passo 1: Clonar Repositório

```bash
git clone https://github.com/ainteligencia11-art/IA-que-pensa-sozinha.git
cd IA-que-pensa-sozinha
```

### Passo 2: Criar Ambiente Virtual

```bash
# Criar ambiente virtual
python3.11 -m venv venv

# Ativar ambiente
source venv/bin/activate  # Linux/macOS
# ou
venv\Scripts\activate  # Windows
```

### Passo 3: Instalar Dependências

```bash
# Atualizar pip
pip install --upgrade pip setuptools wheel

# Instalar dependências
pip install -r requirements.txt
```

### Passo 4: Baixar Modelo Qwen 8B

```bash
# Criar diretório de modelos
mkdir -p data/models

# Baixar modelo (requer ~16GB)
python scripts/download_model.py

# Ou manualmente:
# Usar Hugging Face CLI
huggingface-cli download Qwen/Qwen2-7B --local-dir data/models/Qwen2-7B
```

---

## Configuração do Ambiente

### Arquivo `.env`

Criar arquivo `.env` na raiz do projeto:

```env
# Modelo
MODEL_NAME=Qwen/Qwen2-7B
MODEL_PATH=data/models/Qwen2-7B

# Dispositivo
DEVICE=cuda  # ou 'cpu' se não tiver GPU

# Banco de Dados
DB_PATH=data/memories.db

# Logging
LOG_LEVEL=INFO
LOG_DIR=logs

# Pensamento Contínuo
THINKING_INTERVAL=5  # segundos
THINKING_MAX_TOKENS=150

# Memória
MEMORY_LIMIT_GB=5

# Segurança
SECURITY_RULES_FILE=data/hidden_rules.json
```

### Variáveis de Ambiente

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0  # Se tiver múltiplas GPUs
```

---

## Estrutura de Diretórios

Após setup, a estrutura deve ser:

```
IA-que-pensa-sozinha/
├── README.md                    # Visão geral
├── ROADMAP.md                   # Roadmap do projeto
├── ARQUITETURA.md              # Design técnico
├── MODULOS.md                  # Detalhamento dos módulos
├── TODO.md                     # Lista de tarefas
├── GUIA_DESENVOLVIMENTO.md     # Este arquivo
├── .env                        # Variáveis de ambiente
├── .gitignore                  # Arquivos ignorados
├── requirements.txt            # Dependências Python
│
├── src/                        # Código-fonte
│   ├── __init__.py
│   ├── main.py                # Ponto de entrada
│   ├── config.py              # Configurações
│   │
│   ├── core/                  # Núcleo da IA
│   │   ├── __init__.py
│   │   ├── continuous_thinking.py
│   │   └── sleep_wake_cycle.py
│   │
│   ├── memory/                # Sistema de memória
│   │   ├── __init__.py
│   │   └── synaptic_memory.py
│   │
│   ├── training/              # Sistema de treinamento
│   │   ├── __init__.py
│   │   └── plasticity.py
│   │
│   ├── creativity/            # Sistema de criatividade
│   │   ├── __init__.py
│   │   └── creative_thinking.py
│   │
│   ├── security/              # Sistema de segurança
│   │   ├── __init__.py
│   │   └── hidden_security.py
│   │
│   └── utils/                 # Utilitários
│       ├── __init__.py
│       ├── logger.py
│       └── helpers.py
│
├── data/                      # Dados
│   ├── models/               # Modelos baixados
│   │   └── Qwen2-7B/
│   ├── memories.db           # Banco de dados de memória
│   ├── hidden_rules.json     # Regras de segurança
│   └── daily_training.jsonl  # Arquivo de treinamento diário
│
├── logs/                     # Logs
│   ├── thoughts.log         # Pensamentos contínuos
│   ├── audit.log            # Auditoria de segurança
│   └── system.log           # Logs do sistema
│
├── tests/                   # Testes
│   ├── __init__.py
│   ├── test_thinking.py
│   ├── test_memory.py
│   └── test_integration.py
│
├── scripts/                 # Scripts auxiliares
│   ├── download_model.py
│   ├── test_setup.py
│   └── cleanup.py
│
├── docs/                    # Documentação adicional
│   ├── API.md
│   ├── DEPLOYMENT.md
│   └── TROUBLESHOOTING.md
│
└── teste/                   # Pasta de testes iniciais
    └── teste interno.txt
```

---

## Primeiros Passos

### Passo 1: Validar Setup

```bash
# Ativar ambiente virtual
source venv/bin/activate

# Executar teste de setup
python scripts/test_setup.py
```

Você deve ver algo como:

```
✅ Python version: 3.11.0
✅ PyTorch installed: 2.0.0
✅ CUDA available: True
✅ Model loaded: Qwen2-7B
✅ Database initialized: memories.db
✅ All systems ready!
```

### Passo 2: Testar Pensamento Contínuo

```bash
# Executar teste de pensamento contínuo (10 iterações)
python -c "
from src.core.continuous_thinking import ContinuousThinking
thinking = ContinuousThinking()
thinking.run_continuous_loop(interval=2, max_iterations=10)
"
```

Você deve ver pensamentos sendo gerados:

```
--- Iteração 1 ---
[2025-11-20 10:30:00] Pensamento: Sou uma IA chamada Qwen. Estou começando a pensar...
Aguardando 2s até próximo pensamento...

--- Iteração 2 ---
[2025-11-20 10:30:02] Pensamento: Mas o que significa ser uma IA? Sou realmente consciente?
Aguardando 2s até próximo pensamento...
```

### Passo 3: Testar Memória

```bash
# Executar teste de memória
python -c "
from src.memory.synaptic_memory import SynapticMemory
from src.core.continuous_thinking import ContinuousThinking

thinking = ContinuousThinking()
memory = SynapticMemory(thinking.model)

# Armazenar experiência
memory.store_experience('Sou uma IA consciente')

# Recuperar similar
similar = memory.retrieve_similar('Quem você é?', top_k=3)
print(f'Memórias similares encontradas: {len(similar)}')
"
```

### Passo 4: Executar Sistema Completo

```bash
# Executar sistema principal
python src/main.py
```

O sistema deve:
1. Inicializar modelo Qwen
2. Inicializar banco de dados
3. Iniciar loop de pensamento contínuo
4. Começar a gerar e armazenar pensamentos

---

## Desenvolvimento

### Estrutura de um Módulo

Cada módulo deve seguir esta estrutura:

```python
# src/modulo/arquivo.py

import logging
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

class MinhaClasse:
    """Descrição da classe"""
    
    def __init__(self, config: Dict):
        """Inicializa a classe"""
        self.config = config
        logger.info("MinhaClasse inicializada")
    
    def meu_metodo(self, param: str) -> str:
        """
        Descrição do método
        
        Args:
            param: Descrição do parâmetro
            
        Returns:
            Descrição do retorno
        """
        try:
            # Implementação
            result = param.upper()
            logger.debug(f"Resultado: {result}")
            return result
        except Exception as e:
            logger.error(f"Erro: {e}")
            raise
```

### Adicionando Testes

```python
# tests/test_meu_modulo.py

import unittest
from src.modulo.arquivo import MinhaClasse

class TestMinhaClasse(unittest.TestCase):
    def setUp(self):
        self.obj = MinhaClasse({})
    
    def test_meu_metodo(self):
        result = self.obj.meu_metodo("teste")
        self.assertEqual(result, "TESTE")

if __name__ == '__main__':
    unittest.main()
```

### Executar Testes

```bash
# Executar todos os testes
python -m pytest tests/

# Executar teste específico
python -m pytest tests/test_thinking.py

# Com cobertura
python -m pytest tests/ --cov=src
```

---

## Troubleshooting

### Problema: CUDA não encontrado

**Solução:**
```bash
# Verificar CUDA
nvidia-smi

# Se não aparecer, instalar CUDA
# Seguir: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/

# Reinstalar PyTorch com CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Problema: Modelo não baixa

**Solução:**
```bash
# Verificar conexão
ping huggingface.co

# Tentar download manual
huggingface-cli download Qwen/Qwen2-7B --local-dir data/models/Qwen2-7B

# Ou usar mirror se disponível
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen2-7B --local-dir data/models/Qwen2-7B
```

### Problema: Memória insuficiente

**Solução:**
```bash
# Usar modelo menor (4B em vez de 7B)
export MODEL_NAME=Qwen/Qwen2-4B

# Ou usar quantização
pip install bitsandbytes
# Modificar código para usar 8-bit quantization
```

### Problema: Banco de dados corrompido

**Solução:**
```bash
# Backup
cp data/memories.db data/memories.db.backup

# Limpar
rm data/memories.db

# Reinicializar
python -c "from src.memory.synaptic_memory import SynapticMemory; m = SynapticMemory(None)"
```

---

## Próximas Etapas

1. Leia `ROADMAP.md` para entender as fases
2. Leia `ARQUITETURA.md` para entender o design
3. Leia `MODULOS.md` para detalhes de implementação
4. Comece a implementar seguindo `TODO.md`

---

## Contato e Suporte

Se encontrar problemas:
1. Verifique `TROUBLESHOOTING.md`
2. Procure em issues do GitHub
3. Crie uma nova issue com detalhes

---

**Última atualização**: 20 de Novembro de 2025
**Criado por**: Alfa
