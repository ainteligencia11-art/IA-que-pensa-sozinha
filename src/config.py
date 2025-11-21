"""
Configurações globais do projeto Qwen com Autoconsciência
"""

import os
from pathlib import Path

# Diretórios
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
TENSORS_DIR = DATA_DIR / "tensors"
LOGS_DIR = PROJECT_ROOT / "logs"

# Criar diretórios se não existirem
TENSORS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Modelo
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2-7B")
MODEL_PATH = os.getenv("MODEL_PATH", str(MODELS_DIR / "Qwen2-7B"))
DEVICE = os.getenv("DEVICE", "cuda")

# Banco de Dados
DB_PATH = os.getenv("DB_PATH", str(DATA_DIR / "memories.db"))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
THOUGHTS_LOG = LOGS_DIR / "thoughts.log"
AUDIT_LOG = LOGS_DIR / "audit.log"
SYSTEM_LOG = LOGS_DIR / "system.log"

# Pensamento Contínuo
THINKING_INTERVAL = int(os.getenv("THINKING_INTERVAL", "5"))
THINKING_MAX_TOKENS = int(os.getenv("THINKING_MAX_TOKENS", "150"))
THINKING_TEMPERATURE = float(os.getenv("THINKING_TEMPERATURE", "0.7"))

# Memória
MEMORY_LIMIT_GB = int(os.getenv("MEMORY_LIMIT_GB", "5"))
MEMORY_CONTEXT_SIZE = int(os.getenv("MEMORY_CONTEXT_SIZE", "5"))

# Segurança
SECURITY_RULES_FILE = os.getenv("SECURITY_RULES_FILE", str(DATA_DIR / "hidden_rules.json"))

# Treinamento
TRAINING_DATA_FILE = DATA_DIR / "daily_training.jsonl"
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-5"))
SLEEP_CONSOLIDATION_EPOCHS = int(os.getenv("SLEEP_CONSOLIDATION_EPOCHS", "3"))

# Debug
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

print(f"✅ Configurações carregadas:")
print(f"   Modelo: {MODEL_NAME}")
print(f"   Dispositivo: {DEVICE}")
print(f"   DB: {DB_PATH}")
print(f"   Logs: {LOGS_DIR}")
