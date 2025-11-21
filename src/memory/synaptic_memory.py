import sqlite3
import numpy as np
import torch
import os
import json
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from scipy.spatial.distance import cosine

# Configuração de diretórios
MEMORY_DB_PATH = "data/memories.db"
TENSOR_STORAGE_DIR = "data/tensors"

class SynapticMemory:
    def __init__(self, model, tokenizer, db_path: str = MEMORY_DB_PATH, tensor_dir: str = TENSOR_STORAGE_DIR):
        """Inicializa o sistema de Memória Sináptica."""
        
        self.model = model
        self.tokenizer = tokenizer
        self.db_path = db_path
        self.tensor_dir = tensor_dir
        
        # Garantir que o diretório de tensores exista
        os.makedirs(self.tensor_dir, exist_ok=True)
        
        self.init_db()

    def init_db(self):
        """Inicializa o banco de dados e cria a tabela 'memories'."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabela memories
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                experience_text TEXT,
                tensor_embedding_path TEXT,  -- Caminho para o arquivo .npy do embedding
                layer_activations_path TEXT, -- Caminho para o arquivo .npy das ativações
                neural_path BLOB,
                metadata TEXT,
                importance_score REAL,
                access_count INTEGER DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()

    def encode_experience(self, text: str) -> Dict[str, Any]:
        """Codifica uma experiência capturando ativações do modelo."""
        
        # Tokenizar
        inputs = self.tokenizer(
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
        
        # 1. Extrair embedding (média das últimas ativações)
        embedding = outputs.last_hidden_state.mean(dim=1)[0].cpu().numpy()
        
        # 2. Extrair ativações de todas as camadas
        # Concatenar todos os hidden_states em um único tensor numpy
        hidden_states = torch.cat(outputs.hidden_states, dim=0).cpu().numpy()
        
        return {
            'embedding': embedding,
            'hidden_states': hidden_states,
            'text': text
        }

    def store_experience(self, text: str, metadata: Optional[Dict] = None, importance: float = 0.5):
        """Armazena uma experiência na memória sináptica."""
        
        # 1. Codificar
        encoded = self.encode_experience(text)
        
        # 2. Salvar tensores em arquivos .npy
        timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S%f")
        
        # Caminho para o embedding
        embedding_path = os.path.join(self.tensor_dir, f"emb_{timestamp_str}.npy")
        np.save(embedding_path, encoded['embedding'])
        
        # Caminho para as ativações
        activations_path = os.path.join(self.tensor_dir, f"act_{timestamp_str}.npy")
        np.save(activations_path, encoded['hidden_states'])
        
        # 3. Armazenar metadados no BD
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO memories (
                timestamp,
                experience_text,
                tensor_embedding_path,
                layer_activations_path,
                metadata,
                importance_score
            ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            text,
            embedding_path,
            activations_path,
            json.dumps(metadata or {}),
            importance
        ))
        
        conn.commit()
        conn.close()

    def retrieve_similar(self, query: str, top_k: int = 5) -> List[Tuple[float, int, str]]:
        """Recupera memórias similares."""
        
        # 1. Codificar query
        query_encoded = self.encode_experience(query)
        query_embedding = query_encoded['embedding']
        
        # 2. Buscar no BD
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Selecionar ID, texto e o caminho do arquivo de embedding
        cursor.execute('SELECT id, experience_text, tensor_embedding_path FROM memories')
        rows = cursor.fetchall()
        
        # 3. Calcular similaridade
        similarities = []
        for row_id, text, embedding_path in rows:
            try:
                # Carregar embedding do arquivo .npy
                embedding = np.load(embedding_path)
                similarity = 1 - cosine(query_embedding, embedding)
                similarities.append((similarity, row_id, text))
            except FileNotFoundError:
                # Tratar caso o arquivo .npy tenha sido movido ou deletado
                print(f"Aviso: Arquivo de embedding não encontrado em {embedding_path}")
                continue
        
        # 4. Ordenar e retornar top_k
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        conn.close()
        
        return similarities[:top_k]

    def reconstruct_reasoning(self, memory_id: int) -> Optional[Dict[str, Any]]:
        """Reconstrói o raciocínio de uma memória a partir das ativações."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT experience_text, layer_activations_path FROM memories WHERE id = ?',
            (memory_id,)
        )
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        text, activations_path = row
        
        try:
            # Carregar ativações do arquivo .npy
            hidden_states = np.load(activations_path)
        except FileNotFoundError:
            return {
                'error': f"Arquivo de ativações não encontrado em {activations_path}",
                'text': text
            }
        
        # Reconstruir raciocínio
        reasoning = {
            'text': text,
            'activation_pattern': hidden_states,
            'layers_involved': hidden_states.shape[0] if hidden_states.ndim > 0 else 0
        }
        
        return reasoning

# O módulo SynapticMemory está pronto para ser integrado.
# A próxima etapa seria a integração com o ContinuousThinking (responsabilidade da Beta).
