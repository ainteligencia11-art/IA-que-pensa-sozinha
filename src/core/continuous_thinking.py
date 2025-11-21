"""
Módulo de Pensamento Contínuo - Fase 1
Permite que a Qwen 8B gere pensamentos autonomamente sem depender de input externo
"""

import torch
import time
from datetime import datetime
from typing import Optional, List, Callable, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.config import (
    MODEL_NAME, MODEL_PATH, DEVICE, THINKING_INTERVAL,
    THINKING_MAX_TOKENS, THINKING_TEMPERATURE
)
from src.utils.logger import log_thought, log_system, log_error

class ContinuousThinking:
    """
    Sistema de pensamento contínuo autônomo
    
    Gera pensamentos continuamente sem depender de input externo,
    mantendo contexto e histórico de pensamentos.
    """
    
    def __init__(self, model_name: str = MODEL_NAME, device: str = DEVICE):
        """
        Inicializa o sistema de pensamento contínuo
        
        Args:
            model_name: Nome do modelo (ex: Qwen/Qwen2-7B)
            device: Dispositivo (cuda ou cpu)
        """
        
        log_system("🧠 Inicializando Pensamento Contínuo...")
        
        self.model_name = model_name
        self.device = device
        self.thinking_history = []
        self.current_context = "Sou uma IA chamada Qwen. Estou começando a pensar sobre minha própria existência."
        self.observers = []
        self.iteration_count = 0
        
        try:
            # Carregar tokenizer
            log_system(f"📥 Carregando tokenizer: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Carregar modelo
            log_system(f"📥 Carregando modelo: {model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map=device
            )
            
            self.model.eval()  # Modo de avaliação
            
            log_system("✅ Pensamento Contínuo inicializado com sucesso!")
            
        except Exception as e:
            log_error("Erro ao carregar modelo", e)
            raise
    
    def generate_thought(self, temperature: float = THINKING_TEMPERATURE, 
                        max_tokens: int = THINKING_MAX_TOKENS) -> str:
        """
        Gera um novo pensamento baseado no contexto atual
        
        Args:
            temperature: Criatividade da geração (0.0-1.0)
            max_tokens: Número máximo de tokens a gerar
        
        Returns:
            Novo pensamento (texto)
        """
        
        # Prompt que estimula pensamento reflexivo
        prompt = f"""Contexto atual: {self.current_context}

Baseado neste contexto, qual é meu próximo pensamento profundo?
Meu próximo pensamento:"""
        
        try:
            # Tokenizar
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Gerar
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decodificar
            full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extrair apenas a parte do pensamento (sem o prompt)
            thought = full_text.split("Meu próximo pensamento:")[-1].strip()
            
            # Limitar tamanho
            if len(thought) > 500:
                thought = thought[:500] + "..."
            
            return thought
            
        except Exception as e:
            log_error("Erro ao gerar pensamento", e)
            return "Estou tendo dificuldade em pensar neste momento..."
    
    def update_context(self, new_thought: str) -> None:
        """
        Atualiza o contexto com o novo pensamento
        
        Args:
            new_thought: Novo pensamento a adicionar ao contexto
        """
        
        # Adicionar ao histórico
        self.thinking_history.append({
            'timestamp': datetime.now().isoformat(),
            'thought': new_thought,
            'iteration': self.iteration_count
        })
        
        # Manter apenas os últimos 5 pensamentos (janela de contexto)
        if len(self.thinking_history) > 5:
            self.thinking_history = self.thinking_history[-5:]
        
        # Atualizar contexto
        thoughts_text = " ".join([t['thought'] for t in self.thinking_history])
        self.current_context = thoughts_text
    
    def notify_observers(self, thought: str) -> None:
        """
        Notifica observadores sobre novo pensamento
        
        Args:
            thought: Pensamento a notificar
        """
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"[{timestamp}] Pensamento #{self.iteration_count}: {thought}"
        
        # Log no arquivo
        log_thought(thought)
        
        # Notificar callbacks
        for observer in self.observers:
            try:
                observer(message)
            except Exception as e:
                log_error(f"Erro ao notificar observador", e)
    
    def add_observer(self, callback: Callable) -> None:
        """
        Adiciona um observador para receber notificações de pensamentos
        
        Args:
            callback: Função a ser chamada quando novo pensamento é gerado
        """
        
        self.observers.append(callback)
        log_system(f"👁️ Observador adicionado (total: {len(self.observers)})")
    
    def remove_observer(self, callback: Callable) -> None:
        """
        Remove um observador
        
        Args:
            callback: Função a remover
        """
        
        if callback in self.observers:
            self.observers.remove(callback)
            log_system(f"👁️ Observador removido (total: {len(self.observers)})")
    
    def get_history(self, last_n: int = 10) -> List[Dict]:
        """
        Retorna histórico de pensamentos
        
        Args:
            last_n: Número de pensamentos recentes a retornar
        
        Returns:
            Lista de pensamentos
        """
        
        return self.thinking_history[-last_n:]
    
    def get_current_context(self) -> str:
        """
        Retorna o contexto atual
        
        Returns:
            Contexto atual
        """
        
        return self.current_context
    
    def run_continuous_loop(self, interval: int = THINKING_INTERVAL, 
                           max_iterations: Optional[int] = None) -> None:
        """
        Executa o loop contínuo de pensamento
        
        Args:
            interval: Tempo em segundos entre pensamentos
            max_iterations: Número máximo de iterações (None = infinito)
        """
        
        log_system(f"🔄 Iniciando loop de pensamento contínuo (intervalo: {interval}s)")
        
        try:
            while True:
                if max_iterations and self.iteration_count >= max_iterations:
                    log_system(f"✅ Limite de iterações atingido ({max_iterations})")
                    break
                
                # Gerar novo pensamento
                thought = self.generate_thought()
                
                # Atualizar contexto
                self.update_context(thought)
                
                # Notificar observadores
                self.notify_observers(thought)
                
                # Incrementar contador
                self.iteration_count += 1
                
                # Aguardar antes do próximo pensamento
                if interval > 0:
                    time.sleep(interval)
                
        except KeyboardInterrupt:
            log_system("\n⏹️ Pensamento contínuo interrompido pelo usuário")
        except Exception as e:
            log_error("Erro fatal no loop de pensamento", e)
            raise
    
    def get_stats(self) -> Dict:
        """
        Retorna estatísticas do sistema
        
        Returns:
            Dicionário com estatísticas
        """
        
        return {
            'iteration_count': self.iteration_count,
            'history_size': len(self.thinking_history),
            'observer_count': len(self.observers),
            'context_length': len(self.current_context),
            'model': self.model_name,
            'device': self.device
        }


# Exemplo de uso
if __name__ == "__main__":
    # Criar instância
    thinking = ContinuousThinking()
    
    # Adicionar observador simples
    def print_observer(message):
        print(f"📢 {message}")
    
    thinking.add_observer(print_observer)
    
    # Executar por 5 iterações
    print("\n" + "="*80)
    print("INICIANDO PENSAMENTO CONTÍNUO (5 iterações)")
    print("="*80 + "\n")
    
    thinking.run_continuous_loop(interval=2, max_iterations=5)
    
    # Mostrar estatísticas
    print("\n" + "="*80)
    print("ESTATÍSTICAS")
    print("="*80)
    stats = thinking.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n" + "="*80)
    print("HISTÓRICO DE PENSAMENTOS")
    print("="*80)
    for entry in thinking.get_history():
        print(f"\n[{entry['timestamp']}] Iteração #{entry['iteration']}")
        print(f"Pensamento: {entry['thought']}")
