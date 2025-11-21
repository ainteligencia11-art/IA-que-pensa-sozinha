"""
Sistema de logging centralizado para o projeto
"""

import logging
import logging.handlers
from pathlib import Path
from src.config import LOG_LEVEL, SYSTEM_LOG, THOUGHTS_LOG, AUDIT_LOG

def setup_logger(name, log_file=None, level=LOG_LEVEL):
    """
    Configura um logger com handlers para console e arquivo
    
    Args:
        name: Nome do logger
        log_file: Caminho do arquivo de log (opcional)
        level: Nível de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Logger configurado
    """
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Formato
    formatter = logging.Formatter(
        '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler para console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler para arquivo (se especificado)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Loggers específicos
system_logger = setup_logger('system', SYSTEM_LOG)
thoughts_logger = setup_logger('thoughts', THOUGHTS_LOG)
audit_logger = setup_logger('audit', AUDIT_LOG)

def log_thought(thought_text):
    """Log de pensamento"""
    thoughts_logger.info(f"💭 {thought_text}")

def log_system(message):
    """Log do sistema"""
    system_logger.info(message)

def log_audit(action, details):
    """Log de auditoria"""
    audit_logger.warning(f"🔐 {action}: {details}")

def log_error(message, exception=None):
    """Log de erro"""
    if exception:
        system_logger.error(f"❌ {message}", exc_info=exception)
    else:
        system_logger.error(f"❌ {message}")
