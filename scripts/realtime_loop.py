#!/usr/bin/env python3
"""
realtime_loop.py - Loop de previsão otimizado
"""

import os
import sys
import time
import logging
from datetime import datetime

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/realtime_loop.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Path para imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from hac_v6_config import HACConfig
    from scripts.realtime_once import main as run_once
except ImportError as e:
    logger.error(f"Erro de importação: {e}")
    sys.exit(1)


def main_loop():
    """Loop principal de previsão"""
    # Configuração
    config = HACConfig("config.yaml")
    interval_min = config.get("realtime", {}).get("update_interval_minutes", 30)
    interval_sec = max(60, int(interval_min * 60))  # Mínimo 1 minuto
    
    # Cria diretório de logs
    os.makedirs("logs", exist_ok=True)
    
    logger.info(f"🔁 Iniciando loop HAC v6 (intervalo: {interval_min}min)")
    
    consecutive_errors = 0
    max_consecutive_errors = 5
    
    while True:
        try:
            logger.info("=" * 50)
            logger.info(f"⏰ Rodada: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
            
            # Executa previsão
            success = run_once()
            
            if success:
                consecutive_errors = 0
                logger.info(f"✅ Rodada concluída com sucesso")
            else:
                consecutive_errors += 1
                logger.error(f"❌ Erro na rodada ({consecutive_errors}/{max_consecutive_errors})")
                
                # Se muitos erros consecutivos, espera mais
                if consecutive_errors >= max_consecutive_errors:
                    wait_time = interval_sec * 2  # Dobra o tempo de espera
                    logger.warning(f"⚡ Muitos erros consecutivos, aguardando {wait_time}s")
                    time.sleep(wait_time)
                    consecutive_errors = 0  # Reseta contador
                    continue
            
            # Espera normal
            logger.info(f"💤 Aguardando {interval_sec}s até próxima rodada...")
            time.sleep(interval_sec)
            
        except KeyboardInterrupt:
            logger.info("🛑 Interrompido pelo usuário")
            break
        except Exception as e:
            logger.error(f"💥 Erro não tratado no loop: {e}")
            consecutive_errors += 1
            
            # Backoff exponencial
            backoff_time = min(interval_sec * (2 ** consecutive_errors), 3600)  # Máximo 1 hora
            logger.warning(f"⚡ Backoff exponencial: {backoff_time}s")
            time.sleep(backoff_time)


if __name__ == "__main__":
    try:
        main_loop()
    except Exception as e:
        logger.critical(f"💥 Crash do loop principal: {e}")
        sys.exit(1)
