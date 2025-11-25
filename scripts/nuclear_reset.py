#!/usr/bin/env python3
"""
SOLUÇÃO NUCLEAR: Remove TUDO e recria scalers do zero
"""

import os
import sys
import shutil
from datetime import datetime

def nuclear_reset():
    print("💥 SOLUÇÃO NUCLEAR - RECRIAÇÃO COMPLETA")
    print("=" * 50)
    
    # Backup dos modelos atuais
    model_dir = "models/hac_v6"
    backup_dir = f"models/backup_nuclear_{datetime.now().strftime('%Y%m%d_%H%M')}"
    
    if os.path.exists(model_dir):
        shutil.copytree(model_dir, backup_dir)
        print(f"✅ Backup criado: {backup_dir}")
    
    # Remove scalers Y problemáticos mas mantém modelos
    scalers_removed = 0
    for folder in os.listdir(model_dir):
        folder_path = os.path.join(model_dir, folder)
        if os.path.isdir(folder_path):
            scaler_path = os.path.join(folder_path, "scaler_Y.pkl")
            if os.path.exists(scaler_path):
                os.remove(scaler_path)
                scalers_removed += 1
                print(f"🗑️  Removido: {scaler_path}")
    
    print(f"✅ {scalers_removed} scalers Y removidos")
    print(f"\n🎯 AGORA EXECUTE:")
    print("python3 scripts/fix_scalers_emergency_final.py")
    
    return True

if __name__ == "__main__":
    nuclear_reset()
