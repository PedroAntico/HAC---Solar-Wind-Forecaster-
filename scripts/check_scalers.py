#!/usr/bin/env python3
"""
Verifica se os scalers Y estão presentes nos modelos
"""

import os
import sys
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hac_v6_config import HACConfig

def main():
    config = HACConfig("config.yaml")
    model_dir = config.get("paths")["model_dir"]
    
    print(f"🔍 Verificando scalers Y em: {model_dir}")
    
    missing_scalers = []
    
    for folder in os.listdir(model_dir):
        folder_path = os.path.join(model_dir, folder)
        if not os.path.isdir(folder_path):
            continue
            
        scaler_path = os.path.join(folder_path, "scaler_Y.pkl")
        model_path = os.path.join(folder_path, "model.keras")
        
        if os.path.exists(model_path):
            if os.path.exists(scaler_path):
                try:
                    scaler = joblib.load(scaler_path)
                    print(f"✅ {folder}: Scaler Y carregável")
                except Exception as e:
                    print(f"❌ {folder}: Scaler Y corrompido - {e}")
                    missing_scalers.append(folder)
            else:
                print(f"❌ {folder}: SCALER Y AUSENTE")
                missing_scalers.append(folder)
    
    if missing_scalers:
        print(f"\n🚨 PROBLEMA: {len(missing_scalers)} modelos sem scalers Y:")
        for folder in missing_scalers:
            print(f"   - {folder}")
    else:
        print(f"\n🎉 TODOS OS MODELOS TÊM SCALERS Y!")

if __name__ == "__main__":
    main()
