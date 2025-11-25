#!/usr/bin/env python3
"""
SOLUÇÃO NUCLEAR FINAL: Remove TUDO e recria do ZERO
"""

import os
import shutil
import subprocess
from datetime import datetime

def nuclear_final():
    print("💥 SOLUÇÃO NUCLEAR FINAL - RECRIAÇÃO COMPLETA")
    print("=" * 60)
    
    # 1. Backup de TUDO
    backup_dir = f"backup_nuclear_final_{datetime.now().strftime('%Y%m%d_%H%M')}"
    os.makedirs(backup_dir, exist_ok=True)
    
    # Backup dos modelos
    if os.path.exists("models/hac_v6"):
        shutil.copytree("models/hac_v6", f"{backup_dir}/models")
        print("✅ Backup dos modelos criado")
    
    # Backup dos resultados
    if os.path.exists("results"):
        shutil.copytree("results", f"{backup_dir}/results") 
        print("✅ Backup dos resultados criado")
    
    # 2. REMOVE TUDO
    print("\n🗑️  REMOVENDO TUDO...")
    
    if os.path.exists("models/hac_v6"):
        shutil.rmtree("models/hac_v6")
        print("✅ models/hac_v6 removido")
    
    if os.path.exists("results"):
        shutil.rmtree("results") 
        print("✅ results/ removido")
    
    # 3. Recria estrutura limpa
    os.makedirs("models/hac_v6", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    print("✅ Estrutura limpa recriada")
    
    # 4. Retreino COMPLETO do zero
    print(f"\n🎯 INICIANDO RETREINO COMPLETO...")
    
    try:
        # Usa o script de treinamento leve
        result = subprocess.run([
            "python3", "scripts/lightweight_retrain.py"
        ], capture_output=True, text=True, timeout=1200)  # 20 minutos timeout
        
        print("Retreino concluído!")
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
            
    except subprocess.TimeoutExpired:
        print("❌ Retreino excedeu o tempo limite")
    except Exception as e:
        print(f"❌ Erro no retreino: {e}")
    
    # 5. Verificação final
    print(f"\n🔍 VERIFICANDO RESULTADO...")
    
    if os.path.exists("scripts/save_report.py"):
        subprocess.run(["python3", "scripts/save_report.py"])
    
    print(f"\n🎯 BACKUP SALVO EM: {backup_dir}")
    print("🔥 Se ainda houver problemas, restaure do backup e investigue os logs")

if __name__ == "__main__":
    nuclear_final()
