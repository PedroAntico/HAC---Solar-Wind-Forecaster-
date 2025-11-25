#!/usr/bin/env python3
"""
Diagnóstico rápido da estrutura de modelos
"""

import os
import sys
import json

def diagnose_models():
    print("🔍 DIAGNÓSTICO DA ESTRUTURA DE MODELOS")
    print("=" * 50)
    
    # Verifica estrutura de pastas
    model_dir = "models/hac_v6"
    
    if not os.path.exists(model_dir):
        print(f"❌ Pasta não existe: {model_dir}")
        print("\n📁 Estrutura atual do repositório:")
        for root, dirs, files in os.walk("."):
            level = root.replace(".", "").count(os.sep)
            indent = " " * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = " " * 2 * (level + 1)
            for file in files[:5]:  # Mostra só os primeiros 5 arquivos
                print(f"{subindent}{file}")
        return False
    
    print(f"✅ Pasta existe: {model_dir}")
    
    # Lista conteúdo
    items = os.listdir(model_dir)
    if not items:
        print("❌ Pasta vazia - nenhum modelo encontrado")
        return False
    
    print(f"📦 Conteúdo da pasta ({len(items)} itens):")
    
    model_folders = []
    for item in items:
        item_path = os.path.join(model_dir, item)
        if os.path.isdir(item_path):
            model_folders.append(item)
            print(f"   📁 {item}")
            
            # Verifica conteúdo da pasta do modelo
            model_files = os.listdir(item_path)
            for file in model_files:
                file_path = os.path.join(item_path, file)
                file_size = os.path.getsize(file_path) if os.path.isfile(file_path) else 0
                print(f"      📄 {file} ({file_size} bytes)")
    
    if not model_folders:
        print("❌ Nenhuma pasta de modelo encontrada")
        return False
    
    # Verifica se os modelos são carregáveis
    print(f"\n🧪 Testando carregamento de modelos...")
    
    try:
        sys.path.append('.')
        from hac_v6_predictor_github import HACv6PredictorGithub
        
        predictor = HACv6PredictorGithub()
        predictor.ensure_loaded()
        status = predictor.get_status()
        
        print(f"✅ Predictor status: {status}")
        
        if status['models_loaded'] > 0:
            print(f"🎯 {status['models_loaded']} modelos carregados com sucesso!")
            return True
        else:
            print("❌ Predictor não conseguiu carregar nenhum modelo")
            return False
            
    except Exception as e:
        print(f"❌ Erro ao testar predictor: {e}")
        return False

def check_training_status():
    """Verifica se há evidências de treinamento anterior"""
    print(f"\n📊 VERIFICANDO EVIDÊNCIAS DE TREINAMENTO:")
    
    evidence_found = False
    
    # Verifica results/
    if os.path.exists("results"):
        result_files = os.listdir("results")
        if result_files:
            print(f"✅ Arquivos em results/: {result_files}")
            evidence_found = True
        else:
            print("❌ Pasta results/ vazia")
    
    # Verifica logs/
    if os.path.exists("logs"):
        log_files = os.listdir("logs")
        if log_files:
            print(f"✅ Arquivos em logs/: {log_files}")
            evidence_found = True
        else:
            print("❌ Pasta logs/ vazia")
    
    # Verifica se há scripts de treinamento
    training_scripts = [
        "scripts/lightweight_retrain.py",
        "scripts/retrain_fix_scalers.py", 
        "scripts/compare_models.py"
    ]
    
    for script in training_scripts:
        if os.path.exists(script):
            print(f"✅ Script de treinamento: {script}")
            evidence_found = True
    
    return evidence_found

if __name__ == "__main__":
    print("🚀 DIAGNÓSTICO COMPLETO DO HAC v6")
    print()
    
    models_ok = diagnose_models()
    training_evidence = check_training_status()
    
    print(f"\n🎯 DIAGNÓSTICO FINAL:")
    
    if models_ok:
        print("✅ MODELOS: Carregados e funcionais")
    else:
        print("❌ MODELOS: Problemas detectados")
        
    if training_evidence:
        print("✅ TREINAMENTO: Evidências encontradas")
    else:
        print("❌ TREINAMENTO: Nenhuma evidência encontrada")
    
    print(f"\n🔧 PRÓXIMOS PASSOS:")
    
    if not models_ok and not training_evidence:
        print("1. Execute o treinamento inicial:")
        print("   python3 scripts/lightweight_retrain.py")
    elif not models_ok but training_evidence:
        print("1. Os modelos podem estar corrompidos. Recrie:")
        print("   python3 scripts/lightweight_retrain.py")
    else:
        print("1. Tudo OK! Teste as previsões:")
        print("   python3 scripts/save_report.py")
