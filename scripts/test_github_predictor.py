#!/usr/bin/env python3
"""
Teste específico para verificar funcionamento no GitHub Free
"""

import os
import sys
import logging
import psutil
import gc

# Configuração mínima
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def memory_usage():
    """Monitora uso de memória"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # MB

def test_github_compatibility():
    """Teste completo de compatibilidade com GitHub Free"""
    print("🧪 TESTE DE COMPATIBILIDADE GITHUB FREE")
    print("=" * 50)
    
    initial_memory = memory_usage()
    print(f"📊 Memória inicial: {initial_memory:.1f} MB")
    
    try:
        # Import seletivo para economizar memória
        from hac_v6_predictor_github import get_predictor_github
        from hac_v6_config import HACConfig
        from hac_v6_features import HACFeatureBuilder
        
        # 1. Teste de configuração
        print("1. 🔧 Testando configuração...")
        config = HACConfig()
        print("   ✅ Configuração carregada")
        
        # 2. Teste de features (leve)
        print("2. 🧪 Testando feature builder...")
        feature_builder = HACFeatureBuilder(config)
        datasets = feature_builder.build_all()
        print(f"   ✅ Features geradas: {len(datasets)} horizontes")
        
        # 3. Teste do predictor
        print("3. 🧠 Testando predictor...")
        memory_before = memory_usage()
        
        predictor = get_predictor_github()
        status = predictor.get_status()
        
        memory_after = memory_usage()
        memory_used = memory_after - memory_before
        
        print(f"   ✅ Predictor carregado")
        print(f"   📊 Memória usada: {memory_used:.1f} MB")
        print(f"   🎯 Status: {status}")
        
        # 4. Teste de previsão (se houver modelos)
        if status["models_loaded"] > 0:
            print("4. 🔮 Testando previsão...")
            
            # Usa o primeiro horizonte disponível
            test_horizon = status["available_horizons"][0]
            
            if test_horizon in datasets and datasets[test_horizon]["X"].size > 0:
                X_test = datasets[test_horizon]["X"][-1]  # Última janela
                
                result = predictor.predict_safe(X_test, test_horizon)
                
                print(f"   ✅ Previsão H{test_horizon}:")
                for param, value in result.values.items():
                    print(f"      {param:>10}: {value:8.1f}")
                
                print(f"   📋 Validação: {'✅ VÁLIDO' if result.valid else '❌ INVÁLIDO'}")
                if result.warnings:
                    for warning in result.warnings:
                        print(f"      ⚠️  {warning}")
        
        # 5. Limpeza final
        gc.collect()
        final_memory = memory_usage()
        total_used = final_memory - initial_memory
        
        print(f"\n🎯 RESULTADO FINAL:")
        print(f"   📊 Memória total usada: {total_used:.1f} MB")
        print(f"   ✅ Compatibilidade: OK")
        
        return True
        
    except Exception as e:
        print(f"❌ ERRO: {e}")
        return False

if __name__ == "__main__":
    success = test_github_compatibility()
    sys.exit(0 if success else 1)
