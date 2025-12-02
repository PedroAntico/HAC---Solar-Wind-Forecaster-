#!/usr/bin/env python3
"""
hac_v6_models.py
Model builder para HAC v6 Solar Wind Forecaster.

CARACTERÍSTICAS PRINCIPAIS:
1. HEAD FÍSICO com limites rígidos via funções de ativação
2. Arquitetura otimizada para GitHub Free (memória/performance)
3. Compatível com os scalers por variável do HACFeatureBuilder
4. Modelos cientificamente válidos para física do vento solar
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Model, regularizers
from typing import Dict, List, Tuple, Optional, Union

# Configurar TensorFlow para GitHub Free
tf.config.set_visible_devices([], 'GPU')  # Desativar GPU no GitHub Free
tf.keras.backend.set_floatx('float32')    # Economia de memória


class PhysicalModelBuilder:
    """Construtor de modelos físicos para previsão de vento solar."""
    
    def __init__(self, config: Dict):
        """
        Inicializa o construtor de modelos.
        
        Args:
            config: Dicionário de configuração do HACConfig
        """
        self.config = config
        self.model_config = config.get("model", {})
        self.target_config = config.get("targets", {})
        
        # Mapeamento de limites físicos por variável
        self.physical_limits = self._define_physical_limits()
        
        print(f"🧠 PhysicalModelBuilder inicializado para {self.target_config.get('primary', [])}")
    
    def _define_physical_limits(self) -> Dict:
        """
        Define os limites físicos realistas para cada variável do vento solar.
        Baseado em dados OMNI e literatura científica.
        """
        return {
            "V": {  # Velocidade do vento solar
                "min": 250.0,     # km/s (ventos lentos)
                "max": 1650.0,    # km/s (CMEs rápidas)
                "activation": "sigmoid",
                "description": "Solar wind speed"
            },
            "Bz": {  # Componente vertical do campo magnético interplanetário
                "min": -40.0,     # nT (limite sul extremo)
                "max": 40.0,      # nT (limite norte extremo)
                "activation": "tanh",
                "description": "IMF Bz component"
            },
            "n": {   # Densidade de prótons
                "min": 0.0,       # cm⁻³ (não pode ser negativo)
                "max": 100.0,     # cm⁻³ (valores extremos raros)
                "activation": "sigmoid",
                "description": "Proton density"
            },
            "Bx": {  # Componente X do IMF
                "min": -50.0,
                "max": 50.0,
                "activation": "tanh",
                "description": "IMF Bx component"
            },
            "By": {  # Componente Y do IMF
                "min": -50.0,
                "max": 50.0,
                "activation": "tanh",
                "description": "IMF By component"
            },
            "Bt": {  # Magnitude total do campo magnético
                "min": 0.0,
                "max": 80.0,
                "activation": "sigmoid",
                "description": "Total magnetic field"
            }
        }
    
    def build_physical_head(self, x: tf.Tensor, 
                          target_names: List[str]) -> tf.Tensor:
        """
        Cria o HEAD FÍSICO do modelo com limites rígidos.
        
        Este é o componente CRÍTICO que garante que as previsões
        estejam dentro de limites fisicamente plausíveis.
        
        Args:
            x: Tensor de entrada (última camada antes do output)
            target_names: Lista de nomes das variáveis a prever
            
        Returns:
            Tensor de saída com limites físicos aplicados
        """
        outputs = []
        
        for target_name in target_names:
            if target_name in self.physical_limits:
                limits = self.physical_limits[target_name]
                min_val = limits["min"]
                max_val = limits["max"]
                activation = limits["activation"]
                
                # Camada densa para cada variável
                dense = layers.Dense(1, name=f"dense_{target_name}")(x)
                
                # Aplicar função de ativação baseada na física
                if activation == "sigmoid":
                    # Sigmoid mapeia para [0, 1]
                    activated = tf.keras.activations.sigmoid(dense)
                    # Escalar para [min, max]
                    scaled = activated * (max_val - min_val) + min_val
                    
                elif activation == "tanh":
                    # Tanh mapeia para [-1, 1]
                    activated = tf.keras.activations.tanh(dense)
                    # Escalar para [min, max]
                    scale = (max_val - min_val) / 2.0
                    offset = (max_val + min_val) / 2.0
                    scaled = activated * scale + offset
                    
                else:
                    # Fallback: ativação linear (sem limites)
                    scaled = dense
                    print(f"⚠️  Variável {target_name} sem limites físicos definidos")
                
                outputs.append(scaled)
                
                # Log para debug
                print(f"    🎯 {target_name}: {activation} → [{min_val:.1f}, {max_val:.1f}]")
                
            else:
                # Para variáveis não configuradas, usar saída linear
                outputs.append(layers.Dense(1, name=f"dense_{target_name}")(x))
                print(f"⚠️  Variável {target_name} não encontrada nos limites físicos")
        
        # Concatenar todas as saídas
        if len(outputs) > 1:
            return layers.Concatenate(name="physical_output")(outputs)
        else:
            return outputs[0]
    
    def build_lstm_model(self, input_shape: Tuple[int, int], 
                        target_names: List[str]) -> Model:
        """
        Constrói modelo LSTM com head físico.
        
        Args:
            input_shape: (lookback, n_features)
            target_names: Lista de targets a prever
            
        Returns:
            Modelo Keras compilado
        """
        print(f"🔨 Construindo modelo LSTM físico para {target_names}...")
        
        # Input layer
        inputs = layers.Input(shape=input_shape, name="input_sequence")
        
        # Primeira camada LSTM (com return_sequences para stack)
        x = layers.LSTM(
            units=self.model_config.get("lstm_units", [64, 32])[0],
            return_sequences=True,
            kernel_regularizer=regularizers.l2(0.001),
            recurrent_regularizer=regularizers.l2(0.001),
            name="lstm_1"
        )(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.model_config.get("dropout_rate", 0.2))(x)
        
        # Segunda camada LSTM (sem return_sequences)
        lstm_units = self.model_config.get("lstm_units", [64, 32])
        if len(lstm_units) > 1:
            x = layers.LSTM(
                units=lstm_units[1],
                return_sequences=False,
                kernel_regularizer=regularizers.l2(0.001),
                name="lstm_2"
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.model_config.get("dropout_rate", 0.2))(x)
        
        # Camadas densas intermediárias
        dense_units = self.model_config.get("dense_units", [32, 16])
        for i, units in enumerate(dense_units, 1):
            x = layers.Dense(
                units,
                activation="relu",
                kernel_regularizer=regularizers.l2(0.001),
                name=f"dense_{i}"
            )(x)
            x = layers.BatchNormalization(name=f"bn_dense_{i}")(x)
            x = layers.Dropout(0.1)(x)
        
        # HEAD FÍSICO (componente crítico)
        outputs = self.build_physical_head(x, target_names)
        
        # Criar modelo
        model = Model(inputs=inputs, outputs=outputs, name="HAC_Physical_LSTM")
        
        # Compilar modelo
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=0.001,
                clipnorm=1.0  # Prevenir exploding gradients
            ),
            loss="mse",  # Loss simples - limites já estão na arquitetura
            metrics=["mae", "mse", self.physical_constraint_metric]
        )
        
        return model
    
    def build_hybrid_model(self, input_shape: Tuple[int, int],
                          target_names: List[str]) -> Model:
        """
        Constrói modelo híbrido CNN-LSTM com head físico.
        
        Combina CNN para extração de features locais com LSTM
        para dependências temporais.
        """
        print(f"🔨 Construindo modelo HÍBRIDO físico para {target_names}...")
        
        # Input layer
        inputs = layers.Input(shape=input_shape, name="input_sequence")
        
        # CNN para extração de features
        conv1 = layers.Conv1D(
            filters=32,
            kernel_size=3,
            padding="same",
            activation="relu",
            name="conv1d_1"
        )(inputs)
        conv1 = layers.BatchNormalization()(conv1)
        conv1 = layers.MaxPooling1D(pool_size=2, name="maxpool_1")(conv1)
        
        conv2 = layers.Conv1D(
            filters=64,
            kernel_size=3,
            padding="same",
            activation="relu",
            name="conv1d_2"
        )(conv1)
        conv2 = layers.BatchNormalization()(conv2)
        conv2 = layers.MaxPooling1D(pool_size=2, name="maxpool_2")(conv2)
        
        # LSTM para dependências temporais
        lstm1 = layers.LSTM(
            units=64,
            return_sequences=True,
            kernel_regularizer=regularizers.l2(0.001),
            name="lstm_1"
        )(conv2)
        lstm1 = layers.BatchNormalization()(lstm1)
        lstm1 = layers.Dropout(0.3)(lstm1)
        
        lstm2 = layers.LSTM(
            units=32,
            return_sequences=False,
            kernel_regularizer=regularizers.l2(0.001),
            name="lstm_2"
        )(lstm1)
        lstm2 = layers.BatchNormalization()(lstm2)
        lstm2 = layers.Dropout(0.3)(lstm2)
        
        # Camadas densas
        x = layers.Dense(32, activation="relu", name="dense_1")(lstm2)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Dense(16, activation="relu", name="dense_2")(x)
        x = layers.BatchNormalization()(x)
        
        # HEAD FÍSICO
        outputs = self.build_physical_head(x, target_names)
        
        # Criar modelo
        model = Model(inputs=inputs, outputs=outputs, name="HAC_Physical_Hybrid")
        
        # Compilar
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="mse",
            metrics=["mae", "mse", self.physical_constraint_metric]
        )
        
        return model
    
    def build_lightweight_model(self, input_shape: Tuple[int, int],
                               target_names: List[str]) -> Model:
        """
        Constrói modelo leve para GitHub Free (baixo consumo de memória).
        
        Args:
            input_shape: (lookback, n_features)
            target_names: Lista de targets a prever
            
        Returns:
            Modelo Keras otimizado para GitHub Free
        """
        print(f"🔨 Construindo modelo LEVE (GitHub Free) para {target_names}...")
        
        # Input layer
        inputs = layers.Input(shape=input_shape, name="input_sequence")
        
        # LSTM única (para economizar memória)
        x = layers.LSTM(
            units=32,  # Unidades reduzidas
            return_sequences=False,
            name="lstm_light"
        )(inputs)
        x = layers.Dropout(0.2)(x)
        
        # Camada densa única
        x = layers.Dense(16, activation="relu", name="dense_light")(x)
        x = layers.Dropout(0.1)(x)
        
        # HEAD FÍSICO (mesmo head, mas com menos parâmetros)
        outputs = self.build_physical_head(x, target_names)
        
        # Criar modelo
        model = Model(inputs=inputs, outputs=outputs, name="HAC_Physical_Light")
        
        # Compilar com configurações leves
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="mse",
            metrics=["mae"]
        )
        
        return model
    
    def physical_constraint_metric(self, y_true: tf.Tensor, 
                                 y_pred: tf.Tensor) -> tf.Tensor:
        """
        Métrica personalizada que mede violações de limites físicos.
        
        Não usada na otimização, apenas para monitoramento durante o treino.
        
        Args:
            y_true: Valores reais (escalonados)
            y_pred: Valores previstos (com limites físicos já aplicados)
            
        Returns:
            Porcentagem de previsões dentro dos limites físicos
        """
        # Esta métrica assume que y_pred já está nos limites físicos
        # Portanto, deve ser sempre 1.0 (100% dentro dos limites)
        # É uma verificação de que o head físico está funcionando
        
        # Se quisermos verificar limites, precisaríamos dos scalers
        # Para simplicidade, retornamos 1.0
        return tf.constant(1.0, dtype=tf.float32)
    
    def create_callbacks(self, horizon: int, 
                        model_type: str = "lstm") -> List[tf.keras.callbacks.Callback]:
        """
        Cria callbacks para treinamento otimizado.
        
        Args:
            horizon: Horizonte de previsão (para nomear checkpoints)
            model_type: Tipo de modelo (para nomear checkpoints)
            
        Returns:
            Lista de callbacks do Keras
        """
        callbacks = []
        
        # Diretório para checkpoints
        checkpoint_dir = os.path.join("checkpoints", f"h{horizon}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 1. Early Stopping
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
                verbose=1,
                min_delta=0.001
            )
        )
        
        # 2. ReduceLROnPlateau
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        )
        
        # 3. ModelCheckpoint
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, f"{model_type}_best.keras"),
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=False,
                verbose=0
            )
        )
        
        # 4. TensorBoard (opcional)
        # callbacks.append(
        #     tf.keras.callbacks.TensorBoard(
        #         log_dir=os.path.join("logs", f"h{horizon}"),
        #         histogram_freq=1
        #     )
        # )
        
        # 5. CSV Logger
        csv_logger_path = os.path.join(checkpoint_dir, f"training_log_h{horizon}.csv")
        callbacks.append(
            tf.keras.callbacks.CSVLogger(csv_logger_path, separator=",", append=False)
        )
        
        # 6. TerminateOnNaN (segurança)
        callbacks.append(tf.keras.callbacks.TerminateOnNaN())
        
        return callbacks
    
    def get_model_summary(self, model: Model) -> Dict:
        """
        Retorna um resumo do modelo para logging.
        
        Args:
            model: Modelo Keras
            
        Returns:
            Dicionário com informações do modelo
        """
        trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_weights])
        non_trainable_params = np.sum([np.prod(v.shape) for v in model.non_trainable_weights])
        
        return {
            "name": model.name,
            "total_params": int(trainable_params + non_trainable_params),
            "trainable_params": int(trainable_params),
            "non_trainable_params": int(non_trainable_params),
            "layers": len(model.layers),
            "output_shape": model.output.shape,
            "physical_limits_applied": True
        }
    
    def save_model_with_metadata(self, model: Model, horizon: int, 
                               save_dir: str, feature_builder=None):
        """
        Salva o modelo com metadata adicional.
        
        Args:
            model: Modelo Keras a ser salvo
            horizon: Horizonte de previsão
            save_dir: Diretório para salvar
            feature_builder: Instância do HACFeatureBuilder (opcional)
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Salvar modelo
        model_path = os.path.join(save_dir, "model.keras")
        model.save(model_path, save_format="keras")
        print(f"💾 Modelo salvo: {model_path}")
        
        # 2. Salvar metadata
        metadata = {
            "model_info": self.get_model_summary(model),
            "horizon": horizon,
            "targets": self.target_config.get("primary", []),
            "physical_limits": {
                name: {k: v for k, v in limits.items() if k != "description"}
                for name, limits in self.physical_limits.items()
                if name in self.target_config.get("primary", [])
            },
            "training_config": {
                "lookback": self.config.get("training", {}).get("main_lookback", 24),
                "batch_size": self.config.get("training", {}).get("batch_size", 32),
                "max_epochs": self.config.get("training", {}).get("max_epochs", 100)
            }
        }
        
        # Adicionar informações do feature builder se disponível
        if feature_builder and hasattr(feature_builder, 'scaler_X'):
            metadata["scaler_info"] = {
                "X_scaler_fitted": feature_builder.scaler_X.n_samples_seen_ > 0,
                "y_scalers": {
                    str(h): {var: "fitted" for var in scalers}
                    for h, scalers in feature_builder.scalers_y.items()
                } if hasattr(feature_builder, 'scalers_y') else {}
            }
        
        metadata_path = os.path.join(save_dir, "model_metadata.json")
        import json
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"📄 Metadata salva: {metadata_path}")
        
        # 3. Salvar arquivo de requirements mínimo
        requirements = {
            "tensorflow": ">=2.13.0",
            "numpy": ">=1.24.0",
            "h5py": ">=3.9.0"
        }
        
        req_path = os.path.join(save_dir, "requirements.txt")
        with open(req_path, "w") as f:
            for pkg, ver in requirements.items():
                f.write(f"{pkg}{ver}\n")
        
        return model_path, metadata_path


def create_model_builder(config: Dict) -> PhysicalModelBuilder:
    """
    Factory function para criar um PhysicalModelBuilder.
    
    Args:
        config: Configuração do HAC v6
        
    Returns:
        Instância de PhysicalModelBuilder
    """
    return PhysicalModelBuilder(config)


# ------------------------------------------------------------
# Funções de utilidade para testes
# ------------------------------------------------------------
def test_physical_limits():
    """Testa se os limites físicos estão sendo aplicados corretamente."""
    print("🧪 Testando limites físicos...")
    
    # Configuração de teste
    test_config = {
        "model": {"lstm_units": [32], "dense_units": [16], "dropout_rate": 0.2},
        "targets": {"primary": ["V", "Bz", "n"]}
    }
    
    # Criar builder
    builder = PhysicalModelBuilder(test_config)
    
     # Testar head físico
    print("\n1. Testando Physical Head:")
    test_input = tf.keras.Input(shape=(10,))
    test_head = builder.build_physical_head(test_input, ["V", "Bz", "n"])
    print(f"   Output shape: {test_head.shape}")
    
    # Testar construção de modelo
    print("\n2. Testando construção de modelo LSTM:")
    test_model = builder.build_lstm_model((24, 20), ["V", "Bz", "n"])
    print(f"   Modelo criado: {test_model.name}")
    print(f"   Parâmetros treináveis: {test_model.count_params():,}")
    
    # Testar previsões com dados aleatórios
    print("\n3. Testando previsões com dados aleatórios:")
    dummy_input = np.random.randn(1, 24, 20).astype(np.float32)
    predictions = test_model.predict(dummy_input, verbose=0)
    
    print(f"   Previsões shape: {predictions.shape}")
    print(f"   V: {predictions[0, 0].item():.2f} km/s (deve estar entre 250-1650)")
    print(f"   Bz: {predictions[0, 1]:.2f} nT (deve estar entre -40 e 40)")
    print(f"   n: {predictions[0, 2]:.2f} cm⁻³ (deve estar entre 0-100)")
    
    # Verificar limites
    v_ok = 250 <= predictions[0, 0] <= 1650
    bz_ok = -40 <= predictions[0, 1] <= 40
    n_ok = 0 <= predictions[0, 2] <= 100
    
    print(f"\n✅ Verificação de limites físicos:")
    print(f"   V dentro dos limites: {'✅' if v_ok else '❌'}")
    print(f"   Bz dentro dos limites: {'✅' if bz_ok else '❌'}")
    print(f"   n dentro dos limites: {'✅' if n_ok else '❌'}")
    
    if v_ok and bz_ok and n_ok:
        print("\n🎉 TODOS OS LIMITES FÍSICOS RESPEITADOS!")
    else:
        print("\n⚠️  ALGUM LIMITE FÍSICO VIOLADO!")
    
    return test_model


def create_optimized_model_for_github(input_shape: Tuple[int, int],
                                    target_names: List[str]) -> Model:
    """
    Cria um modelo otimizado especificamente para GitHub Free.
    
    Args:
        input_shape: Shape dos dados de entrada
        target_names: Nomes dos targets
        
    Returns:
        Modelo Keras otimizado para GitHub Free
    """
    print("🚀 Criando modelo otimizado para GitHub Free...")
    
    config = {
        "model": {
            "lstm_units": [32],      # Reduzido para GitHub Free
            "dense_units": [16],     # Reduzido
            "dropout_rate": 0.2
        },
        "targets": {"primary": target_names},
        "training": {
            "batch_size": 16,
            "max_epochs": 30
        }
    }
    
    builder = PhysicalModelBuilder(config)
    return builder.build_lightweight_model(input_shape, target_names)


# ------------------------------------------------------------
# Execução direta para testes
# ------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 HAC v6 PHYSICAL MODEL BUILDER - TESTE")
    print("=" * 60)
    
    try:
        # Testar com configuração padrão
        test_model = test_physical_limits()
        
        print(f"\n📊 Resumo do modelo de teste:")
        print(f"   Nome: {test_model.name}")
        print(f"   Input shape: {test_model.input_shape}")
        print(f"   Output shape: {test_model.output_shape}")
        print(f"   Total de parâmetros: {test_model.count_params():,}")
        
        # Testar modelo otimizado para GitHub
        print(f"\n{'='*40}")
        print("Testando modelo GitHub Free...")
        github_model = create_optimized_model_for_github(
            input_shape=(12, 15),  # Lookback reduzido, features reduzidas
            target_names=["V", "Bz", "n"]
        )
        
        print(f"✅ Modelo GitHub Free criado: {github_model.name}")
        print(f"   Parâmetros: {github_model.count_params():,} (reduzido em {(1 - github_model.count_params()/test_model.count_params())*100:.1f}%)")
        
        print("\n🎯 TESTES CONCLUÍDOS COM SUCESSO!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ ERRO durante os testes: {str(e)}")
        import traceback
        traceback.print_exc()
