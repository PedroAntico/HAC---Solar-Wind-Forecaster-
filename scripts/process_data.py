import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

LOOKBACK = 168   # 7 dias
FORECAST = 1     # previsão para 1 hora à frente

TARGETS = ["speed", "bz_gsm", "density"]

def create_sequences(df, lookback, forecast):
    X, y = [], []
    for i in range(len(df) - lookback - forecast):
        X.append(df.iloc[i:i+lookback].values)
        y.append(df.iloc[i+lookback+forecast-1][TARGETS].values)
    return np.array(X), np.array(y)

if __name__ == "__main__":
    print("📥 Lendo omni_prepared.csv ...")
    df = pd.read_csv("omni_prepared.csv")

    print("🔧 Normalizando features ...")

    scaler = StandardScaler()
    df_scaled = df.copy()
    numeric_cols = df.columns.drop(["datetime"])

    df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Salva scaler para uso no predictor
    joblib.dump(scaler, "scaler.pkl")

    print("🧱 Criando janelas temporais ...")
    X, y = create_sequences(df_scaled[numeric_cols], LOOKBACK, FORECAST)

    print("📊 Formato das matrizes:")
    print("X:", X.shape)
    print("y:", y.shape)

    np.save("X.npy", X)
    np.save("y.npy", y)

    print("✅ Dados processados gerados!")
    print("Arquivos: X.npy, y.npy, scaler.pkl")
