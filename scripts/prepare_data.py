import pandas as pd
import numpy as np

def prepare_omni(df):
    """
    Limpeza e preparo dos dados convertidos do OMNI2.
    """
    # Converte datetime
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")

    # Remove duplicadas
    df = df.drop_duplicates(subset=["datetime"])

    # Interpola valores faltantes (método recomendado para OMNI2)
    df = df.set_index("datetime").interpolate(method="time")

    # Remove outliers absurdos
    df["speed"] = df["speed"].clip(lower=200, upper=2000)
    df["density"] = df["density"].clip(lower=0, upper=50)
    df["bz_gsm"] = df["bz_gsm"].clip(lower=-50, upper=50)
    df["bt"] = df["bt"].clip(lower=0, upper=60)

    # Feature engineering básico
    df["dynamic_pressure"] = df["density"] * df["speed"]**2 * 1e-6
    df["bz_neg"] = df["bz_gsm"].apply(lambda x: x if x < 0 else 0)

    # Rolling features
    df["speed_ma3"] = df["speed"].rolling(3).mean()
    df["bz_ma3"] = df["bz_gsm"].rolling(3).mean()
    df["density_ma3"] = df["density"].rolling(3).mean()

    df = df.dropna()

    return df.reset_index()


if __name__ == "__main__":
    print("🔍 Lendo omni_converted.csv ...")
    df = pd.read_csv("omni_converted.csv")

    print("⚙️ Preparando dados ...")
    df2 = prepare_omni(df)

    output = "omni_prepared.csv"
    df2.to_csv(output, index=False)

    print("✅ Arquivo gerado:", output)
