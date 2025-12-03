import pandas as pd

def load_omni_txt(path):
    """
    Converte arquivo OMNI2 TXT para CSV corretamente.
    Formato real esperado:
    YEAR DOY HOUR BX BY BZ BT SPEED DENSITY TEMP PRESSURE
    """

    colnames = [
        "year", "doy", "hour",
        "bx_gsm", "by_gsm", "bz_gsm", "bt",
        "speed", "density", "temperature", "pressure"
    ]

    df = pd.read_csv(
        path,
        delim_whitespace=True,
        header=None,
        names=colnames
    )

    # Converte para datetime real corretamente
    df["datetime"] = (
        pd.to_datetime(df["year"], format="%Y")
        + pd.to_timedelta(df["doy"] - 1, unit="D")
        + pd.to_timedelta(df["hour"], unit="H")
    )

    # Reordena
    df = df[[
        "datetime", "year", "doy", "hour",
        "speed", "bz_gsm", "by_gsm", "bx_gsm", "bt",
        "density", "temperature", "pressure"
    ]]

    return df


if __name__ == "__main__":
    input_file = "omni2_of3LE00pQF.txt"
    output_file = "omni_converted.csv"

    df = load_omni_txt(input_file)
    df.to_csv(output_file, index=False)

    print("✔️ Conversão concluída com ordem física correta!")
    print(f"Arquivo gerado: {output_file}")

    # Quick sanity-check
    print("\n🔍 Estatísticas rápidas:")
    print(df[["speed", "density", "bz_gsm"]].describe())
