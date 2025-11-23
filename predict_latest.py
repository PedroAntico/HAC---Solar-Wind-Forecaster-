#!/usr/bin/env python3
import pandas as pd
from hac_v6_predictor import HACv6Predictor

pred = HACv6Predictor()

df = pd.read_csv("data_real/omni_prepared.csv")
df = df.drop(columns=["datetime"], errors="ignore")

print("Latest predictions:")
for h in [1, 3, 6, 12, 24, 48]:
    try:
        r = pred.predict(df, h)
        print(f"H{h}h:", r)
    except:
        pass
