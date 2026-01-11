import pandas as pd
from sklearn.model_selection import train_test_split

INPUT = "data/diabetes_012_health_indicators_BRFSS2015.csv"

OUT_OFFLINE = "data/offline.csv"
OUT_ONLINE = "data/online.csv"

df = pd.read_csv(INPUT)

target_col = "Diabetes_012"


offline, online = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df[target_col]
)

offline.to_csv(OUT_OFFLINE, index=False)
online.to_csv(OUT_ONLINE, index=False)

print("DONE")
print("offline size:", offline.shape)
print("online size:", online.shape)
