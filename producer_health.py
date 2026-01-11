import json
import time
import pandas as pd
from kafka import KafkaProducer

BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC = "health_data"

INPUT_CSV = "online.csv"

LABEL_COL = "Diabetes_012"

def main():
    df = pd.read_csv(INPUT_CSV)

    if LABEL_COL in df.columns:
        df = df.drop(columns=[LABEL_COL])

    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        linger_ms=10,
    )

    sent = 0
    for _, row in df.iterrows():
        msg = row.to_dict()
        producer.send(TOPIC, msg)
        sent += 1

        if sent % 50 == 0:
            producer.flush()
            time.sleep(0.1)

    producer.flush()
    producer.close()
    print(f"Done. Sent {sent} messages to topic '{TOPIC}'.")

if __name__ == "__main__":
    main()
