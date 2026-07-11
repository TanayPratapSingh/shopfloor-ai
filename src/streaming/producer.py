"""Kafka Producer — Replays AI4I records as a live sensor stream."""

import json, time, os, yaml
import numpy as np
from kafka import KafkaProducer
from loguru import logger
from src.data.download import load_dataset


def load_config():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


def stream(interval_sec: float = 1.0, max_records: int = 0):
    config = load_config()
    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", config["kafka"]["bootstrap_servers"])
    topic = config["kafka"]["topics"]["sensor_raw"]

    producer = KafkaProducer(
        bootstrap_servers=bootstrap,
        value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
        acks="all",
    )

    df = load_dataset()
    rng = np.random.default_rng(42)
    logger.info(f"Streaming {len(df)} records to '{topic}' @ {interval_sec}s interval")

    try:
        for i, row in df.iterrows():
            record = row.to_dict()
            record["_stream_index"] = int(i)
            producer.send(topic, key=str(record.get("Type", "L")), value=record)

            if (i + 1) % 50 == 0:
                producer.flush()
                logger.info(f"Sent {i + 1}/{len(df)} | Type={record['Type']} | Failure={record['Machine failure']}")

            if max_records and i + 1 >= max_records:
                break
            time.sleep(interval_sec)
    except KeyboardInterrupt:
        logger.info(f"Stopped at record {i + 1}")
    finally:
        producer.flush()
        producer.close()


if __name__ == "__main__":
    stream(interval_sec=0.5)
