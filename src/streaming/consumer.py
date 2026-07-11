"""Kafka Consumer — Processes sensor stream, computes rolling KPIs, triggers alerts."""

import json, os, yaml
from collections import deque
from datetime import datetime
from kafka import KafkaConsumer, KafkaProducer
from loguru import logger


def load_config():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


class StreamProcessor:
    def __init__(self, window: int = 100):
        self.window = deque(maxlen=window)
        self.cooldown: dict[str, float] = {}

    def process(self, record: dict) -> dict:
        self.window.append(record)
        w = list(self.window)

        failures = [r.get("Machine failure", 0) for r in w]
        torques = [r.get("Torque [Nm]", 40) for r in w]
        temps = [r.get("Process temperature [K]", 310) - r.get("Air temperature [K]", 300) for r in w]

        record["rolling_failure_rate"] = round(sum(failures) / len(failures), 4)
        record["rolling_torque_avg"] = round(sum(torques) / len(torques), 2)
        record["rolling_temp_diff_avg"] = round(sum(temps) / len(temps), 2)
        record["window_size"] = len(w)
        record["processed_at"] = datetime.now().isoformat()
        return record

    def check_alerts(self, r: dict) -> list[dict]:
        alerts = []
        now = datetime.now().timestamp()

        def can_alert(t):
            if t in self.cooldown and now - self.cooldown[t] < 60:
                return False
            self.cooldown[t] = now
            return True

        if r.get("rolling_failure_rate", 0) > 0.05 and can_alert("failure_high"):
            alerts.append({"type": "FAILURE_RATE_HIGH", "severity": "CRITICAL",
                           "value": r["rolling_failure_rate"],
                           "message": f"Rolling failure rate {r['rolling_failure_rate']:.1%} exceeds 5% target"})

        torque = r.get("Torque [Nm]", 0)
        if torque > 60 and can_alert("torque_spike"):
            alerts.append({"type": "TORQUE_SPIKE", "severity": "WARNING",
                           "value": torque,
                           "message": f"Torque spike: {torque:.1f} Nm (threshold: 60 Nm)"})

        wear = r.get("Tool wear [min]", 0)
        if wear > 200 and can_alert("tool_wear"):
            alerts.append({"type": "TOOL_WEAR_HIGH", "severity": "WARNING",
                           "value": wear,
                           "message": f"Tool wear at {wear} min — replacement recommended"})

        return alerts


def run():
    config = load_config()
    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", config["kafka"]["bootstrap_servers"])

    consumer = KafkaConsumer(
        config["kafka"]["topics"]["sensor_raw"],
        bootstrap_servers=bootstrap,
        group_id=config["kafka"]["consumer_group"],
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="latest",
    )
    alert_producer = KafkaProducer(
        bootstrap_servers=bootstrap,
        value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
    )

    proc = StreamProcessor(window=100)
    count = 0
    logger.info("Consumer started")

    try:
        for msg in consumer:
            enriched = proc.process(msg.value)
            alerts = proc.check_alerts(enriched)
            count += 1

            if count % 25 == 0:
                logger.info(f"Processed {count} | FailRate={enriched['rolling_failure_rate']:.1%} | Torque={enriched['rolling_torque_avg']:.1f}")

            for a in alerts:
                alert_producer.send(config["kafka"]["topics"]["alerts"], value=a)
                logger.warning(f"ALERT: {a['message']}")
    except KeyboardInterrupt:
        logger.info(f"Stopped. Total: {count}")
    finally:
        consumer.close()
        alert_producer.close()


if __name__ == "__main__":
    run()
