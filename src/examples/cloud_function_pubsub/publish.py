import json

from google.cloud import pubsub_v1

publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path("dev-smoke-276109", "cloud-function-topic")

message = json.dumps({"Mole": "Moje", "Good": "Boy"})
future = publisher.publish(topic_path, message.encode("utf-8"))
