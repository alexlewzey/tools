import base64
import json
import time

import functions_framework
from google.cloud import storage


@functions_framework.cloud_event
def hello_pubsub(cloud_event):
    message = base64.b64decode(cloud_event.data["message"]["data"]).decode("utf-8")
    print(message)
    data = json.loads(message)

    gcs = storage.Client()
    bucket = gcs.bucket("lewzey-test")
    blob_name: str = f"pubsub_{int(time.time())}.txt"
    blob = bucket.blob(blob_name)
    blob.upload_from_string(json.dumps(data))
    return {"status": "success", "blob_name": blob_name, "data": data}
