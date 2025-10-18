import time

import flask
import functions_framework
from google.cloud import storage


@functions_framework.http
def hello_world(request: flask.Request):
    gcs = storage.Client()
    bucket = gcs.bucket("lewzey-test")
    blob_name: str = f"{int(time.time())}.txt"
    blob = bucket.blob(blob_name)
    blob.upload_from_string("Hello, world!")
    return {"status": "success", "blob_name": blob_name, "request": request.get_json()}
