"""
uv run --group grpc python -m src.grpc.client
"""

import sys
from pathlib import Path

sys.path.append(Path(__file__).parent.as_posix())

import random
import time

import inference_pb2
import inference_pb2_grpc
import numpy as np

import grpc


def predict_confidence(stub, features):
    request = inference_pb2.InferenceRequest(features=features)
    response = stub.PredictConfidence(request, timeout=2)
    print(f"class_id: {response.class_id}")
    print(f"confidence: {response.confidence}")


def predict_embedding(stub, text):
    request = inference_pb2.EmbeddingRequest(text=text)
    response = stub.PredictEmbedding(request, timeout=2)
    print(f"shape: {response.shape}")

    array = np.frombuffer(response.embedding, dtype=np.float16)
    print(f"array.shape: {array.shape}")
    print(f"array.dtype: {array.dtype}")
    array = array.reshape(response.shape)
    print(f"array.shape: {array.shape}")
    print(array[:3])


def run():
    channel = grpc.insecure_channel("localhost:50051")
    stub = inference_pb2_grpc.ModelInferenceStub(channel)

    for _ in range(100):
        if random.random() > 0.5:
            features = [
                random.random(),
                random.random(),
                random.random(),
            ]
            predict_confidence(stub, features)
        else:
            text = "hello mole"
            predict_embedding(stub, text)

        time.sleep(1 * random.random())


if __name__ == "__main__":
    run()
