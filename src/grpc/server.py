"""
uv run --group grpc python -m src.grpc.server
"""

import sys
from pathlib import Path

sys.path.append(Path(__file__).parent.as_posix())

import os
import random
from concurrent import futures

import inference_pb2
import inference_pb2_grpc
import numpy as np

import grpc


class ModelInferenceServicer(inference_pb2_grpc.ModelInferenceServicer):
    def PredictConfidence(self, request, context):  # noqa: N802
        print("request received")
        features = list(request.features)
        print(f"features: {features}")
        class_id = random.randint(0, 100)
        confidence = sum(features) * random.random()
        print("sending response")
        return inference_pb2.InferenceReply(
            class_id=class_id,
            confidence=confidence,
        )

    def PredictEmbedding(self, request, context):  # noqa: N802
        text = request.text
        print(f"text: {text}")
        embeddings = np.random.rand(5, 1024).astype(np.float16)
        raw_bytes = embeddings.tobytes()
        shape = list(embeddings.shape)
        return inference_pb2.EmbeddingReply(embedding=raw_bytes, shape=shape)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(os.cpu_count()))
    inference_pb2_grpc.add_ModelInferenceServicer_to_server(
        ModelInferenceServicer(), server
    )
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
