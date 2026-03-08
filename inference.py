import os
import numpy as np
import onnxruntime as ort
from transformers import WhisperFeatureExtractor

from audio_utils import truncate_audio_to_last_n_seconds

ONNX_MODEL_PATH = os.path.join(os.path.dirname(__file__), "smart-turn-v3.2-cpu.onnx")


def build_session(onnx_path):
    so = ort.SessionOptions()
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(onnx_path, sess_options=so)


feature_extractor = WhisperFeatureExtractor(chunk_length=8)
session = build_session(ONNX_MODEL_PATH)


def predict_endpoint(audio_array):
    """
    Predict whether an audio segment is complete (turn ended) or incomplete.

    Args:
        audio_array: Numpy array containing audio samples at 16kHz

    Returns:
        Dictionary containing prediction results:
        - prediction: 1 for complete, 0 for incomplete
        - probability: Probability of completion (sigmoid output)
    """
    audio_array = truncate_audio_to_last_n_seconds(audio_array, n_seconds=8)

    inputs = feature_extractor(
        audio_array,
        sampling_rate=16000,
        return_tensors="np",
        padding="max_length",
        max_length=8 * 16000,
        truncation=True,
        do_normalize=True,
    )

    input_features = inputs.input_features.squeeze(0).astype(np.float32)
    input_features = np.expand_dims(input_features, axis=0)

    outputs = session.run(None, {"input_features": input_features})

    probability = outputs[0][0].item()
    prediction = 1 if probability > 0.5 else 0

    return {
        "prediction": prediction,
        "probability": probability,
    }
