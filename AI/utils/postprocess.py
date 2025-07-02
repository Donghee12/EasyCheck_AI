# utils/postprocess.py
import numpy as np

# 질병 클래스 리스트 로드
disease_classes = np.load("models/disease_classes.npy", allow_pickle=True)


def decode_prediction(prediction_array):
    """Softmax 결과 → (질병, 확률)"""
    disease_classes = np.load("models/disease_classes.npy", allow_pickle=True)
    predicted_index = np.argmax(prediction_array)
    predicted_disease = disease_classes[predicted_index]
    confidence = float(prediction_array[0][predicted_index]) * 100
    return predicted_disease, round(confidence, 2)


def decode_top_predictions(prediction_array, top_n=4):
    """Softmax 결과에서 상위 N개의 (질병, 확률%) 튜플 반환"""
    probs = prediction_array[0]
    top_indices = probs.argsort()[-top_n:][::-1]
    result = [(disease_classes[i], round(probs[i] * 100, 2)) for i in top_indices]
    return result
