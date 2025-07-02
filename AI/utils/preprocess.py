# utils/preprocess.py
import numpy as np

# 증상 리스트 로드 (미리 저장된 npy)
all_symptoms = np.load("models/all_symptoms.npy", allow_pickle=True)
symptom_index = {s: i for i, s in enumerate(all_symptoms)}


def symptoms_to_vector(symptom_list):
    """영어 증상 리스트 → 모델 입력용 벡터"""
    vector = np.zeros(len(all_symptoms))
    for s in symptom_list:
        if s in symptom_index:
            vector[symptom_index[s]] = 1
    return np.array([vector])  # (1, N) 벡터 반환
