# train_internal_model.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os

# 1. 데이터 불러오기
df = pd.read_csv("data/dataset.csv")


# ✅ 1-1. 증상 조합 중복 제거 및 셔플
df["symptom_combo"] = df.iloc[:, 1:].apply(
    lambda row: ",".join(sorted(row.dropna())), axis=1
)
df = df.drop_duplicates(subset=["Disease", "symptom_combo"]).drop(
    columns=["symptom_combo"]
)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 2. 고유 증상 리스트 생성
symptom_cols = df.columns[1:]
all_symptoms = set()
for col in symptom_cols:
    all_symptoms.update(df[col].dropna().str.strip())
all_symptoms = sorted(list(all_symptoms))
symptom_index = {s: i for i, s in enumerate(all_symptoms)}


# 3. X 데이터 생성
def row_to_vector(row):
    vector = np.zeros(len(all_symptoms))
    for col in symptom_cols:
        symptom = row[col]
        if pd.notna(symptom):
            s = symptom.strip()
            if s in symptom_index:
                vector[symptom_index[s]] = 1
    return vector


X = np.array([row_to_vector(row) for _, row in df.iterrows()])

# 4. y 데이터: 질병 라벨 인코딩
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df["Disease"])
y = to_categorical(y_encoded)

# 5. 클래스 가중치 계산
class_weights = compute_class_weight(
    class_weight="balanced", classes=np.unique(y_encoded), y=y_encoded
)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}

# 6. 훈련/검증 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 테스트 데이터 저장 (시각화용)
os.makedirs("models", exist_ok=True)
np.save("models/X_test.npy", X_test)
np.save("models/y_test.npy", y_test)

# 7. 모델 구성
model = Sequential(
    [
        Dense(128, input_shape=(X.shape[1],), activation="relu"),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(y.shape[1], activation="softmax"),
    ]
)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# 8. 콜백 설정
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

# 9. 학습
history = model.fit(
    X_train,
    y_train,
    epochs=500,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    class_weight=class_weight_dict,
)

# 10. 평가
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
print(
    classification_report(y_true, y_pred_classes, target_names=label_encoder.classes_)
)

# 11. 저장
model.save("models/internal_model.h5")
np.save("models/all_symptoms.npy", np.array(all_symptoms))
np.save("models/disease_classes.npy", label_encoder.classes_)
joblib.dump(label_encoder, "models/label_encoder.pkl")

print("✅ 모델 학습 및 저장 완료!")


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# 예측
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
labels = label_encoder.classes_

# 1. 정확도 / 손실 그래프
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.title("Accuracy Curve")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss Curve")
plt.legend()
plt.savefig("models/training_curves.png")
plt.close()

# 2. Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(14, 12))
sns.heatmap(
    cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues"
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("models/confusion_matrix.png")
plt.close()

# 3. 클래스별 정확도 막대그래프
class_accuracies = []
for i, label in enumerate(labels):
    class_mask = y_true == i
    correct = np.sum(y_pred_classes[class_mask] == i)
    total = np.sum(class_mask)
    acc = correct / total if total > 0 else 0
    class_accuracies.append((label, acc))

class_accuracies = sorted(class_accuracies, key=lambda x: x[1], reverse=True)
labels_sorted, accs_sorted = zip(*class_accuracies)

plt.figure(figsize=(12, 6))
sns.barplot(x=list(labels_sorted), y=list(accs_sorted))
plt.xticks(rotation=90)
plt.ylim(0, 1)
plt.title("Class-wise Accuracy")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig("models/classwise_accuracy.png")
plt.close()

# 4. 전체 Precision, Recall, F1
macro_p = precision_score(y_true, y_pred_classes, average="macro")
macro_r = recall_score(y_true, y_pred_classes, average="macro")
macro_f1 = f1_score(y_true, y_pred_classes, average="macro")

print(f"📌 Macro Precision: {macro_p:.4f}")
print(f"📌 Macro Recall:    {macro_r:.4f}")
print(f"📌 Macro F1-score:  {macro_f1:.4f}")
