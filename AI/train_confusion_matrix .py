# visualize_training.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
import joblib

# 모델 & 메타 데이터 로드
model = load_model("models/internal_model.h5")
label_encoder = joblib.load("models/label_encoder.pkl")
X_test = np.load("models/X_test.npy")
y_test = np.load("models/y_test.npy")

# 예측
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
labels = label_encoder.classes_

plt.figure(figsize=(12, 10))
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
plt.show()
