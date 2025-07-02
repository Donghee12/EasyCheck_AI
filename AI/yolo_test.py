from ultralytics import YOLO

# 모델 로드
model = YOLO("best.pt")

# 이미지 추론
results = model("test3.jpg", imgsz=640)  # 이미지 크기 지정
results[0].show()  # 결과 출력 (예: 객체 탐지한 이미지)
