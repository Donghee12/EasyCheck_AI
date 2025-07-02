# 🧠 AI 건강 진단 웹 애플리케이션

AI 기반의 **내과 및 외과 진단 웹 서비스**입니다. 사용자의 텍스트 증상이나 이미지 입력을 통해 가능한 질병을 예측하고, Google Gemini API를 통해 한국어로 된 친절한 설명과 예방 정보를 제공합니다.

---

## 📋 목차

- [소개](#소개)
- [주요 기능](#주요-기능)
- [설치 방법](#설치-방법)
- [사용법](#사용법)
- [의존성](#의존성)
- [디렉토리 구조](#디렉토리-구조)
- [HTML 템플릿](#html-템플릿)
- [예시](#예시)
- [문제 해결](#문제-해결)
- [기여자](#기여자)

---

## 🩺 소개

이 웹 애플리케이션은 두 가지 진단 기능을 제공합니다:

- **내과 진단**:
  - 사용자가 한국어로 증상을 입력
  - Gemini 모델이 영어로 변환 후 증상 벡터화
  - TensorFlow 모델이 질병 예측 및 확률 출력

- **외과 진단**:
  - 상처/피부 이상 이미지 업로드
  - YOLOv8 모델이 질병 분류
  - Gemini가 해당 질병 설명 및 예방법 생성

---

## ⚙️ 주요 기능

- 🤖 증상 기반 예측 (내과)
- 🖼 이미지 기반 예측 (외과)
- 🌐 Gemini를 통한 한국어 결과 생성
- 💬 예방 수칙 자동 로딩 (CSV)
- 📈 확률 기반 상위 질병 리스트 제공
- 💡 UI 로딩 애니메이션 제공

---

## 💾 설치 방법

### 1. 패키지 설치
```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정
`.env` 파일 생성 후 아래 내용 추가:
```
GEMINI_API_KEY=your_google_gemini_api_key
```

### 3. 모델 및 데이터 준비
```
models/
├── internal_model.h5
├── disease_classes.npy
├── all_symptoms.npy

data/
├── dataset.csv
├── symptom_precaution.csv

best.pt  # YOLO 모델 가중치
```

### 4. 서버 실행
```bash
python app.py
```

---

## 🧪 사용법

### ✅ 내과 진단
- `/internal` 경로에서 증상 입력
- 분석 결과 및 예방 수칙, 추가 후보 질병 출력

### ✅ 외과 진단
- `/surgical` 경로에서 이미지 업로드
- 예측된 질병 결과 이미지 및 설명 출력

---

## 📦 의존성

- Flask
- TensorFlow
- Ultralytics YOLO
- Pillow
- numpy
- pandas
- python-dotenv
- google-generativeai

---

## 📁 디렉토리 구조

```
.
├── app.py
├── .env
├── models/
│   ├── internal_model.h5
│   ├── disease_classes.npy
│   └── all_symptoms.npy
├── data/
│   ├── dataset.csv
│   ├── symptom_Description.csv
│   ├── Symptom-severity.csv
│   └── symptom_precaution.csv
├── utils/
│   ├── preprocess.py
│   ├── postprocess.py
│   └── text_to_html.py
├── templates/
│   ├── index.html
│   ├── internal.html
│   ├── result.html
│   ├── surgical.html
│   └── advice.html
├── static/
│   └── uploads/
│   ├── index.css
│   ├── internal.css
│   ├── result.css
│   ├── surgical.css
│   ├── internal.png
│   ├── logo.png
│   └── urgical.png
└── best.pt
```

---

## 🖼 HTML 템플릿

- **index.html / internal.html**:  
  내과 진단용 입력 UI 및 로딩 애니메이션 포함

- **surgical.html**:  
  이미지 업로드 기반 외과 진단 UI

- **advice.html**:  
  Gemini가 생성한 질병 설명 HTML 출력용 템플릿

---

## 🧪 예시

- **입력 (내과)**:  
  `"열이 나고 두통이 있어요"`

- **출력 (예시)**:
  ```
  - 예측 질병: Meningitis (93.2%)
  - 예방 수칙:
    - 백신 접종
    - 청결 유지
    ...
  ※ 본 설명은 의학적 진단을 대체하지 않습니다.
  ```

- **입력 (외과)**:  
  베인 상처 이미지 업로드

- **출력**:
  - 예측 질병: cut
  - 이미지 표시 및 설명 출력

---

## 🛠 문제 해결

| 문제 상황                    | 해결 방법                                   |
|-----------------------------|---------------------------------------------|
| 증상 번역 결과 이상함        | 증상 리스트가 누락되지 않았는지 확인         |
| 이미지 예측 결과 없음        | 이미지 화질 또는 YOLO 모델 가중치 확인       |
| Gemini API 응답 없음         | API 키 유효성 및 요청 제한 확인             |
| 모델 파일 불러오기 실패      | `models/` 경로 확인 및 파일 유무 확인        |

---

## 👥 기여자

- **개발자**: *김동희*
- **AI 통합**: Gemini, YOLOv8, TensorFlow


