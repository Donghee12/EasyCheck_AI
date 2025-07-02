from PIL import Image
import numpy as np
from flask import Flask, render_template, request
import os
from dotenv import load_dotenv
import google.generativeai as genai
import tensorflow as tf
from difflib import get_close_matches
from utils.preprocess import symptoms_to_vector, all_symptoms
from utils.postprocess import decode_top_predictions
from utils.text_to_html import extract_html_inside_codeblock
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import tempfile
import pandas as pd


# 초기 설정
app = Flask(__name__)
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# TensorFlow 모델 로드
internal_model = tf.keras.models.load_model("models/internal_model.h5")

# 예방 수칙 CSV 로드
precaution_df = pd.read_csv("data/symptom_precaution.csv")
precaution_dict = precaution_df.set_index("Disease").T.to_dict("list")

# YOLO 모델 로드 (best.pt)
model = YOLO("best.pt")

# 증상 리스트 설정
df = pd.read_csv("data/dataset.csv")
symptom_columns = [col for col in df.columns if col.startswith("Symptom_")]
symptom_set = set()
for col in symptom_columns:
    symptom_set.update(df[col].dropna())

allowed_symptoms = sorted(s.replace(" ", "_").strip().lower() for s in symptom_set if s)
allowed_symptoms_text = ", ".join(allowed_symptoms)


# 홈 페이지
@app.route("/")
def index():
    return render_template("index.html")


def normalize_symptoms(symptoms_en, all_symptoms, cutoff=0.6):
    normalized = []
    for s in symptoms_en:
        match = get_close_matches(s, all_symptoms, n=1, cutoff=cutoff)
        if match:
            normalized.append(match[0])
        else:
            normalized.append(s)
    return normalized


# 내과 진단 페이지
@app.route("/internal", methods=["GET", "POST"])
def internal():
    if request.method == "POST":
        raw_input = request.form["symptoms"]
        symptom_list_ko = [s.strip() for s in raw_input.split(",")]

        translate_prompt = (
            f"Translate the following Korean medical symptoms into English, using ONLY the terms from this predefined list:\n"
            f"{allowed_symptoms_text}\n\n"
            f"Do not infer or add any symptoms. Only use exact matches. Return comma-separated English keywords.\n"
            f"Translate: {', '.join(symptom_list_ko)}"
        )

        # Gemini 번역 결과 처리
        translation = gemini_model.generate_content(translate_prompt).text
        symptoms_en_raw = [s.strip().lstrip("_") for s in translation.split(",")]

        print("Gemini translated (cleaned):", symptoms_en_raw)

        # 정규화
        symptoms_en = normalize_symptoms(symptoms_en_raw, all_symptoms)

        input_vector = symptoms_to_vector(symptoms_en)
        prediction = internal_model.predict(input_vector)

        top_predictions = decode_top_predictions(prediction)
        predicted_disease, confidence = top_predictions[0]
        others = top_predictions[1:]

        precaution_list = precaution_dict.get(predicted_disease, [])
        formatted_precaution = "\n".join(f"- {item}" for item in precaution_list)
        others_text = "\n".join([f"- {d}: {c:.2f}%" for d, c in others])

        advice_prompt = (
            f"사용자가 보고한 증상: {', '.join(symptoms_en)}\n"
            f"가장 가능성 높은 진단: {predicted_disease} ({confidence:.2f}%)\n\n"
            f"기타 예측된 질병:\n{others_text}\n\n"
            f"이 질병의 예방 수칙:\n{formatted_precaution}\n\n"
            f"이 정보를 참고하여 사용자에게 친절한 한국어 설명을 해 주세요.\n"
            f"- 핵심 요약 1문단\n"
            f"- 마지막에 '※ 본 설명은 의학적 진단을 대체하지 않습니다.'로 끝내 주세요."
        )

        result = gemini_model.generate_content(advice_prompt).text
        formatted_advice = result.replace("\n", "<br>")  # 줄바꿈을 <br>로 변환

        print("Korean input:", raw_input)
        print("Normalized symptoms:", symptoms_en)

        return render_template(
            "result.html",
            result=formatted_advice,
            predicted_disease=predicted_disease,
            confidence=confidence,
            diagnosis_type="내과",  # 내과 진단
            others=others,
        )

    return render_template("internal.html")


@app.route("/advice/<disease>")
def detailed_advice(disease):
    prompt = (
        f"'{disease}'이라는 질병에 대한 더 자세한 건강 관리 정보를 다음과 같은 HTML 구조로 출력해 주세요.\n"
        f"단, 코드 블록(```html ... ```) 없이 HTML 태그만 출력해 주세요:\n"
        f"<h3>질병 설명</h3>\n<p>...</p>\n<h3>예방법</h3>\n<ul><li>...</li></ul>\n<h3>대처법</h3>\n<ul><li>...</li></ul>"
    )
    raw = gemini_model.generate_content(prompt).text
    html = extract_html_inside_codeblock(raw)
    return render_template("advice.html", disease=disease, content=html)


# 업로드 폴더 설정
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# 허용된 파일 확장자 체크 함수
def allowed_file(filename):
    allowed_extensions = {"png", "jpg", "jpeg", "gif"}  # 허용할 이미지 확장자
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


# 외과 진단
@app.route("/surgical", methods=["GET", "POST"])
def surgical():
    if request.method == "POST":
        # 이미지 파일 받기
        file = request.files["image"]

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # YOLO 모델로 추론
            results = model(filepath, imgsz=640)  # 이미지 크기 설정

            # 객체가 하나도 감지되지 않은 경우
            if len(results[0].boxes) == 0:
                return render_template(
                    "surgical.html",
                    error="이미지에서 객체를 감지하지 못했습니다. 다른 이미지를 업로드해주세요.",
                )

            img_result = results[0].plot()  # 예측 결과 이미지 얻기

            # numpy.ndarray 형식의 이미지를 PIL 이미지로 변환
            pil_img = Image.fromarray(img_result)

            # 예측된 이미지를 저장
            result_img_path = os.path.join(
                app.config["UPLOAD_FOLDER"], "result_image.jpg"
            )
            pil_img.save(result_img_path)  # 이제 save() 메서드 사용 가능

            # 가장 가능성 높은 질병 예측 (confidence 값 포함)
            predicted_disease = results[0].names[
                results[0].boxes.cls[0].item()
            ]  # 수정된 부분
            confidence = results[0].boxes.conf[0].item() * 100  # confidence 값 추출

            # Gemini 모델에 병에 대한 해결법 요청 (한국어로)
            advice_prompt = (
                f"이 이미지는 '{predicted_disease}'이라는 질병을 나타냅니다. "
                f"이 질병에 대한 자세한 치료 방법과 예방 수칙을 한국어로 제공해 주세요. "
                f"이 진단의 신뢰도는 {confidence:.2f}%입니다. "
                f"또한, 아래와 같은 상태에 대한 해석도 포함하여 설명해 주세요:\n"
                f"1. rash (알레르기)\n"
                f"2. cut (베인상처)\n"
                f"3. bruise (타박상)\n"
                f"4. blister (물집)\n"
                f"를 의미하며 해당 이미지에 대해 올바른 질병만 설명해주세요 즉 결과가 cut이 나올경우 cut에 대해서만 예방 방법을 써주세요\n"
                f"- 핵심 요약 1문단\n"
                f"- 마지막에 '※ 본 설명은 의학적 진단을 대체하지 않습니다.'로 끝내 주세요."
            )

            # Gemini에서 응답을 받음
            result = gemini_model.generate_content(advice_prompt).text
            formatted_advice = result.replace("\n", "<br>")  # 줄바꿈을 <br>로 변환
            return render_template(
                "result.html",
                result_image=result_img_path,
                predicted_disease=predicted_disease,
                confidence=confidence,
                diagnosis_type="외과",  # 외과 진단
                result=formatted_advice,
            )

    return render_template("surgical.html")


# 결과 페이지
@app.route("/result")
def result():
    return render_template("result.html", result="결과가 여기에 표시됩니다.")


@app.route("/internal/loading", methods=["POST"])
def internal_loading():
    symptoms = request.form["symptoms"]
    return render_template("loading.html", symptoms=symptoms)


# Flask 실행
if __name__ == "__main__":
    app.run(debug=True)
