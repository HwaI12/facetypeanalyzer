import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import os
import logging

# 警告の抑制
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlowのログを抑制
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # TensorFlowのログを抑制
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)  # abslのログをエラーのみ表示

# Mediapipeセットアップ
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

st.title("顔タイプ分類アプリ")
st.write("リアルタイムで顔タイプを分類します。")

def classify_face(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        return image, "顔が検出されませんでした。"

    face_landmarks = results.multi_face_landmarks[0]
    h, w, _ = image.shape
    landmarks = [(int(p.x * w), int(p.y * h)) for p in face_landmarks.landmark]

    # ランドマークを利用した分類ロジック
    def euclidean_distance(p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    face_width = euclidean_distance(landmarks[454], landmarks[234])  # 左右顔幅
    face_height = euclidean_distance(landmarks[10], landmarks[152])  # 顔の縦長
    aspect_ratio = face_height / face_width

    # ランドマーク間距離を使用した簡易分類
    curve_score = 0
    for pair in [(33, 160), (133, 243), (61, 146), (291, 375)]:
        curve_score += euclidean_distance(landmarks[pair[0]], landmarks[pair[1]])

    if aspect_ratio > 1.4 and curve_score > 250:
        return image, "キュートタイプ"
    elif aspect_ratio > 1.4 and curve_score <= 250:
        return image, "アクティブキュートタイプ"
    elif 1.2 < aspect_ratio <= 1.4 and curve_score > 200:
        return image, "フレッシュタイプ"
    elif 1.2 < aspect_ratio <= 1.4 and curve_score <= 200:
        return image, "クールカジュアルタイプ"
    elif aspect_ratio <= 1.2 and curve_score > 250:
        return image, "フェミニンタイプ"
    elif aspect_ratio <= 1.2 and 200 < curve_score <= 250:
        return image, "ソフトエレガントタイプ"
    elif aspect_ratio <= 1.2 and 150 < curve_score <= 200:
        return image, "エレガントタイプ"
    else:
        return image, "クールタイプ"

if st.button("カメラを起動"):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("カメラが検出されませんでした。カメラのアクセス許可を確認してください。")
    else:
        FRAME_WINDOW = st.image([])
        st.write("カメラ映像を停止するにはページをリロードしてください。")

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("カメラ映像を取得できませんでした。")
                break

            frame, diagnosis = classify_face(frame)
            FRAME_WINDOW.image(frame, channels="BGR")
            st.write("診断結果:", diagnosis)

        cap.release()
