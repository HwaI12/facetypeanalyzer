import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

# Mediapipeセットアップ
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

st.title("顔タイプ分類アプリ")
st.write("リアルタイムで顔タイプを分類します。")

# 利用可能なカメラデバイスをリストアップ
def list_cameras():
    index = 0
    devices = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            break
        devices.append(f"Camera {index}")
        cap.release()
        index += 1
    return devices

available_cameras = list_cameras()

# カメラ選択
camera_id = st.selectbox("使用するカメラを選択してください", range(len(available_cameras)), format_func=lambda x: available_cameras[x])

# 診断関数
def classify_face(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        return image, None

    face_landmarks = results.multi_face_landmarks[0]
    h, w, _ = image.shape
    landmarks = [(int(p.x * w), int(p.y * h)) for p in face_landmarks.landmark]

    # 簡易分類ロジック
    def euclidean_distance(p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    face_width = euclidean_distance(landmarks[454], landmarks[234])  # 左右顔幅
    face_height = euclidean_distance(landmarks[10], landmarks[152])  # 顔の縦長
    aspect_ratio = face_height / face_width

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

# カメラ起動ボタン
if st.button("カメラを起動"):
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        st.error("カメラが検出されませんでした。カメラのアクセス許可を確認してください。")
    else:
        FRAME_WINDOW = st.image([])
        diagnosis_placeholder = st.empty()  # 診断結果を表示するプレースホルダー

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("カメラ映像を取得できませんでした。")
                break

            frame, diagnosis = classify_face(frame)
            FRAME_WINDOW.image(frame, channels="BGR")

            if diagnosis:
                diagnosis_placeholder.write(f"診断結果: {diagnosis}")
            else:
                diagnosis_placeholder.write("診断中...")  # 結果が出るまで一時的に表示

        cap.release()
