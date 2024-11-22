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

if len(available_cameras) == 0:
    st.error("利用可能なカメラが検出されませんでした。")
else:
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

    # 特徴量計算
    def euclidean_distance(p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    # 顔の縦横比 (全体的な形状)
    face_width = euclidean_distance(landmarks[454], landmarks[234])  # 左右顔幅（目尻間）
    face_height = euclidean_distance(landmarks[10], landmarks[152])  # 顔の縦長（おでこから顎まで）
    aspect_ratio = face_height / face_width

    # 顔全体の曲線性（主に目・唇・輪郭から計算）
    curve_score = 0
    for pair in [(33, 160), (133, 243), (61, 146), (291, 375), (17, 151)]:
        curve_score += euclidean_distance(landmarks[pair[0]], landmarks[pair[1]])

    # 額と顎の比率（直線的か曲線的か）
    forehead_to_chin = euclidean_distance(landmarks[10], landmarks[152])
    cheek_to_cheek = euclidean_distance(landmarks[234], landmarks[454])

    # 頬骨と顎の比率（顔の立体感を測定）
    jawline_score = euclidean_distance(landmarks[152], landmarks[377])

    # 顔タイプ分類
    if aspect_ratio > 1.5 and curve_score > 300:
        return image, "キュートタイプ（可愛らしい雰囲気）"
    elif aspect_ratio > 1.5 and curve_score <= 300:
        return image, "アクティブキュートタイプ（元気で個性的な雰囲気）"
    elif 1.3 < aspect_ratio <= 1.5 and curve_score > 250:
        return image, "フレッシュタイプ（爽やかで親しみやすい）"
    elif 1.3 < aspect_ratio <= 1.5 and curve_score <= 250:
        return image, "クールカジュアルタイプ（ボーイッシュでかっこいい）"
    elif aspect_ratio <= 1.3 and curve_score > 300:
        return image, "フェミニンタイプ（柔らかく女性らしい雰囲気）"
    elif aspect_ratio <= 1.3 and 250 < curve_score <= 300:
        return image, "ソフトエレガントタイプ（清楚で落ち着いた印象）"
    elif aspect_ratio <= 1.3 and 200 < curve_score <= 250:
        return image, "エレガントタイプ（華やかで上品な雰囲気）"
    else:
        return image, "クールタイプ（シャープで洗練された印象）"

# カメラ起動ボタン
if st.button("カメラを起動"):
    if len(available_cameras) == 0:
        st.error("カメラが検出されませんでした。")
    else:
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
