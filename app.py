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

    # 子供顔・大人顔診断
    # 1. 顔の形
    face_width = euclidean_distance(landmarks[454], landmarks[234])  # 左右顔幅
    face_height = euclidean_distance(landmarks[10], landmarks[152])  # 顔の縦長
    aspect_ratio = face_height / face_width

    # 2. 顎の長さ
    chin_to_mouth = euclidean_distance(landmarks[152], landmarks[17])

    # 3. 目の位置
    eye_distance = euclidean_distance(landmarks[33], landmarks[263])

    # 4. 鼻根の高さ
    nose_root = euclidean_distance(landmarks[6], landmarks[195])

    # 5. 顔全体の立体感
    facial_depth = euclidean_distance(landmarks[152], landmarks[168])

    # 6. 目の大きさ
    eye_size = euclidean_distance(landmarks[159], landmarks[145])

    # 7. 口の大きさ
    mouth_width = euclidean_distance(landmarks[78], landmarks[308])

    # 子供顔 / 大人顔 判定ロジック
    childlike_score = 0
    adultlike_score = 0

    if aspect_ratio > 1.3:
        adultlike_score += 1
    else:
        childlike_score += 1

    if chin_to_mouth < 0.3 * face_height:
        childlike_score += 1
    else:
        adultlike_score += 1

    if eye_distance > 0.6 * face_width:
        childlike_score += 1
    else:
        adultlike_score += 1

    if nose_root < 0.05 * face_height:
        childlike_score += 1
    else:
        adultlike_score += 1

    if facial_depth < 0.3 * face_width:
        childlike_score += 1
    else:
        adultlike_score += 1

    # 直線・曲線診断
    curve_score = 0
    straight_score = 0

    # 頬の肉感
    cheek_curve = euclidean_distance(landmarks[234], landmarks[132])
    if cheek_curve > 0.4 * face_width:
        curve_score += 1
    else:
        straight_score += 1

    # 目の形状
    if eye_size > 0.2 * face_width:
        curve_score += 1
    else:
        straight_score += 1

    # 唇の厚み
    lip_thickness = euclidean_distance(landmarks[13], landmarks[14])
    if lip_thickness > 0.05 * face_width:
        curve_score += 1
    else:
        straight_score += 1

    # 判定結果
    child_adult_result = "子供顔" if childlike_score > adultlike_score else "大人顔"
    curve_straight_result = "曲線" if curve_score > straight_score else "直線"

    final_result = f"{child_adult_result}, {curve_straight_result}"

    return image, final_result

# カメラ起動と停止のフラグ
if "camera_active" not in st.session_state:
    st.session_state["camera_active"] = False

# カメラ起動ボタン
if st.button("カメラを起動"):
    st.session_state["camera_active"] = True

if st.session_state["camera_active"]:
    if len(available_cameras) == 0:
        st.error("カメラが検出されませんでした。")
    else:
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            st.error("カメラが検出されませんでした。カメラのアクセス許可を確認してください。")
        else:
            FRAME_WINDOW = st.image([])
            diagnosis_placeholder = st.empty()  # 診断結果を表示するプレースホルダー

            while st.session_state["camera_active"]:
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

# カメラ停止ボタン
if st.button("カメラを停止"):
    st.session_state["camera_active"] = False
