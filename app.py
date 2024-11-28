import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import pandas as pd
import os

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# 顔タイプ診断関数
def classify_face_type(image, file_name):
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        # BGR → RGBに変換
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            return None, "顔が検出されませんでした", {}

        # 顔ランドマークを取得
        face_landmarks = results.multi_face_landmarks[0]
        h, w, _ = image.shape
        landmarks = [(int(p.x * w), int(p.y * h)) for p in face_landmarks.landmark]

        # 特徴量計算
        face_width = np.linalg.norm(np.array(landmarks[454]) - np.array(landmarks[234]))  # 顔幅
        face_height = np.linalg.norm(np.array(landmarks[10]) - np.array(landmarks[152]))  # 顔の高さ
        aspect_ratio = face_height / face_width
        
        # 顎の長さ（唇下）
        chin_to_mouth = np.linalg.norm(np.array(landmarks[152]) - np.array(landmarks[17]))
        
        # 目の位置
        eye_distance = np.linalg.norm(np.array(landmarks[33]) - np.array(landmarks[263]))

        # 顔の立体感（鼻根と顎の高さ）
        facial_depth = np.linalg.norm(np.array(landmarks[152]) - np.array(landmarks[168]))
        
        # 目の大きさ
        eye_size = np.linalg.norm(np.array(landmarks[159]) - np.array(landmarks[145]))

        # 口の大きさ
        mouth_width = np.linalg.norm(np.array(landmarks[78]) - np.array(landmarks[308]))
        
        # 顔タイプ分類（子供顔 or 大人顔）
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

        if facial_depth < 0.3 * face_width:
            childlike_score += 1
        else:
            adultlike_score += 1

        if eye_size > 0.2 * face_width:
            childlike_score += 1
        else:
            adultlike_score += 1

        # 子供顔 / 大人顔 判定
        face_type = "子供顔" if childlike_score > adultlike_score else "大人顔"

        # 直線・曲線分類（直線 or 曲線）
        curve_score = 0
        straight_score = 0
        
        # 頬の肉感
        cheek_curve = np.linalg.norm(np.array(landmarks[234]) - np.array(landmarks[132]))
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
        lip_thickness = np.linalg.norm(np.array(landmarks[13]) - np.array(landmarks[14]))
        if lip_thickness > 0.05 * face_width:
            curve_score += 1
        else:
            straight_score += 1
        
        # 直線・曲線判定
        curve_straight_result = "曲線" if curve_score > straight_score else "直線"
        
        # ランドマーク画像の保存
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(
            annotated_image,
            face_landmarks,
            mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1),
            connection_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1))
        
        image_name = file_name.split('.')[0]
        image_path = f'./landmark_images/{image_name}_landmarks.jpg'
        cv2.imwrite(image_path, annotated_image)

        # 最終結果を辞書として返す
        features = {
            "file_name": file_name,
            "face_width": face_width,
            "face_height": face_height,
            "aspect_ratio": aspect_ratio,
            "chin_to_mouth": chin_to_mouth,
            "eye_distance": eye_distance,
            "facial_depth": facial_depth,
            "eye_size": eye_size,
            "mouth_width": mouth_width,
            "childlike_score": childlike_score,
            "adultlike_score": adultlike_score,
            "face_type": face_type,
            "curve_score": curve_score,
            "straight_score": straight_score,
            "curve_straight_result": curve_straight_result,
            "landmark_image_path": image_path
        }

        return annotated_image, features


# StreamlitアプリのUI
def main():
    st.title("顔タイプ診断アプリ")
    
    st.write("顔画像をアップロードして、顔タイプ診断を行います。")
    
    # 画像ファイルをアップロード
    uploaded_file = st.file_uploader("画像ファイルを選択", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # 画像を読み込む
        image = Image.open(uploaded_file)
        image = np.array(image)
        
        # 顔タイプ診断を実行
        classified_image, features = classify_face_type(image, uploaded_file.name)
        
        if classified_image is not None:
            # 診断結果を表示
            st.image(classified_image, caption='顔ランドマークと診断結果', use_container_width=True)
            st.subheader(f"診断結果: {features['face_type']}, {features['curve_straight_result']}")
            
            # 診断結果をCSVに保存
            result_df = pd.DataFrame([features])
            result_df.to_csv("face_diagnosis.csv", mode="a", header=False, index=False)
            st.success("結果を保存しました。")

if __name__ == "__main__":
    main()
