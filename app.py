import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# mediapipeの顔検出とランドマーク機能を初期化
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# 各顔の特徴ポイント
FACEPT_LIPS = [0, 13, 14, 17, 37, 39, 40, 61, 78, 80,
               81, 82, 84, 87, 88, 91, 95, 146, 178, 181,
               185, 191, 267, 269, 270, 291, 308, 310, 311, 312,
               314, 317, 318, 321, 324, 375, 402, 405, 409, 415]
FACEPT_LEYE = [249, 263, 362, 373, 374, 380, 381, 382, 384, 385,
               386, 387, 388, 390, 398, 466]
FACEPT_LEYEBROW = [276, 282, 283, 285, 293, 295, 296, 300, 334, 336]
FACEPT_LIRIS = [474, 475, 476, 477]
FACEPT_REYE = [7, 33, 133, 144, 145, 153, 154, 155, 157, 158,
               159, 160, 161, 163, 173, 246]
FACEPT_REYEBROW = [46, 52, 53, 55, 63, 65, 66, 70, 105, 107]
FACEPT_RIRIS = [469, 470, 471, 472]
FACEPT_NOSE = [1, 2, 4, 5, 6, 19, 45, 48, 64, 94,
               97, 98, 115, 168, 195, 197, 220, 275, 278, 294,
               326, 327, 344, 440]
FACEPT_OVAL = [10, 21, 54, 58, 67, 93, 103, 109, 127, 132,
               136, 148, 149, 150, 152, 162, 172, 176, 234, 251,
               284, 288, 297, 323, 332, 338, 356, 361, 365, 377,
               378, 379, 389, 397, 400, 454]

def generate_random_color():
    return (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))

def calculate_face_features_and_draw(image_path):
    # 画像を読み込む
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image at {image_path}")
        return None, None
    
    # 画像の幅と高さを取得
    height, width, _ = image.shape

    # BGR -> RGB へ変換
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # mediapipeによる顔検出
    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        # 顔ランドマークを検出
        results = face_mesh.process(image_rgb)
        
        if results.multi_face_landmarks:
            features = []
            for face_landmarks in results.multi_face_landmarks:
                feature = {}
                feature['image_path'] = image_path

                landmarks = [(lm.x * width, lm.y * height) for lm in face_landmarks.landmark]  # 座標をピクセル単位に変換
                
                # 顔の幅と高さの比率
                chin = landmarks[152]
                forehead = landmarks[10]
                face_height = np.linalg.norm(np.array(forehead) - np.array(chin))
                face_width = np.linalg.norm(np.array(landmarks[454]) - np.array(landmarks[234]))
                face_ratio = face_width / face_height
                feature['face_height'] = face_height
                feature['face_width'] = face_width
                feature['face_ratio'] = face_ratio
                
                # 顎の形状（顎の幅）
                jaw_width = np.linalg.norm(np.array(landmarks[454]) - np.array(landmarks[234]))  # 顎の幅
                feature['jaw_width'] = jaw_width
                
                # 目の縦横比（左目）
                left_eye_width = np.linalg.norm(np.array(landmarks[362]) - np.array(landmarks[263]))  # 362から263
                left_eye_height = np.linalg.norm(np.array(landmarks[386]) - np.array(landmarks[374]))  # 386から374
                left_eye_ratio = left_eye_width / left_eye_height
                feature['left_eye_ratio'] = left_eye_ratio

                # 目の縦横比（右目）
                right_eye_width = np.linalg.norm(np.array(landmarks[133]) - np.array(landmarks[33]))  # 133から33
                right_eye_height = np.linalg.norm(np.array(landmarks[159]) - np.array(landmarks[145]))  # 159から145
                right_eye_ratio = right_eye_width / right_eye_height
                feature['right_eye_ratio'] = right_eye_ratio

                # 鼻の高さと幅
                nose_height = np.linalg.norm(np.array(landmarks[1]) - np.array(landmarks[6]))  # 鼻の高さ
                nose_width = np.linalg.norm(np.array(landmarks[64]) - np.array(landmarks[294]))  # 鼻の幅
                feature['nose_height'] = nose_height
                feature['nose_width'] = nose_width

                # 顔面比率（額、鼻、口、顎の比率）
                forehead_to_nose = np.linalg.norm(np.array(landmarks[10]) - np.array(landmarks[1]))  # 額から鼻まで
                nose_to_chin = np.linalg.norm(np.array(landmarks[1]) - np.array(landmarks[152]))  # 鼻から顎まで
                mouth_to_chin = np.linalg.norm(np.array(landmarks[17]) - np.array(landmarks[152]))  # 口から顎まで
                feature['forehead_to_nose_ratio'] = forehead_to_nose / nose_to_chin
                feature['nose_to_chin_ratio'] = nose_to_chin / mouth_to_chin

                # 顔の対称性（左右対称性）
                left_face = landmarks[:len(landmarks)//2]
                right_face = landmarks[len(landmarks)//2:]
                symmetry_score = np.mean([np.linalg.norm(np.array(left_face[i]) - np.array(right_face[i])) for i in range(len(left_face))])
                feature['symmetry_score'] = symmetry_score

                # 目と目の距離
                left_eye_to_right_eye = np.linalg.norm(np.array(landmarks[362]) - np.array(landmarks[133]))
                feature['left_eye_to_right_eye'] = left_eye_to_right_eye

                # 特徴をリストに追加
                features.append(feature)
                
                # 顔の特徴点を線で結ぶ
                # 顔の幅を描画
                cv2.line(image, tuple(map(int, landmarks[234])), tuple(map(int, landmarks[454])), generate_random_color(), 2)

                # 顔の高さを描画
                cv2.line(image, tuple(map(int, landmarks[10])), tuple(map(int, landmarks[152])), generate_random_color(), 2)

                # 顎の幅を描画
                cv2.line(image, tuple(map(int, landmarks[454])), tuple(map(int, landmarks[234])), generate_random_color(), 2)

                # 左目の縦幅を描画
                cv2.line(image, tuple(map(int, landmarks[362])), tuple(map(int, landmarks[263])), generate_random_color(), 2)

                # 左目の横幅を描画
                cv2.line(image, tuple(map(int, landmarks[386])), tuple(map(int, landmarks[374])), generate_random_color(), 2)

                # 右目の縦幅を描画
                cv2.line(image, tuple(map(int, landmarks[133])), tuple(map(int, landmarks[33])), generate_random_color(), 2)

                # 右目の横幅を描画
                cv2.line(image, tuple(map(int, landmarks[159])), tuple(map(int, landmarks[145])), generate_random_color(), 2)

                # 鼻の高さを描画
                cv2.line(image, tuple(map(int, landmarks[1])), tuple(map(int, landmarks[6])), generate_random_color(), 2)

                # 鼻の幅を描画
                cv2.line(image, tuple(map(int, landmarks[64])), tuple(map(int, landmarks[294])), generate_random_color(), 2)

                # 額から鼻までの距離を描画
                cv2.line(image, tuple(map(int, landmarks[10])), tuple(map(int, landmarks[1])), generate_random_color(), 2)

                # 鼻から顎までの距離を描画
                cv2.line(image, tuple(map(int, landmarks[1])), tuple(map(int, landmarks[152])), generate_random_color(), 2)

                # 口から顎までの距離を描画
                cv2.line(image, tuple(map(int, landmarks[17])), tuple(map(int, landmarks[152])), generate_random_color(), 2)

                # 目と目の距離を描画
                cv2.line(image, tuple(map(int, landmarks[362])), tuple(map(int, landmarks[133])), generate_random_color(), 2)

            return image, features
        else:
            return None, None


def process_directory(input_directory, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    all_features = []
    
    for filename in os.listdir(input_directory):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_directory, filename)
            output_image_path = os.path.join(output_directory, f"processed_{filename}")
            
            image, features = calculate_face_features_and_draw(image_path)
            
            if image is not None:
                # 結果画像の保存
                cv2.imwrite(output_image_path, image)
                
                # 特徴をリストに追加
                all_features.extend(features)
    
    # 特徴をCSVに保存
    if all_features:
        df = pd.DataFrame(all_features)
        df.to_csv(os.path.join(output_directory, "face_features.csv"), index=False)
        print("画像の保存とCSVへの書き込みが完了しました。")
    else:
        print("顔の特徴が見つかりませんでした。")

input_directory = "../kao/"  # 画像ファイルが格納されているディレクトリ
output_directory = "kao_land/"  # 処理後の画像とCSVを保存するディレクトリ
process_directory(input_directory, output_directory)
