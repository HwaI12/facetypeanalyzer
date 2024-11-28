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

                landmarks = [(lm.x, lm.y) for lm in face_landmarks.landmark]
                
                # 下唇の中心から顎までの直線距離
                chin = landmarks[152]
                bottom_lip = landmarks[17]
                chin_distance = np.linalg.norm(np.array(chin) - np.array(bottom_lip))
                feature['chin_to_lip_distance'] = chin_distance
                
                # 目の間の距離
                left_eye_inner = landmarks[362]
                right_eye_inner = landmarks[133]
                eye_distance = np.linalg.norm(np.array(left_eye_inner) - np.array(right_eye_inner))
                feature['eye_distance'] = eye_distance
                
                # 目の横幅と縦幅（左目）
                left_eye_width = np.linalg.norm(np.array(landmarks[362]) - np.array(landmarks[263]))  # 362から263
                left_eye_height = np.linalg.norm(np.array(landmarks[386]) - np.array(landmarks[374]))  # 386から374
                feature['left_eye_width'] = left_eye_width
                feature['left_eye_height'] = left_eye_height

                # 目の横幅と縦幅（右目）
                right_eye_width = np.linalg.norm(np.array(landmarks[133]) - np.array(landmarks[33]))  # 133から33
                right_eye_height = np.linalg.norm(np.array(landmarks[159]) - np.array(landmarks[145]))  # 159から145
                feature['right_eye_width'] = right_eye_width
                feature['right_eye_height'] = right_eye_height

                # 口角から口角までの距離
                left_corner = landmarks[61]
                right_corner = landmarks[291]
                mouth_width = np.linalg.norm(np.array(left_corner) - np.array(right_corner))
                feature['mouth_width'] = mouth_width
                
                # 唇の厚み（上唇と下唇の前後方向の距離）
                upper_lip = landmarks[0]
                lower_lip = landmarks[17]
                lip_thickness = np.linalg.norm(np.array(upper_lip) - np.array(lower_lip))
                feature['lip_thickness'] = lip_thickness

                # 小鼻の横幅
                nostril_left = landmarks[64]
                nostril_right = landmarks[294]
                nostril_width = np.linalg.norm(np.array(nostril_left) - np.array(nostril_right))
                feature['nostril_width'] = nostril_width

                # 特徴をリストに追加
                features.append(feature)

                # 顔の特徴を点で描画し、数値を表示
                for i, landmark in enumerate(landmarks):
                    x, y = int(landmark[0] * image.shape[1]), int(landmark[1] * image.shape[0])
                    cv2.circle(image, (x, y), 1, (0, 255, 0), 1)
                
                # 特徴点を線でつなぐ
                # 目の間の距離（色を変える）
                eye_color = (255, 0, 0)
                cv2.line(image, (int(left_eye_inner[0] * image.shape[1]), int(left_eye_inner[1] * image.shape[0])),
                         (int(right_eye_inner[0] * image.shape[1]), int(right_eye_inner[1] * image.shape[0])), eye_color, 2)

                # 目の横幅（左目）
                left_eye_color = (0, 255, 0)
                cv2.line(image, (int(landmarks[362][0] * image.shape[1]), int(landmarks[362][1] * image.shape[0])),
                         (int(landmarks[263][0] * image.shape[1]), int(landmarks[263][1] * image.shape[0])), left_eye_color, 2)

                # 目の縦幅（左目）
                cv2.line(image, (int(landmarks[386][0] * image.shape[1]), int(landmarks[386][1] * image.shape[0])),
                         (int(landmarks[374][0] * image.shape[1]), int(landmarks[374][1] * image.shape[0])), left_eye_color, 2)

                # 目の横幅（右目）
                right_eye_color = (0, 0, 255)
                cv2.line(image, (int(landmarks[133][0] * image.shape[1]), int(landmarks[133][1] * image.shape[0])),
                         (int(landmarks[33][0] * image.shape[1]), int(landmarks[33][1] * image.shape[0])), right_eye_color, 2)

                # 目の縦幅（右目）
                cv2.line(image, (int(landmarks[159][0] * image.shape[1]), int(landmarks[159][1] * image.shape[0])),
                         (int(landmarks[145][0] * image.shape[1]), int(landmarks[145][1] * image.shape[0])), right_eye_color, 2)

                # 口角から口角までの距離
                mouth_color = (255, 0, 255)
                cv2.line(image, (int(left_corner[0] * image.shape[1]), int(left_corner[1] * image.shape[0])),
                         (int(right_corner[0] * image.shape[1]), int(right_corner[1] * image.shape[0])), mouth_color, 2)

                # 唇の厚み
                lip_color = (0, 255, 255)
                cv2.line(image, 
                        (int(upper_lip[0] * image.shape[1]), int(upper_lip[1] * image.shape[0])),
                        (int(lower_lip[0] * image.shape[1]), int(lower_lip[1] * image.shape[0])), lip_color, 2)

                # 小鼻の横幅
                nostril_color = (255, 255, 0)
                cv2.line(image, (int(nostril_left[0] * image.shape[1]), int(nostril_left[1] * image.shape[0])),
                         (int(nostril_right[0] * image.shape[1]), int(nostril_right[1] * image.shape[0])), nostril_color, 2)

                # 顎から下唇までの直線
                chin_color = (0, 0, 0)
                cv2.line(image, (int(chin[0] * image.shape[1]), int(chin[1] * image.shape[0])),
                         (int(bottom_lip[0] * image.shape[1]), int(bottom_lip[1] * image.shape[0])), chin_color, 2)
            
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
