import cv2
import mediapipe as mp
from service_functions import KEY_POINTS
import pandas as pd
import time

# Инициализируем структуру ключевых точек тела
def initialize_key_points():
    keypoints_data = {

      "NOSE_x": [], "NOSE_y": [],
      "LEFT_EYE_INNER_x": [], "LEFT_EYE_INNER_y": [],
      "LEFT_EYE_x": [], "LEFT_EYE_y": [],
      "LEFT_EYE_OUTER_x": [], "LEFT_EYE_OUTER_y": [],
      "RIGHT_EYE_INNER_x": [], "RIGHT_EYE_INNER_y": [],
      "RIGHT_EYE_x": [], "RIGHT_EYE_y": [],
      "RIGHT_EYE_OUTER_x": [], "RIGHT_EYE_OUTER_y": [],
      "LEFT_EAR_x": [], "LEFT_EAR_y": [],
      "RIGHT_EAR_x": [], "RIGHT_EAR_y": [],
      "MOUTH_LEFT_x": [], "MOUTH_LEFT_y": [],
      "MOUTH_RIGHT_x": [], "MOUTH_RIGHT_y": [],
      "LEFT_SHOULDER_x": [], "LEFT_SHOULDER_y": [],
      "RIGHT_SHOULDER_x": [], "RIGHT_SHOULDER_y": [],
      "LEFT_ELBOW_x": [], "LEFT_ELBOW_y": [],
      "RIGHT_ELBOW_x": [], "RIGHT_ELBOW_y": [],
      "LEFT_WRIST_x": [], "LEFT_WRIST_y": [],
      "RIGHT_WRIST_x": [], "RIGHT_WRIST_y": [],
      "LEFT_PINKY_x": [], "LEFT_PINKY_y": [],
      "RIGHT_PINKY_x": [], "RIGHT_PINKY_y": [],
      "LEFT_INDEX_x": [], "LEFT_INDEX_y": [],
      "RIGHT_INDEX_x": [], "RIGHT_INDEX_y": [],
      "LEFT_THUMB_x": [], "LEFT_THUMB_y": [],
      "RIGHT_THUMB_x": [], "RIGHT_THUMB_y": [],
      "LEFT_HIP_x": [], "LEFT_HIP_y": [],
      "RIGHT_HIP_x": [], "RIGHT_HIP_y": [],
      "LEFT_KNEE_x": [], "LEFT_KNEE_y": [],
      "RIGHT_KNEE_x": [], "RIGHT_KNEE_y": [],
      "LEFT_ANKLE_x": [], "LEFT_ANKLE_y": [],
      "RIGHT_ANKLE_x": [], "RIGHT_ANKLE_y": [],
      "LEFT_HEEL_x": [], "LEFT_HEEL_y": [],
      "RIGHT_HEEL_x": [], "RIGHT_HEEL_y": [],
      "LEFT_FOOT_INDEX_x": [], "LEFT_FOOT_INDEX_y": [],
      "RIGHT_FOOT_INDEX_x": [], "RIGHT_FOOT_INDEX_y": []

    }
    return keypoints_data

def get_key_points_from_video(path_to_video ="pupil_fragments/3.mp4" ):
    print('get_key_points_from_video: path_to_video', path_to_video)
    start_time = time.time()

    # Создание объекта для захвата видео
    cap = cv2.VideoCapture(path_to_video)

    # Проверка успешности операции
    if not cap.isOpened():
        print("Ошибка: Невозможно открыть видеофайл.")
        return None, None
    else:
        print(f"Видеофайл {path_to_video} успешно открыт.")

    # Создаем экземпляр модели BlazePose, настроенной на сложный поиск 33 точек одного человека
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=0, smooth_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Инициализируем таблицу временных рядов по ключевым точкам
    keypoints_data = {key: [] for key in KEY_POINTS}
    keypoints_data['timestamp'] = []

    # Получаем параметры видео
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Считываем первый кадр
    ret, frame = cap.read()
    if not ret:
        print("Ошибка: Не удалось считать первый кадр видео.")
        cap.release()
        pose.close()
        return None, None

    landmark_names = list(set('_'.join(landmark.split('_')[:-1]) for landmark in KEY_POINTS))

    while ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            # Сбор данных только для ключевых точек, указанных в KEY_POINTS
            for landmark_name in landmark_names:
                mp_landmark_name = getattr(mp_pose.PoseLandmark, landmark_name)
                keypoints_data[f"{landmark_name}_x"].append(results.pose_landmarks.landmark[mp_landmark_name].x)
                keypoints_data[f"{landmark_name}_y"].append(results.pose_landmarks.landmark[mp_landmark_name].y)

            # Добавляем временную метку
            keypoints_data['timestamp'].append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)

        ret, frame = cap.read()

    # Закрытие объектов и освобождение ресурсов
    pose.close()
    cap.release()

    # Создаем DataFrame из ключевых точек
    df_keypoints = pd.DataFrame(keypoints_data)
    df_keypoints.set_index('timestamp', inplace=True)

    # keypoints_csv = get_output_filename(path_to_video, '_keypoints', '.csv')
    # df_keypoints.to_csv(keypoints_csv)

    # print(f"DataFrame успешно создан и сохранен в {keypoints_csv}.")
    
    end_time = time.time()
    print(f"Время выполнения функции: {end_time - start_time:.2f} секунд")

    return df_keypoints

# Тестовый блок
path_to_video = 'pupil_fragments/3.mp4'

# анализируем видео
df_keypoints = get_key_points_from_video(path_to_video)

print('df_keypoints\n',df_keypoints)