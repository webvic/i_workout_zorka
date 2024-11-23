# Рутовый модуль workout()
import os
import pandas as pd
import argparse

from compare_samples import eval_pupi_df
from mediapipe_video_processing import get_key_points_from_video
from service_functions import KEY_POINTS, filter_rows_by_y_range, normalize_keypoints
from split_series_to_samples import get_serie_breaking_points
from split_workout_to_series import classify_segments, get_change_points
from workout_stat import generate_final_stats

def workout(path_to_video, html_output_path, path_to_instructor_samples):
  # Извлекаем ключевые точки с пом Mediapipe
  df_keypoints, workout_time = get_key_points_from_video(path_to_video)
  # display('workout: df_keypoints',df_keypoints)

  # отбираем только важные 17 точек из исходных 33
  df_keypoints=df_keypoints[KEY_POINTS]

  # фильтрационная маска, для отсечке выхода за кадр
  mask = filter_rows_by_y_range(df_keypoints,-0.1,1.1)

  # Фильтруем df_keypoints от выходящих за рамки экрана значений
  df_keypoints = df_keypoints[mask].reset_index(drop=True)

  # ищем точки смены серий упражнений по спектральным характеристикам сигнала
  change_points,max_energy_cols = get_change_points(df_keypoints)
  # print('workout: len(df_keypoints),change_points',len(df_keypoints),change_points)

  # размечаем релевантные сегменты как "сильные"
  is_strong_segment, metrics_df = classify_segments(df_keypoints, change_points)
  # print('workout: is_strong_segment, metrics_df',is_strong_segment, metrics_df)

  result_workout=[]
  # Закачиваем базу движений инструктора
  df_instructor=pd.read_csv(path_to_instructor_samples,index_col=0)
  path_to_instructor_dir=os.path.dirname(path_to_instructor_samples)

  # Цикл по сериям внутри видео одной зарядки
  for i in range(len(change_points) - 1):
      if not is_strong_segment[i]:
          continue

      start, end = change_points[i], change_points[i+1]
      df_segment = normalize_keypoints(df_keypoints[KEY_POINTS].iloc[start:end])

      # print('start:end',start,end)

      # Разделяем сегмент на сэмплы с помощью н/ч гармоник из преобразования Фурье
      breaking_points=get_serie_breaking_points(df_segment)
      # print('workout: breaking_points ',breaking_points)

      result_workout.append(eval_pupi_df(df_segment, df_instructor, breaking_points,path_to_instructor_dir))
      ############## st.display(graf_image)

  # собираем сырую статистику по всем сэмплам тренировки 
  result_workout_df= pd.concat(result_workout)   
  
  print('workout: result_workout_df',result_workout_df)

  # Генерируем агрегированную статистику и отправляем ее по адресу назначения
  result_stat = generate_final_stats(html_output_path,result_workout_df,round(workout_time/60,1))
  
  return result_stat
  
# path_to_instructor_samples='/project/py-video-processor/scripts/instructor_samples/instructor_0_samples_0.csv'
# path_to_video = '/project/py-video-processor/scripts/Video/first-vid.mp4'

# workout(path_to_video,path_to_instructor_samples)

def main():
    # Создание парсера аргументов
    parser = argparse.ArgumentParser(description="Обработчик видео для анализа упражнений")
    parser.add_argument('--video-path', type=str, required=True, help='Путь к видеофайлу')
    parser.add_argument('--html-output-path', type=str, required=True, help='Путь к HTML файлу для вывода результатов')
    parser.add_argument('--path-to-instructor-samples', type=str, required=True, help='Путь к образцам инструктора')

    # Парсинг аргументов
    args = parser.parse_args()

    # Вывод аргументов для проверки
    print(f"Путь к видео: {args.video_path}")
    print(f"Путь к HTML файлу: {args.html_output_path}")
    print(f"Путь к образцам инструктора: {args.path_to_instructor_samples}")

    # Вызов основной функции
    print(workout(args.video_path, args.html_output_path, args.path_to_instructor_samples))

if __name__ == "__main__":
    main()