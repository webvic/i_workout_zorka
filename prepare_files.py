# Подготовим рабочие файлы упражнений из оцифровки инструктора и файла тайминга
# Нарезаем оцифровку тренировки инструктора на фрагменты для сравнения
# Готовим json для таймингов и id фрагментов для фронта

import pandas as pd
import os
import numpy as np
from count_drills import match_sample_in_series
from service_functions import time_to_seconds, KEY_POINTS

path_to_source='instructor_workout_source'
# df_keypoints_file_name='df_keypoints.csv'
df_annotation_file_name='Тайминг упражнений в зарядке Андрея.csv'

# path_to_instructor_keypoints = os.path.join(path_to_source,df_keypoints_file_name)
# df_keypoints=pd.read_csv(path_to_instructor_keypoints, index_col=0)
# print(df_keypoints.head())

# # Создание временных меток
# df_keypoints['timestamp'] = df_keypoints.index / 30
# # Установка временных меток в качестве индекса
# df_keypoints.set_index('timestamp', inplace=True)
# # Сохраняем дополненный метками образец инструктора
# df_keypoints.to_csv(path_to_instructor_keypoints)

feather_file_name='df_keypoints.feather'

path_to_feather = os.path.join(path_to_source,feather_file_name)

df_keypoints = pd.read_feather(path_to_feather)[KEY_POINTS]

# # Шаг времени: 1/30 секунды
# step = 1/30  # ≈0.033333 секунд

# # Создание временного индекса в секундах с дробной частью
# time_index = np.arange(len(df_keypoints)) * step  # [0.0, 0.033333, 0.066666, ..., ...]

# # Назначение временных меток как индекса
# df_keypoints.index = time_index

# df_keypoints.to_feather(path_to_feather)

print('Начало df_keypoints', df_keypoints.head(5))
print('Конец df_keypoints', df_keypoints.tail(5))

annotation_file_name = os.path.join(path_to_source,df_annotation_file_name)
df_annotation=pd.read_csv(annotation_file_name, index_col=0)
print(df_annotation.head())

general_eval_pose=[]
for index, row in df_annotation.iterrows():
    print(pd.DataFrame([row]))

    fragment = row['Фрагмент']
    activity = row['Активность']

    drill_end = time_to_seconds(row['t_hand_end'])
    drill_start = time_to_seconds(row['t_hand_start'])
    drill_time=drill_end-drill_start
    drill_index = row['ID фрагмента']

    print(f"Обрабатываем фрагмент: {fragment}: Начало: {drill_start}, Конец: {drill_end}, Активность: {activity}")
    if activity == 'Демонстрация':
      print(f'пропускаем фрагмент {index}, активность {activity}')
      continue

    sample_end = time_to_seconds(row['t_sample_end'])
    sample_start = time_to_seconds(row['t_sample_start'])
    sample_time=sample_end-sample_start
    print(f'Длительность серии {fragment} - {drill_time}, длительность сэмпла {sample_time} Количество движений {int(drill_time/sample_time)})')

    df_drill = df_keypoints.loc[drill_start:drill_end]
    print ('drill_start:drill_end len(df_drill)',drill_start,drill_end,len(df_drill))
    df_drill_sample = df_keypoints.loc[sample_start:sample_end]
    print('sample_start:sample_end len(df_drill_sample)',sample_start,sample_end, len(df_drill_sample))

    max_energy_columns = row['MAX_ENERGY_COLS'].split(', ')

    num_drills_instructor_apr = int(np.round(drill_time/sample_time))
    print('Предв. оценка числа сэмплов',)
    print( "Анализируем инструктора")
    num_drills_instructor = match_sample_in_series(df_drill, df_drill_sample, max_energy_columns)

    df_annotation.loc[index,'N Повторений'] = num_drills_instructor

    print(f'Детектировано сэмплов в видео Инструктора/Априорно {num_drills_instructor}/{num_drills_instructor_apr}')

print(df_annotation)
df_annotation.to_csv(annotation_file_name)

# 1. Объединённая фильтрация строк и выбор столбцов с использованием .loc[]
selected_columns = ['Фрагмент', 'mp3_start', 'mp3_end']
df_mp3_annotation = df_annotation.loc[df_annotation['ID фрагмента'].notna(), selected_columns]

print("\nОтфильтрованный DataFrame (без 'ID фрагмента'):")
print(df_mp3_annotation)

# 2. Сброс индекса и добавление его как колонки 'Index'
df_mp3_annotation = df_mp3_annotation.reset_index().rename(columns={'index': 'Index'})

print("\nОтфильтрованный DataFrame с колонкой 'Index':")
print(df_mp3_annotation)

# 3. Экспорт в JSON
json_file_name = 'mp3_annotation.json'
json_file_path = os.path.join(path_to_source,json_file_name)
df_mp3_annotation.to_json(json_file_path, orient='records', indent=4, force_ascii=False)

print(f"\nDataFrame успешно сохранён в {path_to_source}")