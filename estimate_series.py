# На входе получим ссылку на папку с ресурсами по зарядке и ссылку на файл с оцифровкой серии
# и параметр = номер строки с таймингом для обработки файла
# и папку, куда класть результат
# в эту папку сложим json-ы с оценками упражнений соотв номеру строки талицы с таймингом

import pandas as pd
import os
import numpy as np
from count_drills import match_sample_in_series
from service_functions import time_to_seconds,create_filename_ext, KEY_POINTS
from compare_samples import compare_samples

def get_rate_by_score(val):
    if val <=5:
        rate=5
    elif val < 10:
        rate=4
    elif val < 15:
        rate=3
    else:
        rate=2

    return rate

def append_dict_to_csv_pandas(result_path, data_dict, result_csv='result.csv'):
    
    file_path = os.path.join(result_path, result_csv)
    
    # Проверяем, существует ли файл
    if os.path.isfile(file_path):
        # Читаем существующий CSV
        df = pd.read_csv(file_path)
        # Создаем DataFrame из словаря
        new_row = pd.DataFrame([data_dict])
        # Добавляем новую строку
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        # Если файла нет, создаем новый DataFrame
        df = pd.DataFrame([data_dict])
    
    # Сохраняем обратно в CSV
    df.to_csv(file_path, index=False, encoding='utf-8')
    print("Данные успешно добавлены в CSV-файл с использованием pandas.")
    
    return len(df)

annotation_file_name='df_keypoints_annotation.csv'
feather_file_name='df_keypoints.feather'
pupil_fragments_path='pupil_fragments'

def estimate_series(workout_path='instructor_workout_source', series_file_path='pupil_fragments/1.csv',result_path='result_estim'):

    path_to_feather = os.path.join(workout_path, feather_file_name)
    df_keypoints = pd.read_feather(path_to_feather)[KEY_POINTS]

    print('Начало df_keypoints', df_keypoints.head(5))
    print('Конец df_keypoints', df_keypoints.tail(5))

    annotation_file_path = os.path.join(workout_path,annotation_file_name)
    df_annotation=pd.read_csv(annotation_file_path, index_col=0)
    print(df_annotation.head())

    base_name = os.path.splitext(os.path.basename(series_file_path))[0]
    print(base_name)
    try:
        # Пытаемся преобразовать строку в целое число
        drill_index = int(base_name)
    except ValueError:
        # Если преобразование не удалось, выводим ошибку и возвращаем None
        print(f"Ошибка: '{base_name}' не является номером фрагмента")
        return None
    if drill_index >= len(df_annotation):
        print(f"Ошибка: 'Номер фрагмента {drill_index}' не соответствует разметке")
        return None
    
    df_drill_pupil = pd.read_csv(series_file_path,index_col=0)
    row = df_annotation.iloc[drill_index]
    
    print('Работаетм с фрагментом\n',pd.DataFrame([row]))

    fragment = row['Фрагмент']
    activity = row['Активность']

    drill_end = time_to_seconds(row['t_hand_end'])
    drill_start = time_to_seconds(row['t_hand_start'])
    drill_time=drill_end-drill_start
 
    print(f"Обрабатываем фрагмент: {fragment}: Начало: {drill_start}, Конец: {drill_end}, Активность: {activity}")
    if activity == 'Демонстрация':
        print(f'Фрагмент {drill_index}, имеет некорректную метку {activity}')
        return None

    sample_end = time_to_seconds(row['t_sample_end'])
    sample_start = time_to_seconds(row['t_sample_start'])
    sample_time=sample_end-sample_start
    print(f'Длительность серии {fragment} - {drill_time}, длительность сэмпла {sample_time} Количество движений {int(drill_time/sample_time)})')

    df_drill_instructor = df_keypoints.loc[drill_start:drill_end]
    print ('drill_start:drill_end len(df_drill)',drill_start,drill_end,len(df_drill_instructor))
    
    df_drill_instructor_sample = df_keypoints.loc[sample_start:sample_end]
    print('sample_start:sample_end len(df_drill_sample)',sample_start,sample_end, len(df_drill_instructor_sample))

    max_energy_columns = row['MAX_ENERGY_COLS'].split(', ')

    num_drills_instructor = row['N Повторений']

    print( "Анализируем ученика")
    num_drills_pupil = match_sample_in_series(df_drill_pupil, df_drill_instructor_sample, max_energy_columns)

    print(f'Детектировано сэмплов в видео Инструктора/ученика {num_drills_instructor}/{num_drills_pupil}, отличие {abs(num_drills_instructor-num_drills_pupil)}')

    eval_pose = compare_samples(df_drill_instructor,df_drill_pupil)
    eval_pose['Упражнение'] = fragment
    eval_pose['Id'] = drill_index

    # Считаем оценку для количества упражнений
    if eval_pose['GENERAL'] < 5:
        drill_num_rate = 5
    elif num_drills_pupil >= num_drills_instructor-1:
        drill_num_rate = 5
    elif  num_drills_pupil/num_drills_instructor >= 0.75:
        drill_num_rate = 5
    elif  num_drills_pupil/num_drills_instructor >= 0.5:
        drill_num_rate = 4
    elif  num_drills_pupil/num_drills_instructor >= 0.25:
        drill_num_rate = 3
    else:
        drill_num_rate = 2

    eval_pose['Оценка за количество'] = drill_num_rate

    drill_count = append_dict_to_csv_pandas(result_path, eval_pose)

    return drill_count


# for filename in os.listdir(pupil_fragments_path):
#     filename_path =os.path.join(pupil_fragments_path, filename)

#     print(estimate_series(series_file_path=filename_path))

def highlight_cells(val):
    if val == 5:
        color = 'background-color: green'
    elif val == 4:
        color = 'background-color: yellow'
    elif val == 3:
        color = 'background-color: orange'
    else:
        color = 'background-color: red'

    return color


result_file='result.csv'
mp3_annotation_file='mp3_annotation.json'

def total_estim(result_path='result_estim', mp3_annotation_path='instructor_workout_source'):
    
    df_mp3_annotation=pd.read_json(os.path.join(mp3_annotation_path,mp3_annotation_file))
    drill_num=len(df_mp3_annotation)
    print ('df_mp3_annotation', df_mp3_annotation)

    general_eval_pose = pd.read_csv(os.path.join(result_path,result_file))
    general_eval_pose.set_index('Упражнение', inplace=True)
    general_eval_pose = general_eval_pose.sort_values(by='Id')

    # Фильтрация строк, где значение в колонке 'General' больше 2 (<15 в сырых значениях DTW)
    valid_drill_num = len(general_eval_pose[general_eval_pose['GENERAL'] < 15])

    if valid_drill_num < drill_num:
        # Если общая оценка хотя бы по одному упражнению - не зачет - то за всю зарядку - незачет
        return 2
    else:
        # Добавление итоговой строки со средними значениями
        summary = general_eval_pose.mean().to_frame().T
        summary.index = ['ИТОГ']
        general_eval_pose = pd.concat([general_eval_pose, summary])

        print(general_eval_pose)

        subset=['HEAD','TORSO','ARMS','LEGS','GENERAL']

        general_eval_pose[subset] = general_eval_pose[subset].applymap(get_rate_by_score)

        print(general_eval_pose)

        result_json_file=create_filename_ext(result_path, result_file)

        general_eval_pose.to_json(result_json_file)

        return general_eval_pose.loc['ИТОГ']['GENERAL']
    
total_score = total_estim()

print('Общая оценка за зарядку', total_score)




