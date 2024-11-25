import argparse
import pandas as pd
import os
import json
from service_functions import create_filename_ext

result_filename='result.csv'
total_filename='total_result.csv'
total_score_filename='total_result.csv'
json_filename = 'result.json'
mp3_annotation_file='mp3_annotation.json'

def create_path(path,filename):
    return os.path.join(path,filename)

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

def total_score(result_path='result_estim', workout_path='instructor_workout_source'):
    
    df_mp3_annotation=pd.read_json(os.path.join(workout_path,mp3_annotation_file))
    drill_num=len(df_mp3_annotation)
    print ('df_mp3_annotation', df_mp3_annotation)

    general_eval_pose = pd.read_csv(os.path.join(result_path,result_filename))
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

        general_eval_pose.to_csv(os.path.join(result_path, total_filename))
        
        # Конвертация DataFrame в словарь с ключами из индекса
        general_eval_dict = general_eval_pose.to_dict(orient='index')

        # Добавление общей оценки
        general_eval_dict["Общая оценка"] = int(general_eval_pose.loc['ИТОГ']['GENERAL'])
        print(general_eval_dict)

        result_json_path=os.path.join(result_path, json_filename)
       
        # Запись словаря в JSON файл с правильной кодировкой и форматированием
        with open(result_json_path, 'w', encoding='utf-8') as f:
            json.dump(general_eval_dict, f, ensure_ascii=False, indent=4)
        
        return general_eval_pose.loc['ИТОГ']['GENERAL']
    
print('Общая оценка за зарядку', total_score())

def main():
    # Создание парсера аргументов
    parser = argparse.ArgumentParser(description="Обработчик видео для общей оценки зарядки")
    parser.add_argument('--result_path', type=str, required=True, help='Путь к папке результата')
    parser.add_argument('--workout_path', type=str, required=True, help='Путь к паке с ресурсами зарядки')

    # Парсинг аргументов
    args = parser.parse_args()

    # Вывод аргументов для проверки
    print(f"Путь к видео: {args.result_path}")
    print(f"Путь к HTML файлу: {args.workout_path}")

    # Вызов основной функции
    print(total_score(args.result_path, args.workout_path))

if __name__ == "__main__":
    main()