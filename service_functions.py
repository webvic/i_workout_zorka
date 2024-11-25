
############# Служебные функции для анализа keypoints_df #################
import os
import pandas as pd
import numpy as np
from scipy.stats import median_abs_deviation

# Cтруктура данных Ключевые точки тела
BODY = {
    "HEAD":{
    "NOSE": {'x': "NOSE_x", 'y': "NOSE_y"},
    "LEFT_EAR": {'x': "LEFT_EAR_x", 'y': "LEFT_EAR_y"},
    "RIGHT_EAR": {'x': "RIGHT_EAR_x", 'y': "RIGHT_EAR_y"}
    },
    "TORSO":{
    "LEFT_SHOULDER": {'x': "LEFT_SHOULDER_x", 'y': "LEFT_SHOULDER_y"},
    "RIGHT_SHOULDER": {'x': "RIGHT_SHOULDER_x", 'y': "RIGHT_SHOULDER_y"},
    "LEFT_HIP": {'x': "LEFT_HIP_x", 'y': "LEFT_HIP_y"},
    "RIGHT_HIP": {'x': "RIGHT_HIP_x", 'y': "RIGHT_HIP_y"}
    },
    "ARMS":{
    "LEFT_ELBOW": {'x': "LEFT_ELBOW_x", 'y': "LEFT_ELBOW_y"},
    "RIGHT_ELBOW": {'x': "RIGHT_ELBOW_x", 'y': "RIGHT_ELBOW_y"},
    "LEFT_WRIST": {'x': "LEFT_WRIST_x", 'y': "LEFT_WRIST_y"},
    "RIGHT_WRIST": {'x': "RIGHT_WRIST_x", 'y': "RIGHT_WRIST_y"}
    },
    "LEGS":{
    "LEFT_KNEE": {'x': "LEFT_KNEE_x", 'y': "LEFT_KNEE_y"},
    "RIGHT_KNEE": {'x': "RIGHT_KNEE_x", 'y': "RIGHT_KNEE_y"},
    "LEFT_HEEL": {'x': "LEFT_HEEL_x", 'y': "LEFT_HEEL_y"},
    "RIGHT_HEEL": {'x': "RIGHT_HEEL_x", 'y': "RIGHT_HEEL_y"}
    }
}

KEY_POINTS = [coordinate for body_part in BODY.values() for body_part_point in body_part.values() for coordinate in body_part_point.values()]

# # Вариант с заполнением концов крайними значениями средней части
# def adaptive_moving_average(series, window_size=30):
#     """
#     Сглаживание временного ряда с использованием скользящего среднего для центральной части и фиксированных значений для концов.

#     Args:
#     series (pd.Series): Временной ряд.
#     window_size (int): Размер окна для сглаживания.

#     Returns:
#     pd.Series: Сглаженный временной ряд с равной длиной.
#     """
#     # Применение rolling для центральных значений
#     rolling_avg = series.rolling(window=window_size, center=True).mean()

#     # Получение крайних значений средней части
#     left_value = rolling_avg.iloc[window_size // 2]
#     right_value = rolling_avg.iloc[-window_size // 2 - 1]

#     # Заполнение концов фиксированными значениями
#     rolling_avg.iloc[:window_size // 2] = left_value
#     rolling_avg.iloc[-window_size // 2:] = right_value

#     return rolling_avg

# Вариант с постепенным снижением размера окна усреднения до единиц на концах
def adaptive_moving_average(series, window_size=30):
    """
    Комбинированное скользящее среднее с использованием rolling и expanding для обработки краев.

    Args:
    series (pd.Series): Временной ряд.
    window_size (int): Размер окна для сглаживания.

    Returns:
    pd.Series: Сглаженный временной ряд с равной длиной.
    """
    # Применение rolling для центральных значений
    rolling_avg = series.rolling(window=window_size, center=True).mean()

    # Обработка краев с помощью expanding
    expanding_start = series.iloc[:window_size // 2 + 1].expanding().mean()
    expanding_end = series.iloc[-window_size // 2:].iloc[::-1].expanding().mean().iloc[::-1]

    # Объединение всех частей
    combined_avg = pd.concat([expanding_start, rolling_avg.iloc[window_size // 2 + 1: -window_size // 2], expanding_end])
    combined_avg = combined_avg.reindex(series.index)  # Приведение длины к исходному ряду
    return combined_avg

def filter_rows_by_y_range(df, min_value=-0.2, max_value=1.2):
    """
    Фильтрует строки DataFrame, удаляя те, в которых хотя бы одно значение в столбцах,
    содержащих '_x', выходит за пределы заданного диапазона.

    Parameters:
    df : DataFrame
        Исходный DataFrame для фильтрации.
    min_value : float
        Минимальное допустимое значение.
    max_value : float
        Максимальное допустимое значение.

    Returns:
    DataFrame
        Отфильтрованный DataFrame.
    """
    # Выбираем только столбцы, которые содержат '_y'
    x_columns = [col for col in df.columns if '_y' in col]

    # Создаем маску для строк, где все значения в этих столбцах находятся в диапазоне
    mask = (df[x_columns] >= min_value).all(axis=1) & (df[x_columns] <= max_value).all(axis=1)

    # if has_internal_false(mask):
    #     return None

    # Возвращаем отфильтрованный df
    return mask

def get_coordinates(body_part, side="BOTH"):
    if side in ["LEFT", "RIGHT"]:
        filtered_body_part = {key: value for key, value in BODY[body_part].items() if side in key}
        return [coordinate for body_part_point in filtered_body_part.values() for coordinate in body_part_point.values()]
    else:
        return [coordinate for body_part_point in BODY[body_part].values() for coordinate in body_part_point.values()]

LEGS=get_coordinates("LEGS")
ARMS=get_coordinates("ARMS")
RIGHT_ARM = get_coordinates("ARMS","RIGHT")
HEAD = get_coordinates("HEAD")
TORSO = get_coordinates("TORSO")

# display(LEGS,'\n',ARMS,'\n',RIGHT_ARM,'\n',HEAD,'\n',TORSO)

# Найти в df num_col колонoк с максимальной амплитудой
def get_max_amplitude_col(df,num_col=1):

  num_columns=len(df.columns)
  # гарантируем, что не превысим индекс
  if num_col>num_columns: num_col=num_columns

  # Вычислить амплитуду (разницу между макс. и мин. значениями) для каждой колонки
  amplitudes = df.apply(lambda x: np.max(x) - np.min(x))

  # Отсортировать колонки по убыванию амплитуды и вернуть num_col названий столбцов с самыми блольшими амплитудами
  return amplitudes.sort_values(ascending=False).head(num_col).index

def get_max_energy_columns(df,num_max_energy_col, columns=[]):
    """
    Возвращает num_max_energy_col имен столбцов из списка `df.columns` с максимальной дисперсией относительно медианного значения,
    что защищает от влияния случайных выбросов
    """
    if len(columns) == 0:
        columns=df.columns

    if  num_max_energy_col >  len(columns):
        num_max_energy_col =  len(columns)

    top_cols = df[columns].apply(lambda col: median_abs_deviation(col, scale='normal')).sort_values(ascending=False).index[:num_max_energy_col]
    return top_cols


# выбрать из списка колонок колонки по определенной оси (x,y,z)
def get_ax_columns(columns,ax):
  return [col for col in columns if '_'+ax in col]

# нормализовать данные в колонке df
def normalize_time_series(time_series):
    min = np.min(time_series)
    max = np.max(time_series)
    if max==min:
      return None
    return (time_series - min) / (max - min)

# Нормируем данные df_keypoints по глобальному мин/макс для унификации размеров тела
def normalize_keypoints(df_keypoints):
  # display(df_keypoints)

  #Удаляем тренды
  # df_keypoints=df_keypoints.apply(lambda x: x - np.polyval(np.polyfit(df_keypoints.index, x, 1), df_keypoints.index))

  # Находим минимальные и максимальные значения координат x и y
  x_columns = get_ax_columns(df_keypoints.columns,'x')
  y_columns = get_ax_columns(df_keypoints.columns,'y')
  # display(x_columns)
  # display(y_columns)

  min_x = df_keypoints[x_columns].min().min()
  max_x = df_keypoints[x_columns].max().max()
  min_y = df_keypoints[y_columns].min().min()
  max_y = df_keypoints[y_columns].max().max()

  # print('min_x, min_y, max_x, max_y', min_x, min_y, max_x, max_y)

  # Вычисляем ширину и высоту прямоугольника
  width = max_x - min_x
  height = max_y - min_y
  # print('width,height', width,height)

  # Нормируем координаты относительно прямоугольника

  df_normalized = pd.concat([
      df_keypoints[x_columns].apply(lambda x: (x - min_x) / width),
      df_keypoints[y_columns].apply(lambda y: (y - min_y) / height)
      ],axis=1)

  # display(df_normalized)

  return df_normalized

def time_to_seconds(time_str):
    # Разделяем минуты и секунды
    try:
        if ':' in time_str:
            minutes, seconds = time_str.split(':')
            total_seconds = int(minutes) * 60 + float(seconds)
        else:
            total_seconds = float(time_str)
    except ValueError:
        print(f"Неправильный формат времени: {time_str}")
        return 0

    return total_seconds

def get_output_filename(input_path,suffix=None,ext=None):
    # Разделяем путь на директорию, имя файла и расширение
    directory, filename = os.path.split(input_path)
    name, extension = os.path.splitext(filename)

    if ext:
        extension=ext

    # Создаем новый путь с добавленным суффиксом
    new_filename = f"{name}{suffix}{extension}"
    output_path = os.path.join(directory, new_filename)

    return output_path