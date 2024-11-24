# import seaborn as sns
import pandas as pd
import numpy as np
# from scipy.spatial.distance import cdist
from tslearn.metrics import dtw
from service_functions import get_max_energy_columns, normalize_keypoints, get_coordinates
from service_functions import KEY_POINTS, BODY

################# Модуль вычисления близости между рядами #####################

def cyclic_shift(df, shift):
    return df.apply(lambda x: np.roll(x, shift))

# # Функция для калибровки и визуализации вычисления расстояний между временными рядами
# def calibrate_comparison(df1, df2):
#     # Создаем копии исходных данных, чтобы избежать изменений в оригинальных DataFrame
#     df1 = df1[KEY_POINTS].copy().reset_index(drop=True)
#     df2 = df2[KEY_POINTS].copy().reset_index(drop=True)

#     # Нормализуем
#     df1, df2 = normalize_keypoints(df1), normalize_keypoints(df2)

#     # Выбираем 3 самых энергичных ряда
#     num_reg_optim_cols = 3
#     max_energy_cols = get_max_energy_columns(df1, num_reg_optim_cols)
#     print('Самые энергичные ряды:', max_energy_cols)

#     # Сдвигаем df2 относительно df1
#     df1, df2 = shift_sample(df1, df2, max_energy_cols)

#     # Получаем список всех ключевых точек для сравнения
#     essential_points = df1.columns if 'ESSENTIAL_POINTS' not in globals() else ESSENTIAL_POINTS

#     # Инициализируем матрицы для расстояний DTW и корреляций
#     dtw_matrix = pd.DataFrame(index=essential_points, columns=essential_points, dtype=float)
#     corr_matrix = pd.DataFrame(index=essential_points, columns=essential_points, dtype=float)

#     # Заполняем матрицы значениями DTW и корреляции
#     for point1 in essential_points:
#         for point2 in essential_points:
#             dtw_distance = dtw(df1[point1], df2[point2])
#             correlation = df1[point1].corr(df2[point2])
#             dtw_matrix.loc[point1, point2] = dtw_distance
#             corr_matrix.loc[point1, point2] = correlation

#     # Нормируем значения в матрицах для визуализации
#     max_dtw = dtw_matrix.max().max()
#     dtw_matrix_normalized = dtw_matrix / max_dtw
#     corr_matrix_normalized = (1 - corr_matrix) / 2  # Нормируем значения корреляции в диапазон от 0 до 1

#     # Визуализация матриц DTW и корреляции
#     fig, axes = plt.subplots(1, 2, figsize=(20, 10))

#     # Матрица DTW
#     sns.heatmap(dtw_matrix_normalized, annot=dtw_matrix, cmap="RdYlGn_r", ax=axes[0], cbar=True, fmt='.2f')
#     axes[0].set_title("Матрица расстояний DTW (нормированная)")
#     axes[0].set_xticklabels(dtw_matrix.columns, rotation=45, ha='right')
#     axes[0].set_yticklabels(dtw_matrix.index, rotation=0)

#     # Матрица корреляций
#     sns.heatmap(corr_matrix, annot=corr_matrix, cmap="RdYlGn", ax=axes[1], cbar=True, fmt='.2f')
#     axes[1].set_title("Матрица корреляций (нормированная)")
#     axes[1].set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
#     axes[1].set_yticklabels(corr_matrix.index, rotation=0)

#     plt.tight_layout()
#     plt.show()

def compare_samples(df1, df2):

    """
    ВНИМАНИЕ!!!
    df1 и df2 должны бытьв одних и тех же временных единицах, т.е. азы должно быть 30
    Если у df2 fps меньше нужно либо соответственно ужать df1 либо аппроксимировать df2
    """

    # Создаем копии исходных данных, чтобы избежать изменений в оригинальных DataFrame
    df1 = df1[KEY_POINTS].copy().reset_index(drop=True)
    df2 = df2[KEY_POINTS].copy().reset_index(drop=True)
    num_reg_optim_cols = 5

    # Нормализуем
    df1, df2 = normalize_keypoints(df1), normalize_keypoints(df2)

    # Выбираем 3 самых энергичных ряда
    max_energy_cols = get_max_energy_columns(df1, num_reg_optim_cols)
    print('Самые энергичные ряды:', max_energy_cols)

    # Сдвигаем df относительно друг друга, чтобы по максимуму совместить пики
    df1, df2 = shift_sample(df1, df2, max_energy_cols)

    print('После сдвига:')

    # Посчитаем близость сэмплов с учетом весов рядов для каждой части тела
    eval_poze = {}
    for body_part in BODY:
        body_patr_points=get_coordinates(body_part)

        print(f'Сравнение для части тела: {body_part}, точки {body_patr_points}')

        eval_poze[body_part] = DTW_compare(df1, df2, body_patr_points)
        print(f'Результат сравнения для {body_part}: {eval_poze[body_part]}')

        # # Визуализация сравнения DTW для выбранной части тела
        # plt.figure(figsize=(10, 4))
        # plt.plot(df1.index, df1[get_coordinates(body_part)], label=f'df1 - {body_part}')
        # plt.plot(df2.index, df2[get_coordinates(body_part)], linestyle='--', label=f'df2 - {body_part}')
        # plt.title(f'Сравнение DTW для {body_part}')
        # plt.legend()
        # plt.show()
        # print(f'Результат сравнения для {body_part}: {eval_poze[body_part]}')

    eval_poze['GENERAL'] = sum(eval_poze.values()) / len(eval_poze)
    print('Общий результат сравнения:', eval_poze['GENERAL'])

    return eval_poze

def shift_sample(df1, df2, columns):
    max_corr = -np.inf  # Начальное значение для максимальной корреляции
    best_shift = 0      # Начальное значение для лучшего сдвига

    # Ограничиваем диапазон возможного сдвига до ±5% длины ряда
    max_shift = int(0.05 * len(df1))

    # Перебираем возможные сдвиги в диапазоне от -max_shift до +max_shift
    for shift in range(-max_shift, max_shift + 1):
        if shift >= 0:
            # Сдвигаем df1 вправо, обрезаем df1 и df2
            truncated_df1 = df1[columns].iloc[shift:].reset_index(drop=True)
            truncated_df2 = df2[columns].iloc[:len(truncated_df1)].reset_index(drop=True)
        else:
            # Сдвигаем df2 влево, обрезаем df1 и df2
            truncated_df1 = df1[columns].iloc[:shift].reset_index(drop=True)
            truncated_df2 = df2[columns].iloc[-shift:].reset_index(drop=True)

        # Вычисляем корреляцию для группы столбцов
        combined_correlation = 0
        for column in columns:
            if column in df1.columns and column in df2.columns:
                correlation = truncated_df1[column].corr(truncated_df2[column])
                if not np.isnan(correlation):
                    combined_correlation += correlation

        # Проверяем, является ли текущая корреляция максимальной
        if combined_correlation > max_corr:
            max_corr = combined_correlation
            best_shift = shift

    # Печать результата лучшего сдвига и корреляции
    print(f'Лучший сдвиг: {best_shift}')
    print(f'Максимальная корреляция: {max_corr}')

    # Сдвигаем df1 и df2 на оптимальное количество шагов
    if best_shift >= 0:
        df1_shifted = df1.iloc[best_shift:].reset_index(drop=True)
        df2_shifted = df2.iloc[:len(df1_shifted)].reset_index(drop=True)
    else:
        df1_shifted = df1.iloc[:best_shift].reset_index(drop=True)
        df2_shifted = df2.iloc[-best_shift:].reset_index(drop=True)

    return df1_shifted, df2_shifted

# Функция для подсчета расстояний между одноименными столбцами в однородных df
# В качестве columns ожидаем список точек определенной части тела
def DTW_compare(df1, df2, compare_columns=[]):
    print('Запуск DTW_compare...')
    print(f'DTW_compare: Размеры df1: {df1.shape}, Размеры df2: {df2.shape}')

    # Используем по 4 точки для анализа каждой части тела
    num_compare_cols = 4

    if len(compare_columns) == 0:
        compare_columns = df1.columns.to_list()

    print('DTW_compare: columns для сравнения', compare_columns)

    # Уменьшаем список рядов для сравнения до num_compare_cols, выбирая самые энергичные
    compare_columns = get_max_energy_columns(df1, num_compare_cols, compare_columns)

    # num_plots = num_compare_cols
    # fig, axes = plt.subplots(1, num_plots, figsize=(4 * num_plots, 4))

    distances = []

    for i, col in enumerate(compare_columns):
        print(f'Сравнение столбца: {col}')

        # Рассчитываем расстояние методом DTW
        distance = dtw(df1[col], df2[col])
        print(f'DTW расстояние для столбца {col}: {distance}')
        distances.append(distance)

    #     # Построение графика
    #     ax = axes[i]  # Исправлено
    #     ax.plot(df1[col], label='df1 ' + col)
    #     ax.plot(df2[col], label='df2 ' + col)
    #     ax.set_title(f'График {col}')
    #     ax.set_xlabel(f'DTW {distances[-1]:.2f}')
    #     ax.legend()

    # # Настраиваем фигуру и выводим ее
    # plt.subplots_adjust(wspace=0.5)
    # plt.tight_layout()
    # plt.show()


    return np.array(distances).mean()

# def visualize_column_pairs(df1, df2, columns):
#     """
#     Визуализирует пары одноименных столбцов из df1 и df2 в одной строке.

#     Parameters:
#     df1 : DataFrame
#         Первый DataFrame для сравнения.
#     df2 : DataFrame
#         Второй DataFrame для сравнения.
#     columns : list
#         Список одноименных столбцов для визуализации.
#     """
#     num_cols = len(columns)
#     fig, axs = plt.subplots(1, num_cols, figsize=(5 * num_cols, 5))

#     for i, column in enumerate(columns):
#         ax = axs[i] if num_cols > 1 else axs
#         if column in df1.columns and column in df2.columns:
#             ax.plot(df1.index, df1[column], label=f'{column} - df1', alpha=0.7)
#             ax.plot(df2.index, df2[column], label=f'{column} - df2', alpha=0.7)
#             ax.set_title(f'Comparison: {column}')
#             ax.set_xlabel('Index')
#             ax.set_ylabel('Value')
#             ax.legend()
#             ax.grid(True)

#     plt.tight_layout()
#     plt.show()

