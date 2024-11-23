from scipy.signal import savgol_filter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter

# Нормализация серии от -1 до 1
def series_normalization(series):
    min_val = np.min(series)
    max_val = np.max(series)
    return 2 * (series - min_val) / (max_val - min_val) - 1

# Динамическая нормализация корреляционной кривой по размерам окна
def dynamic_normalization(correlation_curve, window_size=50):
    """
    Нормализация центрированной вокруг нуля корреляционной кривой по амплитуде в окне нормализации.

    Args:
    correlation_curve (pd.Series or np.ndarray): Центрированная вокруг нуля кривая.
    window_size (int): Размер окна для нормализации.

    Returns:
    np.ndarray: Нормализованная кривая в диапазоне от -1 до 1.
    """
    # Вычисляем скользящее значение амплитуды для окна
    rolling_amplitude = pd.Series(correlation_curve).rolling(window=window_size, min_periods=1, center=True).apply(lambda x: max(x.max(), abs(x.min())), raw=True)

    # Нормализация корреляционной кривой по амплитуде
    normalized_curve = correlation_curve / (rolling_amplitude + 1e-8)

    # Нормируем значения в диапазоне от -1 до 1
    return normalized_curve

# Вычисляет адаптированную корреляционную кривую с обработкой краев и динамической нормализацией
def compute_full_correlation(series, sample):
    # Преобразуем серии в массивы
    series = series.values if isinstance(series, pd.Series) else np.array(series)
    sample = sample.values if isinstance(sample, pd.Series) else np.array(sample)

    # Длина серий
    series_len = len(series)
    sample_len = len(sample)
    half_sample_len = sample_len // 2
    correlation_curve = np.zeros(series_len)

    # Начальная фаза: вычисляем корреляцию с половины сэмпла и постепенно увеличиваем
    for i in range(half_sample_len):
        current_sample_len = half_sample_len + i + 1
        current_sample = sample[-current_sample_len:]
        correlation_curve[i] = np.corrcoef(series[:current_sample_len], current_sample)[0, 1]

    # Основная фаза: полная корреляция
    for i in range(series_len - sample_len + 1):
        series_segment = series[i:i + sample_len]
        correlation_curve[half_sample_len + i] = np.corrcoef(series_segment, sample)[0, 1]

    # Конечная фаза: уменьшение сэмпла
    for i in range(half_sample_len):
        current_sample_len = sample_len - i - 1
        current_sample = sample[:current_sample_len]
        correlation_curve[-half_sample_len + i] = np.corrcoef(series[-current_sample_len:], current_sample)[0, 1]

    return correlation_curve

def adaptive_moving_average(series, window_size=30):
    """
    Сглаживание временного ряда с использованием рекурсивного метода для обработки концов и скользящего среднего для центральной части.

    Args:
    series (pd.Series): Временной ряд.
    window_size (int): Размер окна для сглаживания.

    Returns:
    pd.Series: Сглаженный временной ряд с равной длиной.
    """
    # Применение rolling для центральных значений
    rolling_avg = series.rolling(window=window_size, center=True).mean()

    # Рекурсивное сглаживание для левого края
    def smooth_left(index, left_points):
        if index < 0:
            return
        rolling_avg.iloc[index] = (series.iloc[index - left_points:index + left_points + 1].mean())
        smooth_left(index - 1, left_points - 1)

    # Рекурсивное сглаживание для правого края
    def smooth_right(index, right_points):
        if index >= len(series):
            return
        rolling_avg.iloc[index] = (series.iloc[index - right_points:index + right_points + 1].mean())
        smooth_right(index + 1, right_points - 1)

    # Начало рекурсивного сглаживания
    smooth_left(window_size // 2, window_size // 2)
    smooth_right(len(series) - window_size // 2 - 1, window_size // 2)

    return rolling_avg


# Очистка от выбросов, нормализация и детрендирование сигнала по скользящему окну
def remove_trend_with_sliding_window(series, window_size=30):
    """
    Нормирование сигнала по скользящему окну для удаления плавающего тренда и замены значений, выходящих за 1.5 стандартного отклонения.

    Args:
    series (pd.Series): Временной ряд для обработки.
    window_size (int): Размер окна для вычисления скользящих статистик.

    Returns:
    pd.Series: Сигнал с убранным трендом.
    """
    half_window = window_size // 2

    # Замена значений, выходящих за 1.5 стандартного отклонения в исходном сигнале
    std_dev = series.std()
    series = pd.Series(np.clip(series, -3 * std_dev, 3 * std_dev))

    # Центрирование сигнала (вычитание скользящего среднего)
    rolling_avg = series.rolling(window=window_size, center=True).mean()

    # Вычисление скользящего стандартного отклонения
    rolling_std = series.rolling(window=window_size, center=True).std()

    # Заполнение концов для соответствия длине исходного сигнала
    rolling_avg = rolling_avg.fillna(method='bfill').fillna(method='ffill')
    rolling_std = rolling_std.fillna(method='bfill').fillna(method='ffill')

    # Центрирование сигнала
    centered_signal = series - rolling_avg

    # Получение огибающей пиков центрированного сигнала и нормирование по ней
    peaks_envelope = centered_signal.abs().rolling(window=window_size, center=True).max()
    peaks_envelope = peaks_envelope.fillna(method='bfill').fillna(method='ffill')
    detrended_signal = centered_signal / (peaks_envelope + 1e-8)

    return detrended_signal

def get_peak_distances(peaks):
    """
    Вычисление расстояний между соседними элементами списка пиков.

    Args:
    peaks (list of int): Список индексов пиков.

    Returns:
    list of int: Список расстояний между соседними пиками.
    """
    return np.diff(peaks).tolist()

# Ищем пики на корреляционной кривой
def find_peaks_in_curve(curve_values, sample_len, threshold = 0.5, prominence=0.2):
        threshold = 0.5  # Устанавливаем порог для поиска похожих сегментов
        prominence=0.2
        distance=int(0.45*sample_len)
        height =0.6
        # Найти пики, включая края с учетом плоских вершин
        peaks, _ = find_peaks(curve_values, prominence=prominence, height=height)

        filtered_peaks = []
        # Отфильтровываем пики с дистанцией менее половины сэмпла
        for peak in peaks:
            if not filtered_peaks or (peak - filtered_peaks[-1]) >= distance:
                filtered_peaks.append(peak)

        print('distance,peaks',distance,peaks)
        print('расстояния между пиками', np.diff(peaks).tolist())
        print('прореженные пики', filtered_peaks)

        # Отфильтровываем крайние пики, если они имеют сильно меньшую дистанцию
        if len(filtered_peaks) < 3:
            return filtered_peaks

        distances = get_peak_distances(peaks)
        median_distance = np.median(distances)
        trimmed_peaks = filtered_peaks.copy()

        if distances[0] < 0.5*median_distance:
            trimmed_peaks.pop(0)
        if distances[-1] < 0.5*median_distance:
            trimmed_peaks.pop(-1)

        return trimmed_peaks


def display_curves_in_diff_plots(disp_curves,plot_name, sample_len, peaks):

    for curv_name,curve_values in disp_curves.items():
        plt.figure(figsize=(20, 5))
        plt.plot(range(len(curve_values)), curve_values, label=f'{curv_name}: {plot_name}', linestyle='-', linewidth=1)

        for peak in peaks:
            plt.axvline(x=peak, color='red', linestyle='--', alpha=0.7)

        # plt.axhline(y=threshold, color='green', linestyle='--', alpha=0.7)


        # Подсчет количества пиков
        # counts.append(len(peaks))
        plt.title(f'{plot_name} - Number of Peaks: {len(peaks)}')
        plt.xlabel('Порядковый индекс')
        plt.ylabel('Амплитуда')
        plt.legend()
        plt.grid(True)
        plt.show()

        return len(peaks)

def display_curves_in_one_plot(disp_curves,plot_name,sample_len, peaks):

    plt.figure(figsize=(20, 5))
    for curv_name,curve_values in disp_curves.items():

        plt.plot(range(len(curve_values)), curve_values, label=f'{curv_name}: {plot_name}', linestyle='-', linewidth=1)

    for peak in peaks:
        plt.axvline(x=peak, color='red', linestyle='--', alpha=0.7)

    # plt.axhline(y=threshold, color='green', linestyle='--', alpha=0.7)

    # Подсчет количества пиков
    # counts.append(len(peaks))
    plt.title(f'{plot_name} - Number of Peaks: {len(peaks)}')
    plt.xlabel('Порядковый индекс')
    plt.ylabel('Амплитуда')
    plt.legend()
    plt.grid(True)
    plt.show()

def dynamic_peak_normalization(series, window_size=30):
    """
    Динамическое центрирование и нормирование ряда по скользящему окну.

    Args:
    series (pd.Series): Временной ряд для нормирования.
    window_size (int): Размер окна для нормирования.

    Returns:
    pd.Series: Центрированный и нормированный временной ряд в диапазоне от -1 до +1.
    """
    if isinstance(series, np.ndarray):
        series = pd.Series(series)

    # # Центрирование сигнала (вычитание скользящего среднего)
    # rolling_mean = series.rolling(window=window_size, center=True).mean()
    # rolling_mean = rolling_mean.fillna(method='bfill').fillna(method='ffill')
    # centered_series = series - rolling_mean

    # Нормирование сигнала по скользящему максимуму
    rolling_max = series.rolling(window=window_size, center=True).max()
    rolling_max = rolling_max.fillna(method='bfill').fillna(method='ffill')
    normalized_series = series / (rolling_max + 1e-8)
    normalized_series = normalized_series.clip(-1, 1)

    return normalized_series

def match_sample_in_series(df_keypoints, df_sample, significant_series_names):
    # Тестовая печать форматов данных перед вызовом функции
    print(f"Формат df_keypoints: {df_keypoints.shape}")
    print(f"Формат df_sample: {df_sample.shape}")
    """
    Находит количество повторений сэмпла в каждом выбранном ряду DataFrame путем вычисления скользящей корреляции и поиска пиков.

    Параметры:
    - df_keypoints: DataFrame с временными рядами для анализа
    - df_sample: DataFrame с временными рядами сэмпла для поиска

    Возвращает медианное округленное значение количества сэмплов в рядах.
    """

    counts = []
    for column in significant_series_names:
        if column in df_sample.columns:
            series = df_keypoints[column]
            sample = df_sample[column]

            # Проверяем, что ряды не пустые
            if series.empty or sample.empty:
                print(f"Пустой ряд для колонки: {column}")
                continue

            # display('match_sample_in_series: series, sample', series, sample)

            orig_curves={}
            disp_curves={}
            sample_len = len(sample)

            orig_curves['Sample']=series_normalization(sample)
            orig_curves['Series']=series_normalization(series)
            # Применение фильтра Савицкого-Голея
            window_size = 31  # должен быть нечетным
            poly_order = 3
            filtered_series = savgol_filter(series, window_size, poly_order)
            window_size = 31  # должен быть нечетным
            filtered_sample = savgol_filter(sample, window_size, poly_order)

            # Нормирование сэмпла и ряда перед сравнением
            norm_series = series_normalization(filtered_series)
            norm_sample = series_normalization(filtered_sample)
            orig_curves['Filtered Sample']=norm_sample
            orig_curves['Filtered Series']=norm_series

            peaks=find_peaks_in_curve(norm_series, sample_len)
            display_curves_in_one_plot(orig_curves,column,sample_len,peaks)

            # display('match_sample_in_series: normalized series, sample', series, sample)

            # Вычисление полной корреляции
            # correlation_curve = series_normalization(compute_full_correlation(series, sample))
            # correlation_curve = np.correlate(series, sample, mode='same')
            correlation_curve = compute_full_correlation(filtered_series, filtered_sample)
            disp_curves['correlation_curve']=series_normalization(correlation_curve)

            # Вычисление нормализованной кривой корреляции по скользящему окну  в 7 периодов. Если их меньше, то обычная нормализация
            norm_correlation_curve= dynamic_peak_normalization(correlation_curve,min(sample_len,len(correlation_curve)))
            disp_curves['norm_correlation_curve']=norm_correlation_curve

            # # Вычисление сглаженной корреляционной кривой
            # smothed_correlation_curve = adaptive_moving_average(pd.Series(norm_correlation_curve),window_size=60)
            # disp_curves['smothed_correlation_curve']=series_normalization(smothed_correlation_curve)

            # # Вычисление среднего корреляционной кривой
            # avg_correlation_curve = adaptive_moving_average(pd.Series(correlation_curve),window_size=200)
            # disp_curves['avg_correlation_curve']=avg_correlation_curve

            # # Центрирование корреляц кривой
            # centered_correlation_curve = smothed_correlation_curve-avg_correlation_curve
            # disp_curves['centered_correlation_curve']=centered_correlation_curve

            # # Динамически Нормируем корреляционную кривую для выравнивания пиков по огибающей
            # lifted_correlation_curve = dynamic_normalization(centered_correlation_curve, 60)
            # disp_curves['lifted_correlation_curve']=lifted_correlation_curve

            # Ищем пики
            peaks=find_peaks_in_curve(norm_correlation_curve, sample_len)
            display_curves_in_one_plot({"norm_correlation_curve":norm_correlation_curve},'Итоговые пики',sample_len, peaks)
            counts.append(len(peaks))

    # Возвращаем медианное округленное значение количества сэмплов
    if counts:
        return int(np.round(np.median(counts)))
    else:
        return 0