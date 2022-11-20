import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder

from constants import *

np.random.seed(seed)   # Устанавливаем сид (sklearn использует сид от numpy)


# Функция, которая извлекает и обрабатывает данные из файла Education.csv
# Данные парсяться, чтобы получить категорию образования сотрудника, которая отображается в виде числа
def get_education_info(input_data):
    ALLOWED_COLUMNS = ['id', 'Вид образования', 'Специальность']  # Исключаем данные о специальности, т.к. там слишком много категорий

    education_data = pd.read_csv('dataset/Education.csv')  # Читаем файл
    education_data = education_data[ALLOWED_COLUMNS]       # Оставляем только выбранные столбцы

    education_data.drop_duplicates('id', inplace=True)     # Сбрасываем дубликаты (это нужно чтобы не было проблем позже - вопрос - почему у нас изначально дубликаты?)
    education_data.set_index('id', inplace=True)           # Устанавливаем столбец 'id' как индексовый

    data = input_data.join(education_data, how='left')     # Делаем left join, чтобы добавить данные только по тем сотрудникам, которые есть в train/test
    # Записываем 'Нет Данных' туда, где есть NaN
    data[education_data.columns.values] = data[education_data.columns.values].fillna("Нет данных")
    return data # Возвращаем данные и кодировщик

def process_education_info(data, encoders=None):
    # ПРИМЕЧАНИЕ: некоторые категории очень редко встречаются в датасете - по этому мы их просто пометим как Other (другое)
    encoders_to_return = {}
    for column in ['Вид образования', 'Специальность']:
        if column not in data.columns.values:
            continue

        if encoders is None:
            column_encoder = LabelEncoder()     # Создаем энкодер для строковых данных
            column_encoder.fit(data[column])
            encoders_to_return[column] = column_encoder
            # Переводим все строки в числа с помощью кодировщика
            data[column] = column_encoder.transform(data[column])
        else:
            column_encoder = encoders[column]
            data_to_transform = data[column]
            data_to_transform = data_to_transform.map(lambda s: '<unknown>' if s not in column_encoder.classes_ else s)
            column_encoder.classes_ = np.append(column_encoder.classes_, '<unknown>')
            # Переводим все строки в числа с помощью кодировщика
            data[column] = column_encoder.transform(data_to_transform)

    return data, encoders_to_return

# Функция которая извлекает и обрабатывает данные из файла Tasks.csv
# Данные парсяться чтобы получить количество просроченных и не-просроченных задач каждого сотрудника
def get_tasks_info(input_data):
    # Берём только столбец 'Статус по просрочке' из этого датасета (остальные данные либо признаны не важными, либо слишком несбалансированные)
    # ПРИМЕЧАНИЕ - У нас примерно в 5 раз больше предметов с просрочкой чем без (Статус по просрочке)
    # Позже столбец по просрочкам делится на два столбца - 'Просрочено' и 'Не Просрочено'
    # Для каждого работника мы суммируем количество просроченных и не-просроченных задач
    ALLOWED_COLUMNS = ['id', 'Not Lates', 'Lates']

    tasks_data = pd.read_csv('dataset/Tasks.csv', dtype='object')  # Читаем датасет

    tasks_encoder = {'Без нарушения срока': 0, 'С нарушением срока': 1}

    lates_column = [tasks_encoder[i] for i in tasks_data['Статус по просрочке'].values]  # Кодируем строки в числа (0 или 1 по сколько у нас две категории)

    n_values = 2  # Указываем, что у нас всегда 2 категории
    lates_column_categorical = np.eye(n_values)[lates_column]   # Переводим числа в onehot вектор
    tasks_data['Not Lates'] = lates_column_categorical[:, 0]   # Записываем все не просроченные задачи в один столбец (0 или 1)
    tasks_data['Lates'] = lates_column_categorical[:, 1]       # Записываем все просроченные задачи в другой столбец (0 или 1)

    tasks_data = tasks_data[ALLOWED_COLUMNS]   # Берём только ранее указанные столбцы

    # Groupby суммирует все значения для каждого сотрудника
    # таким образом мы получаем количество просроченных и не просроченных задач
    tasks_data = tasks_data.groupby('id').sum()

    data = input_data.join(tasks_data, how='left')  # Стоит отметить - из-за join появляются NaN-ы по сколько не для всех работников есть данные
    data[tasks_data.columns.values] = data[tasks_data.columns.values].fillna(0)   # Везде где у нас нет данных на количество задач - записываем 0

    return data  # Возвращаем данные и кодировщик


# Функция, которая извлекает и обрабатывает данные из файла SKUD.csv
# Данная функция считает общее число часов учётом обедов и без для каждого сотрудника
def get_skud_data(input_data):
    # Берём только два новых столбца из этого файла: количество часов потраченных на работу с одебом и без обедов
    ALLOWED_COLUMNS = ['id', 'Длительность общая']
    skud_data = pd.read_csv('dataset/SKUD.csv') # Читаем файл
    skud_data = skud_data[ALLOWED_COLUMNS]   # Берём только нужные столбцы

    # Меняем все запятые в числовых столбцах на точки, чтобы можно было перевести в числовой тип данных
    for column in ALLOWED_COLUMNS[1:]:
        skud_data[column] = skud_data[column].str.replace(',', '.').astype(float)

    skud_data = skud_data.groupby('id').sum()      # Группируем данные и суммируем, чтобы получить общее количество часов для каждого сотрудника
    data = input_data.join(skud_data, how='left')  # Делаем join, чтобы добавить данные только для сотрудников из обучающей выборки
    data[skud_data.columns.values] = data[skud_data.columns.values].fillna(0)  # Записываем 0 туда, где нет данных
    return data


#
def get_connection_data(input_data):
    ALLOWED_COLUMNS = ['id', 'Время опоздания', 'Признак опоздания']
    connection_data = pd.read_csv('dataset/ConnectionTime.csv', dtype=str)

    connection_data['Признак опоздания'][connection_data['Признак опоздания'] == 'Опоздание'] = 1
    connection_data['Время опоздания'] = connection_data['Время опоздания'].str.replace(',', '.').astype(float)
    connection_data = connection_data[ALLOWED_COLUMNS].fillna(0)

    connection_data = connection_data.groupby('id').sum()
    data = input_data.join(connection_data, how='left')
    data[connection_data.columns.values] = data[connection_data.columns.values].fillna(0)  # Записываем 0 туда, где нет данных
    return data


#
def get_working_data(input_data):
    ALLOWED_COLUMNS = ['id', 'activeTime', 'monitorTime']
    working_data = pd.read_csv('dataset/WorkingDay.csv')
    working_data = working_data[ALLOWED_COLUMNS]
    working_data[['activeTime', 'monitorTime']] = working_data[['activeTime', 'monitorTime']].astype(float)
    working_data.rename({'monitorTime': 'monitorTimeWorking'}, axis=1, inplace=True)
    working_data = working_data.groupby('id').sum()
    data = input_data.join(working_data, how='left')
    data[working_data.columns.values] = data[working_data.columns.values].fillna(0)  # Записываем 0 туда, где нет данных
    return data


#
def get_network_data(input_data):
    ALLOWED_COLUMNS = ['id', 'monitor_Time']
    network_data = pd.read_csv('dataset/TimenNetwork.csv')
    network_data = network_data[ALLOWED_COLUMNS]
    network_data.rename({'monitor_Time': 'monitorTimeNetwork'}, axis=1, inplace=True)
    network_data = network_data.groupby('id').sum()
    data = input_data.join(network_data, how='left')
    data[network_data.columns.values] = data[network_data.columns.values].fillna(0)  # Записываем 0 туда, где нет данных
    return data


#
def get_calls_data(input_data):
    # Берём только эти столбцы из датасета
    ALLOWED_COLUMNS = ['id', 'NumberOfCalls', 'InOut', 'CallTime']
    calls_data = pd.read_csv('dataset/Calls.csv', dtype=str)
    calls_data = calls_data[ALLOWED_COLUMNS]

    for number_column in ['NumberOfCalls', 'CallTime']:
        calls_data[number_column] = calls_data[number_column].str.replace(',', '.').astype(float)

    in_calls = calls_data[calls_data['InOut'] == 'ToUser']
    out_calls = calls_data[calls_data['InOut'] == 'FromUser']

    in_calls = in_calls.groupby('id').sum(numeric_only=True).rename(
        {'NumberOfCalls': 'NumberOfInCalls', 'CallTime': 'InCallTime'}, axis=1
    )
    out_calls = out_calls.groupby('id').sum(numeric_only=True).rename(
        {'NumberOfCalls': 'NumberOfOutCalls', 'CallTime': 'OutCallTime'}, axis=1
    )

    data = input_data.join(in_calls, how='left')
    data = data.join(out_calls, how='left')
    columns_to_fill = in_calls.columns.values.tolist() + out_calls.columns.values.tolist()
    data[columns_to_fill] = data[columns_to_fill].fillna(0)          # Записываем 0 туда, где нет данных
    return data


# Функция, которая получает файл и извлекает соответствующий датафрейм вместе со всеми фичами
@st.cache(allow_output_mutation=True)
def get_dataframe(file):
    data = pd.read_csv(file, index_col='id')
    data = get_tasks_info(data)
    data = get_skud_data(data)
    data = get_connection_data(data)
    data = get_working_data(data)
    data = get_network_data(data)
    data = get_calls_data(data)
    data = get_education_info(data)
    return data


# Главная функция файла - извлекает и обрабатывает все данные для обучения и валидации
@st.cache(allow_output_mutation=True)
def get_train_dataset(train_data, val_percentage, scale_data=False, onehot_encode=False):
    train_data, education_encoders = process_education_info(train_data)
    # Масштабирование скалярных значений (трансформация столбцов в ст. распределение)
    scaler = StandardScaler()
    if scale_data:  # Скалируем релевантные столбцы (если алгоритм этого требует)
        common_columns = [i for i in train_data.columns.values if i in scalar_columns]  # Список всех столбцов которые подлежат скалированию
        train_data[common_columns] = scaler.fit_transform(train_data[common_columns])  # Скалируем выше-указанные столбцы

    onehot_encoders = {}
    if onehot_encode:  # Кодируем релевантные столбцы в one-hot encoding (если алгоритм этого требует)
        columns_in_dataset = [i for i in categorical_columns if i in train_data.columns]  # Список всех столбцов которые подлежат onehot кодированию
        arrays_store = []
        for column in columns_in_dataset:
            if column in train_data.columns:
                OHE_encoder = OneHotEncoder(sparse=False, min_frequency=10, handle_unknown='infrequent_if_exist')
                onehot_data = OHE_encoder.fit_transform(train_data[column].values.reshape(-1, 1))
                onehot_encoders[column] = OHE_encoder
                arrays_store.append(onehot_data)

        train_data.drop(columns_in_dataset, axis=1, inplace=True)

    # Делим итоговый датафрейм на х и у
    x, y = train_data.loc[:, train_data.columns != 'type'].values, train_data.loc[:, train_data.columns == 'type'].values

    if onehot_encode:  # Если мы делали OHE, склеиваем OHE массивы в х
        x = np.concatenate([x] + arrays_store, axis=1)

    label_encoders = {}  # Добавляем все лейбл-энкодеры в единый словарь
    if education_encoders is not None:
        label_encoders.update(education_encoders)

    # Если нет валидационной выборки, возвращаем х и у как целые массивы
    if val_percentage == 0:
        return x, None, y, None, label_encoders, onehot_encoders, scaler
    # Иначе, делаем train_test_split и возвращаем результат
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=val_percentage)
    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.5)
    return x_train, x_val, x_test, y_train, y_val, y_test, label_encoders, onehot_encoders, scaler


@st.cache(allow_output_mutation=True)
def get_test_dataset(file, dataset_columns, label_encoders, onehot_encoders, scaler, scale_data=False, onehot_encode=False):
    test_data = get_dataframe(file)
    data_index = test_data.index.values

    test_data = test_data[dataset_columns]
    test_data, _ = process_education_info(test_data, label_encoders)

    if scale_data:  # Скалируем релевантные столбцы (если алгоритм этого требует)
        common_columns = [i for i in test_data.columns.values if i in scalar_columns]  # Список всех столбцов которые подлежат скалированию
        test_data[common_columns] = scaler.transform(test_data[common_columns])  # Скалируем выше-указанные столбцы

    if onehot_encode:  # Кодируем релевантные столбцы в one-hot encoding (если алгоритм этого требует)
        columns_in_dataset = [i for i in categorical_columns if i in test_data.columns]  # Список всех столбцов которые подлежат onehot кодированию
        arrays_store = []
        for column in columns_in_dataset:
            if column in test_data.columns:
                onehot_data = onehot_encoders[column].transform(test_data[column].values.reshape(-1, 1))
                arrays_store.append(onehot_data)
        test_data.drop(columns_in_dataset, axis=1, inplace=True)

    # Делим итоговый датафрейм на х и у
    x_test, y_test = test_data.loc[:, test_data.columns != 'type'].values, test_data.loc[:, test_data.columns == 'type'].values

    if onehot_encode:  # Если мы делали OHE, склеиваем OHE массивы в х
        x_test = np.concatenate([x_test] + arrays_store, axis=1)

    return x_test, y_test, data_index



