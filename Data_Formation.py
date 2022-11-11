import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder

from constants import *

np.random.seed(seed)   #Устанавливаем сид (sklearn использует сид от numpy)

#Функция которая извлекает и обрабатывает данные из файла Education.csv
#Данные парсяться чтобы получить категорию образования соотрудника, которая отображается в виде числа
def get_education_info(input_data):
    education_encoder = LabelEncoder()            #Создаем энкодер для строковых данных
    ALLOWED_COLUMNS = ['id', 'Вид образования']   #Исключаем данные о специальности, т.к. там слишком много категорий

    education_data = pd.read_csv('dataset/Education.csv') #Читаем файл
    education_data = education_data[ALLOWED_COLUMNS]      #Оставляем только выбранные столбцы

    education_data.drop_duplicates('id', inplace=True)    #Сбрасываем дубликаты (это нужно чтобы не было проблем позже - вопрос - почему у нас изначально дубликаты?)
    education_data.set_index('id', inplace=True)          #Устанавливаем столбец 'id' как индексовый

    data = input_data.join(education_data, how='left')    #Делаем left join чтобы добавить данные только по тем соотрудникам, которые есть в train/test


    #ПРИМЕЧАНИЕ: некоторые категории очень редко встречаются в датасете - по этому мы их просто пометим как Other (другое)
    unique_categories, counts = np.unique(data['Вид образования'].astype(str).values, return_counts=True)  #Считаем как часто каждая категория встречается
    frequency_dict = dict(zip(unique_categories, counts))                                    #Создаем словарь из полученных категорий и сколько раз они встречаются

    #Переводим все редкие категории, а также категорию nan в Other
    infrequent_categories = [k for k, v in frequency_dict.items() if v < 10 or k == 'nan']
    for category in infrequent_categories:
        data.loc[data['Вид образования'] == category, 'Вид образования'] = 'Other'

    #Переводим все строки в числа с помощью кодировщика
    data['Вид образования'] = education_encoder.fit_transform(data['Вид образования'])
    return data, education_encoder #Возвращаем данные и кодировщик

#Функция которая извлекает и обрабатывает данные из файла Tasks.csv
#Данные парсяться чтобы получить количество просроченных и не-просроченных задач каждого соотрудника
def get_tasks_info(input_data):
    #Берём только столбец 'Статус по просрочке' из этого датасета (остальные данные либо признаны не важными, либо слишком разбалансированные)
    #ПРИМЕЧАНИЕ - У нас примерно в 5 раз больше предметов с просрочкой чем без (Статус по просрочке)
    tasks_encoder = LabelEncoder()
    #Позже столбец по просрочкам делится на два столбца - 'Просрочено' и 'Не Просрочено'
    #Для каждого работника мы суммируем количество просроченных и не-просроченных задач
    ALLOWED_COLUMNS = ['id', 'Not Lates', 'Lates']

    tasks_data = pd.read_csv('dataset/Tasks.csv', dtype='object') #Читаем датасет

    lates_column = tasks_encoder.fit_transform(tasks_data['Статус по просрочке'].values) #Кодируем строки в числа (0 или 1 по сколько у нас две категории)
    lates_column_categorical = OneHotEncoder(sparse=False).fit_transform(lates_column.reshape(-1, 1))  #Переводим числа в onehot вектор
    tasks_data['Not Lates'] = lates_column_categorical[:, 0] #Записываем все не просроченные задачи в один столбец (0 или 1)
    tasks_data['Lates'] = lates_column_categorical[:, 1]     #Записываем все просроченные задачи в другой столбец (0 или 1)

    tasks_data = tasks_data[ALLOWED_COLUMNS]   #Берём только ранее указанные столбцы

    #Groupby суммирует все значения для каждого соотрудника
    #таким образом мы получаем количество просроченных и не просроченных задач
    tasks_data = tasks_data.groupby('id').sum()

    data = input_data.join(tasks_data, how='left') #Стоит отметить - из-за join появляются NaN-ы по сколько не для всех работников есть данные
    data.fillna(-1, inplace=True)   #Везде где у нас нет данных на количество задач - записываем -1

    return data, tasks_encoder #Возвращаем данные и кодировщик

#Функция которая извлекает и обрабатывает данные из файла SKUD.csv
#Данная функция считает общее число часов учётом обедов и без для каждого соотрудника
def get_skud_data(input_data):
    #Берём только два новых столбца из этого файла: количество часов потраченных на работу с одебом и без обедов
    ALLOWED_COLUMNS = ['id', 'Длительность общая', 'Длительность раб.дня без обеда']
    skud_data = pd.read_csv('dataset/SKUD.csv') #Читаем файл
    skud_data = skud_data[ALLOWED_COLUMNS]   #Берём только нужные столбцы

    #Меняем все запятые в числовых столбцах на точки, чтобы можно было перевести в числовой тип данных
    for column in ALLOWED_COLUMNS[1:]:
        skud_data[column] = skud_data[column].str.replace(',', '.').astype(float)

    skud_data = skud_data.groupby('id').sum()     #Группируем данные и суммируем чтобы получить общее количество часов для каждого соотрудника
    data = input_data.join(skud_data, how='left') #Делаем join чтобы добавить данные только для соотрудников из обучающей выборки
    data.fillna(-1, inplace=True)          #Записываем -1 туда, где нет данных
    return data

def get_connection_data(input_data):
    ALLOWED_COLUMNS = ['id', 'Время опоздания', 'Признак опоздания']
    connection_data = pd.read_csv('dataset/ConnectionTime.csv', dtype=str)

    connection_data['Признак опоздания'][connection_data['Признак опоздания'] == 'Опоздание'] = 1
    connection_data['Время опоздания'] = connection_data['Время опоздания'].str.replace(',', '.').astype(float)
    connection_data = connection_data[ALLOWED_COLUMNS].fillna(0)

    connection_data = connection_data.groupby('id').sum()
    data = input_data.join(connection_data, how='left')
    data.fillna(-10, inplace=True)          #Записываем -10 туда, где нет данных
    return data

def get_working_data(input_data):
    ALLOWED_COLUMNS = ['id', 'activeTime', 'monitorTime']
    working_data = pd.read_csv('dataset/WorkingDay.csv')
    working_data = working_data[ALLOWED_COLUMNS]
    working_data[['activeTime', 'monitorTime']] = working_data[['activeTime', 'monitorTime']].astype(float)
    working_data = working_data.groupby('id').sum()
    data = input_data.join(working_data, how='left')
    data.fillna(-1e6, inplace=True)          #Записываем -1e6 туда, где нет данных
    return data

def get_network_data(input_data):
    ALLOWED_COLUMNS = ['id', 'monitor_Time']
    network_data = pd.read_csv('dataset/TimenNetwork.csv')
    network_data = network_data[ALLOWED_COLUMNS]
    network_data = network_data.groupby('id').sum()
    data = input_data.join(network_data, how='left')
    data.fillna(-1e6, inplace=True)          #Записываем -1e6 туда, где нет данных
    return input_data

#Главная функция файла - извлекает и обрабатывает все данные для обучения и валидации
def get_train_dataset():
    train_data = pd.read_csv('dataset/train.csv', index_col='id')
    train_data, education_encoder = get_education_info(train_data)
    train_data, tasks_encoder = get_tasks_info(train_data)
    train_data = get_skud_data(train_data)
    train_data = get_connection_data(train_data)
    train_data = get_working_data(train_data)
    train_data = get_network_data(train_data)

    #Масштабирование скалярных значений (преведенье столбцов в ст. распределение)
    scaler = StandardScaler()
    if scale_data:
        # Список всех столбцов которые подлежат скалированию
        scalar_columns = ['Not Lates', 'Lates', 'Длительность общая', 'activeTime', 'monitorTime',
                          'Длительность раб.дня без обеда', 'Время опоздания', 'Признак опоздания']

        train_data[scalar_columns] = scaler.fit_transform(train_data[scalar_columns]) #Скалируем выше-указанные столбцы

    if onehot_encode:
        categorical_columns = ['Вид образования'] #Список всех столбцов которые подлежат onehot кодированию
        arrays_store = []
        for column in categorical_columns:
            onehot_data = OneHotEncoder(sparse=False).fit_transform(train_data[column].values.reshape(-1, 1))
            arrays_store.append(onehot_data)
        train_data.drop(categorical_columns, axis=1, inplace=True)

    x, y = train_data.loc[:, train_data.columns != 'type'].values, train_data.loc[:, train_data.columns == 'type'].values

    if onehot_encode:
        x = np.concatenate([x] + arrays_store, axis=1)

    encoders = {'education_encoder':education_encoder, 'tasks_encoder':tasks_encoder}

    x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=train_size)
    return x_train, x_val, y_train, y_val, encoders, scaler

#Код ниже используется для дебаггинга/проверки данных
if __name__ == "__main__":
    get_train_dataset()


