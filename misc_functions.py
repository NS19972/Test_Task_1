import numpy as np
from Data_Formation import get_dataframe
import streamlit as st
from streamlit import session_state as sst


# Функция для взвешивания важности разных классов (чем реже класс наблюдается, тем больше у него вес)
def calculate_class_weights(y):
    unique_categories, counts = np.unique(y, return_counts=True)
    counts = counts/counts.sum()
    weights_dict = {i: j for i, j in enumerate(counts)}
    return weights_dict


# Функция для выбора столбцов нейронной сети внутри приложения
def select_columns(file):
    if file is not None:
        train_dataframe, _ = get_dataframe(file)
        dataset_columns = st.sidebar.multiselect("Выберите столбцы для обучения нейросети",
                       options=train_dataframe.columns.values, default=train_dataframe.columns.values
                       )
    return file, dataset_columns


# Колбэк который запоминает что кнопка "train button" была нажата
def train_buttons_callback():
    sst.train_button_clicked = True
    sst.model_trained = False


# Колбэк который запоминает что кнопка "test button" была нажата
def test_buttons_callback():
    sst.train_button_clicked = True


# Колбэк который сбрасывает вывод (точность) при изменении выбора алгоритма
def possible_algorithms_callback():
    sst.train_button_clicked = False
