import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
import os

from Data_Formation import *


def analyze_class_frequency(dataset):
    def func(pct):
        return "{:1.1f}%".format(pct)

    classes = dataset['type']
    unique_categories, counts = np.unique(classes.values, return_counts=True)  # Считаем как часто каждая категория встречается

    plt.figure(figsize=(14, 8))
    plt.pie(counts, labels=[f"Class {i}" for i in range(len(counts))], autopct=lambda pct: func(pct))
    plt.title("Относительное количество каждого класса в выборке")
    plt.show()


@st.cache(allow_output_mutation=True)
def create_cdf_streamlit(data_column):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.hist(data_column, bins=200, color='red', cumulative=True, alpha=0.5)
    ax.set_title('кумулятивная функция распределения', fontsize=16)
    ax.set_xlabel('значение параметра', fontsize=14)
    ax.set_ylabel('доля сотрудников <= параметру', fontsize=14)
    ax.grid()
    return fig


@st.cache(allow_output_mutation=True)
def create_histogram_streamlit(data_column):
    # Считаем как часто каждая категория встречается
    unique_categories, counts = np.unique(data_column.astype(str).values, return_counts=True)

    # Создаем словарь из полученных категорий и сколько раз они встречаются
    frequency_dict = dict(zip(unique_categories, counts))

    # Переводим все редкие категории, а также категорию nan в Other
    infrequent_categories = [k for k, v in frequency_dict.items() if v < 10 or k == 'nan']
    for category in infrequent_categories:
        data_column.loc[data_column == category] = 'Другое'

    unique_categories, counts = np.unique(data_column, return_counts=True)

    fig, ax = plt.subplots(figsize=(28, 20))
    ax.bar(unique_categories, counts, color='orange', align='center', alpha=0.5)
    ax.set_title('гистограмма', fontsize=16)
    ax.set_xlabel('категория', fontsize=14)
    ax.set_ylabel('число сотрудников подходящие под категорию', fontsize=14)
    ax.grid()
    return fig


# Функция analyze_class_frequency, адаптированна для стримлита
def analyze_class_frequency_streamlit(dataset):
    def func(pct):
        return "{:1.1f}%".format(pct)

    classes = dataset['type']
    unique_categories, counts = np.unique(classes.values, return_counts=True)  # Считаем как часто каждая категория встречается

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.pie(counts, labels=[f"Class {i}" for i in range(len(counts))], autopct=lambda pct: func(pct))
    ax.set_title("Относительное количество каждого класса в выборке")
    return fig


# Данная функция выводит КУМУЛЯТИВНЫЕ гистограммы
# Именно кумулятивные гистограммы нам показывают, что почти все данные для всех столбцов находяться около нуля
# Примерно половина данных имеют нулевые значения (от части из-за того, что nan-ы заполняются нулями)
def analyze_calls_data(dataset):
    dataset = dataset[['NumberOfInCalls', 'InCallTime', 'NumberOfOutCalls', 'OutCallTime']]
    colors_list = ['blue', 'red', 'green', 'orange']
    fig, ax = plt.subplots(2, 2, figsize=(16, 10))
    ax = ax.ravel()

    for i, a in enumerate(ax):
        a.hist(dataset.iloc[:, i], bins=200, color=colors_list[i], cumulative=True, alpha=0.5)
        a.set_title(dataset.columns[i])
        a.set_xlabel('Category Value')
        a.set_ylabel('Number of Instances')
        a.grid()

    plt.tight_layout()
    plt.show()


def analyze_monitor_data(dataset):
    dataset = dataset[['activeTime', 'monitorTimeWorking', 'monitorTimeNetwork']]
    colors_list = ['blue', 'green', 'orange']
    fig, ax = plt.subplots(1, 3, figsize=(24, 8))
    ax = ax.ravel()

    for i, a in enumerate(ax):
        a.hist(dataset.iloc[:, i], bins=200, color=colors_list[i], cumulative=True, alpha=0.5)
        a.set_title(dataset.columns[i])
        a.set_xlabel('Category Value')
        a.set_ylabel('Number of Instances')
        a.grid()

    plt.tight_layout()
    plt.show()


def create_heatmap(dataset):
    for column in dataset.columns:
        if dataset[column].dtype not in [int, float]:
            dataset[column] = LabelEncoder().fit_transform(dataset[column])
    correlation_matrix = dataset.corr(method='spearman')
    plt.figure(figsize=(14, 8))
    sns.heatmap(correlation_matrix, annot=True)
    plt.title("Матрица корреляции")
    plt.show()


# Функция create_heatmap, адаптированна для стримлита
def create_heatmap_streamlit(dataset):
    for column in dataset.columns:
        if dataset[column].dtype not in [int, float]:
            dataset[column] = LabelEncoder().fit_transform(dataset[column])


    correlation_matrix = dataset.corr(method='kendall')
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(correlation_matrix, annot=True, ax=ax)
    ax.set_title("Матрица корреляции")
    return fig


if __name__ == "__main__":
    train_data = pd.read_csv('dataset/train.csv', index_col='id')
    train_data, tasks_encoder = get_tasks_info(train_data)
    train_data = get_skud_data(train_data)
    train_data = get_connection_data(train_data)
    train_data = get_working_data(train_data)
    train_data = get_network_data(train_data)
    train_data = get_calls_data(train_data)
    train_data = get_education_info(train_data)
    train_data, _ = process_education_info(train_data)

    create_heatmap(train_data)
