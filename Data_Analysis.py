import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os

from Data_Formation import *

@st.cache
def analyze_class_frequency(dataset):
    def func(pct):
        return "{:1.1f}%".format(pct)

    classes = dataset['type']
    unique_categories, counts = np.unique(classes.values, return_counts=True)  # Считаем как часто каждая категория встречается

    plt.figure(figsize = (14, 8))
    plt.pie(counts, labels = [f"Class {i}" for i in range(len(counts))], autopct=lambda pct: func(pct))
    plt.title("Относительное количество каждого класса в выборке")
    plt.show()


#Данная функция выводит КУМУЛЯТИВНЫЕ гистограммы
#Именно кумулятивные гистограммы нам показывают, что почти все данные для всех столбцов находяться около нуля
#Примерно половина данных имеют нулевые значения (от части из-за того, что nan-ы заполняются нулями)
@st.cache
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

@st.cache
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

@st.cache
def create_heatmap(dataset):
    correlation_matrix = dataset.corr(method='spearman')
    plt.figure(figsize=(14, 8))
    sns.heatmap(correlation_matrix, annot=True)
    plt.title("Матрица корреляции")
    plt.show()

@st.cache
def create_heatmap_streamlit(dataset):
    correlation_matrix = dataset.corr(method='spearman')
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(correlation_matrix, annot=True, ax=ax)
    ax.set_title("Матрица корреляции")
    st.sidebar.write(fig)


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