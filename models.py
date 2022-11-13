import numpy as np
import sklearn
import tensorflow as tf
import streamlit

from sklearn.ensemble import *
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score
from sklearn.svm import *
from constants import *

np.random.seed(seed)   #Устанавливаем сид (sklearn использует сид от numpy)

class GradientBoostingAlgorithm:
    def __init__(self):
        self.model = GradientBoostingClassifier(n_estimators=40, max_depth=2)  # Задаем параметры алгоритма

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train.flatten())  # Обучаем алгоритм

    def validate(self, x_val, y_val):
        y_pred = self.model.predict(x_val)                                     #Извлекаем предсказание
        score = recall_score(y_val, y_pred, average='micro')                   #Считаем метрику
        print(f"Model recall score on validation subset is {score}")
        return score

    def test(self, x_test, y_test):
        y_pred = self.model.predict(x_test)
        score = recall_score(y_test, y_pred, average='micro')
        print(f"Model recall score on test subset is {score}")
        return score


class SVMAlgorithm(GradientBoostingAlgorithm):
    def __init__(self):
        self.model = SVC()  # Создаем объект алгоритма SVC

#Практика показывает, что именно дерево решений достигает максимально высокой точности на тестовой выборке (~30%)
#Однако при этом, точность на валидационной выборке значительно падает (до 46%)
class DecisionTreeAlgorithm(GradientBoostingAlgorithm):
    def __init__(self):
        self.model = DecisionTreeClassifier()  # Создаем объект алгоритма DecisionTreeClassifier

class RandomForestAlgorithm(GradientBoostingAlgorithm):
    def __init__(self):
        self.model = RandomForestClassifier

class GaussianAlgorithm(GradientBoostingAlgorithm):
    def __init__(self):
        self.model = GaussianProcessClassifier()  # Создаем объект алгоритма Gaussian


class NeuralNetwork:
    def __init__(self):
        self.model = tf.keras.Sequential(
            layers=[tf.keras.layers.Dense(i, activation='relu') for i in neural_network_hidden_neurons]
        )
        self.model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=NN_learning_rate), loss='sparse_categorical_crossentropy'
        )

    def train(self, x_train, y_train):
        class_weights = self.calculate_class_weights(y_train) if use_class_weights else None
        x_train = np.array(x_train, ndmin=2)
        y_train = np.array(y_train, ndmin=2)
        self.model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, class_weight=class_weights)

    def validate(self, x_val, y_val):
        x_val = np.array(x_val, ndmin=2)
        y_val = np.array(y_val, ndmin=2)
        y_pred = self.model.predict(x_val, batch_size=batch_size)
        y_pred = np.argmax(y_pred, axis=1)
        score = recall_score(y_val.flatten(), y_pred.flatten(), average='micro')
        print(f"Model recall score on validation subset is {score}")
        return score

    def test(self, x_test, y_test):
        x_test = np.array(x_test, ndmin=2)
        y_test = np.array(y_test, ndmin=2)
        y_pred = self.model.predict(x_test, batch_size=batch_size)
        y_pred = np.argmax(y_pred, axis=1)
        score = recall_score(y_test.flatten(), y_pred.flatten(), average='micro')
        print(f"Model recall score on test subset is {score}")
        return score

    @classmethod
    def calculate_class_weights(cls, y):
        unique_categories, counts = np.unique(y, return_counts=True)
        counts = counts/counts.sum()
        weights_dict = {i: j for i, j in enumerate(counts)}
        return weights_dict