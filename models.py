import numpy as np
import sklearn
import tensorflow as tf
import streamlit

from sklearn.ensemble import *
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score, accuracy_score
from sklearn.svm import *
from constants import *
from misc_functions import calculate_class_weights

np.random.seed(seed)   #Устанавливаем сид (sklearn использует сид от numpy)

class GradientBoostingAlgorithm:
    requires_OHE = False           #Градиентный бустинг не требует onehot кодирования
    requires_normalization = False #Градиентный бустинг не требует нормализации
    def __init__(self, **kwargs):
        self.model = GradientBoostingClassifier(
            n_estimators=kwargs['n_estimators'], max_depth=kwargs['GB_max_depth'],
            learning_rate=kwargs['GB_learning_rate'], min_samples_split=kwargs['min_samples_split'],
            random_state=kwargs['random_state']
            )  # Задаем параметры алгоритма

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train.flatten())  # Обучаем алгоритм

    def validate(self, x_val, y_val, subset_type='validation'):
        y_pred = self.model.predict(x_val)                                     #Извлекаем предсказание
        score_1 = recall_score(y_val, y_pred, average='micro')                   #Считаем метрику
        score_2 = accuracy_score(y_val, y_pred)                   #Считаем метрику
        print(f"Model recall score on {subset_type} subset is {score_1}")
        print(f"Model accuracy score on {subset_type} subset is {score_2}")
        return score_1, score_2

    def test(self, x_test, y_test):
        y_pred = self.model.predict(x_test)
        score_1 = recall_score(y_test, y_pred, average='micro')
        score_2 = accuracy_score(y_test, y_pred)
        print(f"Model recall score on test subset is {score_1}")
        print(f"Model accuracy score on test subset is {score_2}")
        return score_1, score_2, y_pred


class SVMAlgorithm(GradientBoostingAlgorithm):
    requires_OHE = True            #SVM ТРЕБУЕТ onehot кодирования
    requires_normalization = True  #SVM ТРЕБУЕТ нормализации
    def __init__(self, **kwargs):
        self.model = SVC(C=kwargs['C'], class_weight=kwargs['class_weight'], kernel=kwargs['kernel'],
                         random_state=kwargs['random_state'])  # Создаем объект алгоритма SVC


#Практика показывает, что именно дерево решений достигает максимально высокой точности на тестовой выборке (~30%)
#Однако при этом, точность на валидационной выборке значительно падает (до 46%)
class DecisionTreeAlgorithm(GradientBoostingAlgorithm):
    requires_OHE = False           #Дерево решений не требует onehot кодирования
    requires_normalization = False #Дерево решений не требует нормализации
    def __init__(self, **kwargs):
        self.model = DecisionTreeClassifier(max_depth=kwargs['Tree_max_depth'], criterion=kwargs['criterion'],
                                            min_samples_split=kwargs['min_samples_split'],
                                            random_state=kwargs['random_state']
                                            )  # Создаем объект алгоритма DecisionTreeClassifier


class RandomForestAlgorithm(GradientBoostingAlgorithm):
    requires_OHE = False           #Рандомный лес не требует onehot кодирования
    requires_normalization = False #Рандомный лес не требует нормализации
    def __init__(self, **kwargs):
        self.model = RandomForestClassifier(n_estimators=kwargs['n_estimators'], max_depth=kwargs['Tree_max_depth'],
                                            min_samples_split=kwargs['min_samples_split'],
                                            random_state=kwargs['random_state']
                                            ) # Создаем объект алгоритма RandomForestClassifier


class GaussianAlgorithm(GradientBoostingAlgorithm):
    requires_OHE = True           #Гауссовский алгоритм ТРЕБУЕТ onehot кодирования
    requires_normalization = True #Гауссовский алгоритм ТРЕБУЕТ onehot кодирования
    def __init__(self, **kwargs):
        self.model = GaussianProcessClassifier(max_iter_predict=kwargs['max_iter_predict'],
                                               warm_start=kwargs['warm_start'],
                                               random_state=kwargs['random_state'])  # Создаем объект алгоритма Gaussian


class NeuralNetwork:
    requires_OHE = True           #Нейронная сеть ТРЕБУЕТ onehot кодирования
    requires_normalization = True #Нейронная сеть ТРЕБУЕТ onehot кодирования
    def __init__(self, **kwargs):
        neural_network_hidden_neurons=self.filter_zeros(kwargs['neural_network_hidden_neurons'])
        np.random.seed(kwargs['random_state'])
        tf.keras.utils.set_random_seed(kwargs['random_state'])

        self.model = tf.keras.Sequential(
            layers=[tf.keras.layers.Dense(i, activation='relu') for i in neural_network_hidden_neurons]
        )
        self.model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=kwargs['NN_learning_rate']), loss='sparse_categorical_crossentropy'
        )

        self.kwargs = kwargs

    def train(self, x_train, y_train):
        class_weights = calculate_class_weights(y_train) if self.kwargs['use_class_weights'] else None
        x_train = np.array(x_train, ndmin=2)
        y_train = np.array(y_train, ndmin=2)
        self.model.fit(x_train, y_train, epochs=self.kwargs['num_epochs'], batch_size=self.kwargs['batch_size'], class_weight=class_weights)

    def validate(self, x_val, y_val, subset_type='validation'):
        x_val = np.array(x_val, ndmin=2)
        y_val = np.array(y_val, ndmin=2)
        y_pred = self.model.predict(x_val, batch_size=self.kwargs['batch_size'])
        y_pred = np.argmax(y_pred, axis=1)
        score_1 = recall_score(y_val.flatten(), y_pred.flatten(), average='micro')
        score_2 = accuracy_score(y_val.flatten(), y_pred.flatten())
        print(f"Model recall score on {subset_type} subset is {score_1}")
        print(f"Model accuracy score on {subset_type} subset is {score_2}")
        return score_1, score_2

    def test(self, x_test, y_test):
        x_test = np.array(x_test, ndmin=2)
        y_test = np.array(y_test, ndmin=2)
        y_pred = self.model.predict(x_test, batch_size=batch_size)
        y_pred = np.argmax(y_pred, axis=1)
        score_1 = recall_score(y_test.flatten(), y_pred.flatten(), average='micro')
        score_2 = accuracy_score(y_test.flatten(), y_pred.flatten())
        print(f"Model recall score on test subset is {score_1}")
        print(f"Model accuracy score on test subset is {score_2}")
        return score_1, score_2, y_pred.flatten()

    # Метод, который принимает на вход список (количество нейронов в каждом слое), и удаляет всё в списке, что находится
    # после первого нуля (включая сам 0).
    @classmethod
    def filter_zeros(cls, array):
        result = []
        for i in array:
            if i == 0:
                break
            result.append(i)
        return result
