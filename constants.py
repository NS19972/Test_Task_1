# ВСЕ КОНСТАНТЫ КОТОРЫЕ ИСПОЛЬЗУЮТСЯ В ПРОЕКТЕ

# General
seed = 100             # Сид (для фиксации результата)
train_size = 0.8       # Коэффициент деления на обучающую/валидационную выборку
num_classes = 4        # Количество классов в датасете

# Data Visualization
scalar_columns = ['Число задач без нарушения срока', 'Число задач с нарушением срока', 'Общая длительность прибывания в офисе',
                  'Активное время за клавиатурой', 'Активное время за монитором', 'Общее время опозданий', 'Общее количество опозданий',
                  'Время активности в сети', 'Время опоздания', 'Признак опоздания', 'Число входящих звонков',
                  'Время на входящие звонки', 'Число выходящих звонков', 'Время на выходящие звонки']

categorical_columns = ['Вид образования', 'Специальность']

# Classical ML
GB_learning_rate = 0.1    # Скорость обучения
n_estimators = 100   # Количество деревьев в алгоритме
GB_max_depth = 3          # Глубина деревьев
min_samples_split = 2
C = 1.0
Tree_max_depth = None
class_weight = None
warm_start = True
criterion = 'gini'
kernel = 'rbf'

# Neural Network
neural_network_hidden_neurons = [32, 16, 8]
NN_learning_rate = 1e-3
num_epochs = 10
use_class_weights = True
batch_size = 32

# Словарь аргументов
kwargs = {
    'GB_learning_rate': GB_learning_rate, 'n_estimators': n_estimators, 'GB_max_depth': GB_max_depth,
    'neural_network_hidden_neurons': neural_network_hidden_neurons, 'NN_learning_rate': NN_learning_rate,
    'num_epochs': num_epochs, 'use_class_weights': use_class_weights, 'batch_size': batch_size, 'C': C,
    'class_weight': class_weight, 'kernel': kernel, 'criterion': criterion, 'min_samples_split': min_samples_split,
    'Tree_max_depth': Tree_max_depth
}

# Словарь лейблов
label_to_classname = \
    {0: "Перевод по должности",
     1: "Перевод по подразделению с сохранением должности",
     2: "Перевод и по подразделению и по должности",
     3: "Перевод с сохранением должности и подразделения"
     }
