import optuna
import tensorflow as tf
import argparse
from models import *
from misc_functions import *

np.random.seed(seed)  # Устанавливаем сид (sklearn использует сид от numpy)
tf.random.set_seed(seed)  # Устанавливаем сид для нейросетей

def optuna_optimization(x_train, y_train, x_val, y_val, selected_algorithm, optimization_epochs):
    def objective_function(trial):
        kwargs = {}
        if isinstance(selected_algorithm, NeuralNetwork): #Подбор гиперпараметров для нейросети
            kwargs['hidden_layers'] = trial.suggest_int("hidden_layers", 0, 4) #Задаем количество скрытых слоев
            #Задаем количество нейронов в слоях
            layer_1_size = trial.suggest_int("layer_1_size", 4, 64, log=True) if kwargs['hidden_layers'] >= 1 else 0
            layer_2_size = trial.suggest_int("layer_2_size", 4, 64, log=True) if kwargs['hidden_layers'] >= 2 else 0
            layer_3_size = trial.suggest_int("layer_3_size", 4, 64, log=True) if kwargs['hidden_layers'] >= 3 else 0
            layer_4_size = trial.suggest_int("layer_4_size", 4, 64, log=True) if kwargs['hidden_layers'] >= 4 else 0
            # Складываем все слои в один список
            kwargs['neural_network_hidden_neurons'] = [layer_1_size, layer_2_size, layer_3_size, layer_4_size]
            kwargs['num_epochs'] = trial.suggest_int('num_epochs', 1, 10) #Количество эпох обучения
            kwargs['NN_learning_rate'] = trial.suggest_float('NN_learning_rate', 1e-5, 1e-2, log=True) #Скорость обучения
            kwargs['batch_size'] = trial.suggest_int('batch_size', 16, 256) #Размер батча
            kwargs['use_class_weights'] = trial.suggest_categorical('use_class_weights', [True, False]) #Использовать взвешанные классы?

            algorithm = NeuralNetwork(**kwargs) #Создаем новый объект нейронной сети

        elif isinstance(selected_algorithm, GradientBoostingAlgorithm):
            kwargs['GB_learning_rate'] = trial.suggest_float('GB_learning_rate', 1e-2, 5e-1, log=True)
            kwargs['n_estimators'] = trial.suggest_int('n_estimators', 10, 500, log=True)
            kwargs['GB_max_depth'] = trial.suggest_int('GB_max_depth', 1, 5)
            kwargs['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 4)

            algorithm = GradientBoostingAlgorithm(**kwargs)

        elif isinstance(selected_algorithm, RandomForestAlgorithm):
            kwargs['n_estimators'] = trial.suggest_int('n_estimators', 10, 500, log=True)
            use_max_depth = trial.suggest_categorical('use_max_depth', [True, False])
            kwargs['Tree_max_depth'] = trial.suggest_int('Tree_max_depth', 1, 5) if use_max_depth else None
            kwargs['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 4)

            algorithm = RandomForestAlgorithm(**kwargs)

        elif isinstance(selected_algorithm, DecisionTreeAlgorithm):
            use_max_depth = trial.suggest_categorical('use_max_depth', [True, False])
            kwargs['Tree_max_depth'] = trial.suggest_int('Tree_max_depth', 1, 5) if use_max_depth else None
            kwargs['criterion'] = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
            kwargs['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 4)

            algorithm = DecisionTreeAlgorithm(**kwargs)

        elif isinstance(selected_algorithm, SVMAlgorithm):
            kwargs['C'] = trial.suggest_float('C', 0.5, 2)
            kwargs['use_class_weights'] = trial.suggest_categorical('use_class_weights', [True, False])
            kwargs['class_weight'] = calculate_class_weights(y_train) if kwargs['use_class_weights'] else None
            kwargs['kernel'] = trial.suggest_categorical('kernel', ['poly', 'rbf', 'sigmoid'])

            algorithm = SVMAlgorithm(**kwargs)

        elif isinstance(selected_algorithm, GaussianAlgorithm):
            kwargs['max_iter_predict'] = trial.suggest_int('max_iter_predict', 10, 500, log=True)
            kwargs['warm_start'] = trial.suggest_categorical('warm_start', [True, False])

            algorithm = GaussianAlgorithm(**kwargs)

        else:
            raise KeyError

        algorithm.train(x_train, y_train)  # Задаем параметры алгоритма
        val_score = algorithm.validate(x_val, y_val) # Оптимизируем патамерты только на валидационной выборке
        return val_score


    study = optuna.create_study(direction="maximize", storage=optuna.storages.RDBStorage(
        url=f'sqlite:///optimization.db', engine_kwargs={"connect_args": {"timeout": 100}})
                                )
    study.optimize(objective_function, n_trials=optimization_epochs)
    return study.best_params