import optuna
import tensorflow as tf
import argparse
from models import *
from misc_functions import *


def optuna_optimization(x_train, y_train, x_val, y_val, selected_algorithm, optimization_epochs):

    def objective_function(trial):
        kwargs = {}
        if selected_algorithm.lower() == 'нейросеть':
            kwargs['hidden_layers'] = trial.suggest_int("hidden_layers", 0, 4)
            layer_1_size = trial.suggest_int("layer_1_size", 16, 64, log=True) if kwargs['hidden_layers'] >= 1 else 0
            layer_2_size = trial.suggest_int("layer_2_size", 16, 64, log=True) if kwargs['hidden_layers'] >= 2 else 0
            layer_3_size = trial.suggest_int("layer_3_size", 16, 64, log=True) if kwargs['hidden_layers'] >= 3 else 0
            layer_4_size = trial.suggest_int("layer_4_size", 16, 64, log=True) if kwargs['hidden_layers'] >= 4 else 0
            kwargs['neural_network_hidden_neurons'] = [layer_1_size, layer_2_size, layer_3_size, layer_4_size]
            kwargs['num_epochs'] = trial.suggest_int('num_epochs', 1, 10)
            kwargs['NN_learning_rate'] = trial.suggest_float('NN_learning_rate', 1e-5, 1e-2, log=True)
            kwargs['batch_size'] = trial.suggest_int('batch_size', 16, 256)
            kwargs['use_class_weights'] = trial.suggest_categorical('use_class_weights', [True, False])

            algorithm = NeuralNetwork(**kwargs)

        elif selected_algorithm.lower() == 'xgboost':
            kwargs['GB_learning_rate'] = trial.suggest_float('GB_learning_rate', 1e-2, 5e-1, log=True)
            kwargs['n_estimators'] = trial.suggest_int('N_estimators', 10, 500, log=True)
            kwargs['max_depth'] = trial.suggest_int('max_depth', 1, 5)
            kwargs['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 4)

            algorithm = GradientBoostingAlgorithm(**kwargs)

        elif selected_algorithm.lower() == 'случайный лес':
            kwargs['n_estimators'] = trial.suggest_int('N_estimators', 10, 500, log=True)
            use_max_depth = trial.suggest_categorical('use_max_depth', [True, False])
            kwargs['max_depth'] = trial.suggest_int('max_depth', 1, 5) if use_max_depth else None
            kwargs['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 4)

            algorithm = RandomForestAlgorithm(**kwargs)

        elif selected_algorithm.lower() == 'дерево решений':
            use_max_depth = trial.suggest_categorical('use_max_depth', [True, False])
            kwargs['max_depth'] = trial.suggest_int('max_depth', 1, 5) if use_max_depth else None
            kwargs['criterion'] = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
            kwargs['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 4)

            algorithm = DecisionTreeAlgorithm(**kwargs)

        elif selected_algorithm.lower() == 'svm':
            kwargs['C'] = trial.suggest_float('C', 0.5, 2)
            kwargs['use_class_weights'] = trial.suggest_categorical('use_class_weights', [True, False])
            kwargs['kernel'] = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'])

            algorithm = SVMAlgorithm(**kwargs)

        elif selected_algorithm.lower() == 'гауссовский классификатор':
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