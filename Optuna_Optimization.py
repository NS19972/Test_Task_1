import optuna
import tensorflow as tf
from models import *
from misc_functions import *

np.random.seed(seed)  # Устанавливаем сид (sklearn использует сид от numpy)
tf.random.set_seed(seed)  # Устанавливаем сид для нейросетей

def optuna_optimization(x_train, y_train, x_val, y_val, selected_algorithm, optimization_epochs, kwargs):
    def objective_function(trial):
        # Подбор гиперпараметров для нейросети
        if isinstance(selected_algorithm, NeuralNetwork):
            kwargs.update({'hidden_layers': trial.suggest_int("hidden_layers", 0, 4)})  # Задаем количество скрытых слоев
            # Задаем количество нейронов в слоях
            layer_1_size = trial.suggest_int("layer_1_size", 4, 64, log=True) if kwargs['hidden_layers'] >= 1 else 0
            layer_2_size = trial.suggest_int("layer_2_size", 4, 64, log=True) if kwargs['hidden_layers'] >= 2 else 0
            layer_3_size = trial.suggest_int("layer_3_size", 4, 64, log=True) if kwargs['hidden_layers'] >= 3 else 0
            layer_4_size = trial.suggest_int("layer_4_size", 4, 64, log=True) if kwargs['hidden_layers'] >= 4 else 0
            # Складываем все слои в один список
            kwargs.update({'neural_network_hidden_neurons': [layer_1_size, layer_2_size, layer_3_size, layer_4_size]})
            kwargs.update({'num_epochs': trial.suggest_int('num_epochs', 1, 10)})  # Количество эпох обучения
            kwargs.update({'NN_learning_rate': trial.suggest_float('NN_learning_rate', 1e-5, 1e-2, log=True)})  # Скорость обучения
            kwargs.update({'batch_size': trial.suggest_int('batch_size', 16, 256)})  # Размер батча
            kwargs.update({'use_class_weights': trial.suggest_categorical('use_class_weights', [True, False])})  # Использовать взвешенные классы?

            algorithm = NeuralNetwork(**kwargs)  # Создаем новый объект нейронной сети

        # Подбор гиперпараметров для градиентного бустинга
        elif isinstance(selected_algorithm, GradientBoostingAlgorithm):
            kwargs.update({'GB_learning_rate': trial.suggest_float('GB_learning_rate', 1e-2, 5e-1, log=True)})
            kwargs.update({'n_estimators': trial.suggest_int('n_estimators', 10, 500, log=True)})
            kwargs.update({'GB_max_depth': trial.suggest_int('GB_max_depth', 1, 5)})
            kwargs.update({'min_samples_split': trial.suggest_int('min_samples_split', 2, 4)})

            algorithm = GradientBoostingAlgorithm(**kwargs)

        # Подбор гиперпараметров для случайного леса
        elif isinstance(selected_algorithm, RandomForestAlgorithm):
            kwargs.update({'n_estimators': trial.suggest_int('n_estimators', 10, 500, log=True)})
            use_max_depth = trial.suggest_categorical('use_max_depth', [True, False])
            kwargs.update({'Tree_max_depth': trial.suggest_int('Tree_max_depth', 1, 5) if use_max_depth else None})
            kwargs.update({'min_samples_split': trial.suggest_int('min_samples_split', 2, 4)})
            kwargs.update({'use_class_weights': trial.suggest_categorical('use_class_weights', [True, False])})  # Использовать взвешенные классы?

            algorithm = RandomForestAlgorithm(**kwargs)

        # Подбор гиперпараметров для дерева решений
        elif isinstance(selected_algorithm, DecisionTreeAlgorithm):
            use_max_depth = trial.suggest_categorical('use_max_depth', [True, False])
            kwargs.update({'Tree_max_depth': trial.suggest_int('Tree_max_depth', 1, 5) if use_max_depth else None})
            kwargs.update({'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])})
            kwargs.update({'min_samples_split': trial.suggest_int('min_samples_split', 2, 4)})
            kwargs.update({'use_class_weights': trial.suggest_categorical('use_class_weights', [True, False])})  # Использовать взвешенные классы?

            algorithm = DecisionTreeAlgorithm(**kwargs)

        # Подбор гиперпараметров для метода опорных векторов
        elif isinstance(selected_algorithm, SVMAlgorithm):
            kwargs.update({'C': trial.suggest_float('C', 0.5, 2)})
            kwargs.update({'use_class_weights': trial.suggest_categorical('use_class_weights', [True, False])})
            kwargs.update({'class_weight': calculate_class_weights(y_train) if kwargs['use_class_weights'] else None})
            kwargs.update({'kernel': trial.suggest_categorical('kernel', ['poly', 'rbf', 'sigmoid'])})

            algorithm = SVMAlgorithm(**kwargs)

        # Подбор гиперпараметров для гауссовского алгоритма
        elif isinstance(selected_algorithm, GaussianAlgorithm):
            kwargs.update({'max_iter_predict': trial.suggest_int('max_iter_predict', 10, 500, log=True)})
            kwargs.update({'warm_start': trial.suggest_categorical('warm_start', [True, False])})

            algorithm = GaussianAlgorithm(**kwargs)

        else:
            raise KeyError

        algorithm.train(x_train, y_train)  # Задаем параметры алгоритма
        val_score = algorithm.validate(x_val, y_val)  # Оптимизируем параметры только на валидационной выборке
        return val_score

    percent_progress = 0.0  # Значение прогресс-бара инициализируется как 0
    my_bar = st.progress(percent_progress)  # Создаем прогресс-бар

    # Функция, которая выводит прогресс-бар
    def optuna_progress_bar(study, frozen_trial):
        # Объявляем переменные как нелокальные, по сколько мы не можем их применять на вход в функцию
        nonlocal my_bar, percent_progress
        # Обновляем значение прогресс-бара (min нужен для избежания дурацких ошибок на последней эпохе)
        percent_progress = min(percent_progress + 1/optimization_epochs, 1.0)
        my_bar.progress(percent_progress)

    # Создаем объект исследования, с которым будем оптимизировать функцию
    study = optuna.create_study(direction="maximize", storage=optuna.storages.RDBStorage(
        url=f'sqlite:///optimization.db', engine_kwargs={"connect_args": {"timeout": 100}}))

    # Оптимизируем гиперпараметры алгоритма используя оптуну
    study.optimize(objective_function, n_trials=optimization_epochs, callbacks=[optuna_progress_bar])
    return study.best_params
