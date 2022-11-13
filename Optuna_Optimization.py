import optuna
import tensorflow as tf
from sklearn.ensemble import *
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score
from sklearn.svm import *


def optuna_optimization(algorithm, optimization_epochs):
    def objective_function():
        if isinstance(algorithm, tf.keras.models.Model) or isinstance(algorithm, tf.keras.models.Sequential):
            pass
        elif isinstance(algorithm, GradientBoostingClassifier):
            pass
        elif isinstance(algorithm, DecisionTreeClassifier):
            pass
        elif isinstance(algorithm, SVC):
            pass
        elif isinstance(algorithm, GaussianProcessClassifier):
            pass

    study = optuna.create_study(direction="maximize", storage=optuna.storages.RDBStorage(url=f'sqlite:///optimization.db', engine_kwargs={"connect_args": {"timeout": 100}}))
    study.optimize(objective_function, n_trials=optimization_epochs)