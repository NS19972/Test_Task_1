import streamlit as st

from Data_Formation import get_train_dataset, get_test_dataset
from Optuna_Optimization import optuna_optimization
from models import *
from constants import *

np.random.seed(seed)   #Устанавливаем сид (sklearn использует сид от numpy)
tf.random.set_seed(seed) #Устанавливаем сид для нейросетей

if __name__ == "__main__":
    st.markdown(
        """
        #Прогноз перевода в должности сотрудников
        """
    )
    st.caption("Автор: Никита Серов")

    train_file = st.file_uploader("Загрузить обучающую выборку", key='upload_train_dataset', type=["csv"])
    test_file = st.file_uploader("Загрузить тестовую выборку", key='upload_test_dataset', type=["csv"])

    possible_algorithms = st.selectbox("Выберите алгоритм для модели: ",
                                ['Нейросеть', 'XGBoost', 'Гауссовский классификатор', 'SVM', 'Дерево Решений'],
                                       key='algorithm_selection')

    optuna_epochs = st.slider("Кол-во эпох для оптимизации алгоритма Оптуной (оставьте 0 чтобы не использовать Оптуну)", min_value=0, max_value=1000, value=0, key='optuna_box')

    train_button = st.button(label='Обучить', key='train_button')

    str_to_algorithm = {'Нейросеть': NeuralNetwork, 'XGBoost': GradientBoostingAlgorithm, 'SVM': SVMAlgorithm,
                        'Дерево Решений': DecisionTreeAlgorithm, 'Гауссовский классификатор': GaussianAlgorithm}

    if train_button:
        x_train, x_val, y_train, y_val, encoders, scaler = get_train_dataset(train_file)  # Загружаем обработанный датасет
        algorithm = str_to_algorithm[possible_algorithms]()
        if optuna_epochs:
            optuna_optimization(algorithm, optuna_epochs)
        algorithm.train(x_train, y_train)  # Задаем параметры алгоритма
        val_score = algorithm.validate(x_val, y_val)
        st.text(f"Точность модели на валидационной выборке: {val_score}")

        x_test, y_test = get_test_dataset(test_file, encoders, scaler)
        test_score = algorithm.test(x_test, y_test)
        st.text(f"Точность модели на тестовой выборке: {test_score}")



#Стоит отметить, что данную задачу вероятно можно решить как задачу регрессии, и в таком случае можно использовать метрику MSE или MAE.
#Для этого, необходимо выставить все категории в один ряд - сортировать их с условного "худшего" до "лучшего"
#(Например - если 0 это точное понижение в должности, 3 это точное повышение, а 1 и 2 - где-то между повышением и понижением)