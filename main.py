import streamlit as st

from Data_Formation import get_train_dataset, get_test_dataset
from Optuna_Optimization import optuna_optimization
from models import *
from constants import kwargs

np.random.seed(seed)   #Устанавливаем сид (sklearn использует сид от numpy)
tf.random.set_seed(seed) #Устанавливаем сид для нейросетей

if __name__ == "__main__":
    st.title('Тестовое задание РАНХиГС')
    st.markdown(
        """
        #Прогноз перевода в должности сотрудников
        """
    )
    st.caption("Автор: Никита Серов")

    train_file = st.file_uploader("Загрузить обучающую выборку", key='upload_train_dataset', type=["csv"])
    test_file = st.file_uploader("Загрузить тестовую выборку", key='upload_test_dataset', type=["csv"])

    selected_algorithm = st.selectbox(
        "Выберите алгоритм для модели: ",
        ['Нейросеть', 'XGBoost', 'Гауссовский классификатор', 'SVM', 'Дерево Решений', 'Случайный Лес'],
        key='algorithm_selection')

    with st.expander("Использовать автоматический подбор гиперпараметров"):
        st.write("""
            TPE (Tree-Structured Parzen Estimator) - алгоритм, который основан на Байесовских методах, и используется для подбора гиперпараметров.
            Этот алгоритм способен автоматический перебирать комбинации гиперпараметров исользуя метод "проба и ошибок".
            
            Если Вы установите этот флажок, программа сама попытается подобрать оптимальный набор гиперпараметров.
            В таком случае, все ранее Вами указанные значения гиперпараметров будут игнорированы.
            
        """)
        use_optuna = st.checkbox("Использовать TPE для автоматического подбора гиперпараметров", key='use_optuna')

        if use_optuna:
            optuna_epochs = st.slider("Кол-во эпох для оптимизации алгоритма с помощью TPE (0 = не использовать автоматическую оптимизацию)",
                              min_value=1, max_value=1000, value=100, key='optuna_epochs')

            st.markdown("""
                Рекомендуется использовать не меньше 100 эпох для оптимизации.
                
                **ПРИМЕЧАНИЕ: Программе может потребоваться много времени для выполнения большого количества эпох.**
                """)


    train_button = st.button(label='Обучить', key='train_button')

    str_to_algorithm = {'Нейросеть': NeuralNetwork, 'XGBoost': GradientBoostingAlgorithm, 'SVM': SVMAlgorithm,
                        'Дерево Решений': DecisionTreeAlgorithm, 'Случайный Лес': RandomForestAlgorithm,
                        'Гауссовский классификатор': GaussianAlgorithm}

    scale_data = True if selected_algorithm == 'Нейросеть' else False
    onehot_encode = True if selected_algorithm in ['Нейросеть', 'SVM'] else False

    if train_button:
        x_train, x_val, y_train, y_val, encoders, scaler = get_train_dataset(
            train_file, scale_data=scale_data, onehot_encode=onehot_encode  # Загружаем обработанный датасет
        )

        if use_optuna:
            kwargs = optuna_optimization(x_train, y_train, x_val, y_val, selected_algorithm, optuna_epochs)
        algorithm = str_to_algorithm[selected_algorithm](**kwargs)
        algorithm.train(x_train, y_train)  # Задаем параметры алгоритма
        val_score = algorithm.validate(x_val, y_val)
        st.text(f"Точность модели на валидационной выборке: {val_score}")

        x_test, y_test = get_test_dataset(
            test_file, encoders, scaler, scale_data=scale_data, onehot_encode=onehot_encode
        )

        test_score = algorithm.test(x_test, y_test)
        st.text(f"Точность модели на тестовой выборке: {test_score}")



#Стоит отметить, что данную задачу вероятно можно решить как задачу регрессии, и в таком случае можно использовать метрику MSE или MAE.
#Для этого, необходимо выставить все категории в один ряд - сортировать их с условного "худшего" до "лучшего"
#(Например - если 0 это точное понижение в должности, 3 это точное повышение, а 1 и 2 - где-то между повышением и понижением)