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
        ### Прогноз перевода в должности сотрудников компании
        """
    )
    st.image('title_image.jpg')

    train_file = st.file_uploader("Загрузите обучающую выборку", key='upload_train_dataset', type=["csv"])
    st.markdown("---")
    test_file = st.file_uploader("Загрузите тестовую выборку", key='upload_test_dataset', type=["csv"])
    st.markdown("---")

    possible_algorithms = ['XGBoost', 'Нейросеть', 'Гауссовский классификатор', 'SVM', 'Дерево Решений', 'Случайный Лес']
    selected_algorithm = st.selectbox(
        "Выберите алгоритм для модели: ", possible_algorithms, key='selected_algorithm')

    st.write("Настройте гиперпараметры для модели:")
    if selected_algorithm == possible_algorithms[0]:
        kwargs['n_estimators'] = st.slider("Количество деревьев решений", min_value=1, max_value=200, value=100)
        kwargs['GB_max_depth'] = st.slider("Макс. глубина деревьев", min_value=1, max_value=5, value=3)
        learning_rate = st.slider('Скорость обучения', min_value=0.0, max_value=1.0, value=0.5)
        kwargs['GB_learning_rate'] = 10 ** (2*learning_rate - 2)
        kwargs['min_samples_split'] = st.slider('Минимальное кол-во для сплита', min_value=2, max_value=5, value=2)

    elif selected_algorithm == possible_algorithms[1]:
        num_neurons = []
        kwargs['hidden_layers'] = st.slider("Количество скрытых слоев", min_value=0, max_value=5, value=2)
        for i in range(kwargs['hidden_layers']):
            num_neurons.append(st.slider(f'Количество нейронов в {i+1}-m слое', min_value=1, max_value=256, value=32))
        kwargs['neural_network_hidden_neurons'] = num_neurons

        kwargs['num_epochs'] = st.slider("Эпохи обучения", min_value=1, max_value=10, value=5)
        learning_rate = st.slider('Скорость обучения', min_value = 0.0, max_value=1.0, value = 0.5)
        kwargs['NN_learning_rate'] = 10 ** (4 * learning_rate - 5)
        kwargs['batch_size'] = st.slider('Размер батча', min_value = 8, max_value=256, value = 32, step=4)

    elif selected_algorithm == possible_algorithms[2]:
        pass

    elif selected_algorithm == possible_algorithms[3]:
        pass

    elif selected_algorithm == possible_algorithms[4]:
        pass

    elif selected_algorithm == possible_algorithms[5]:
        kwargs['n_estimators'] = st.slider("Количество деревьев решений", min_value=1, max_value=200, value=100, step=1)



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
    if train_button and train_file is not None and test_file is not None:
        x_train, x_val, y_train, y_val, encoders, scaler = get_train_dataset(
            train_file, scale_data=scale_data, onehot_encode=onehot_encode  # Загружаем обработанный датасет
        )

        if use_optuna:
            kwargs = optuna_optimization(x_train, y_train, x_val, y_val, selected_algorithm, optuna_epochs)
        algorithm = str_to_algorithm[selected_algorithm](**kwargs)
        algorithm.train(x_train, y_train)  # Задаем параметры алгоритма
        val_score = algorithm.validate(x_val, y_val)
        st.info(f"Точность модели на валидационной выборке: {round(100*val_score, 3)}%")

        x_test, y_test = get_test_dataset(
            test_file, encoders, scaler, scale_data=scale_data, onehot_encode=onehot_encode
        )

        test_score = algorithm.test(x_test, y_test)
        st.info(f"Точность модели на тестовой выборке: {round(100*test_score, 3)}%")
    st.caption("Автор: Никита Серов")



#Стоит отметить, что данную задачу вероятно можно решить как задачу регрессии, и в таком случае можно использовать метрику MSE или MAE.
#Для этого, необходимо выставить все категории в один ряд - сортировать их с условного "худшего" до "лучшего"
#(Например - если 0 это точное понижение в должности, 3 это точное повышение, а 1 и 2 - где-то между повышением и понижением)