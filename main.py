import streamlit as st
import pandas as pd

from streamlit import session_state as sst
from Data_Formation import get_train_dataset, get_test_dataset, get_dataframe
from Optuna_Optimization import optuna_optimization
from models import *
from constants import kwargs
from misc_functions import *
from Data_Analysis import create_heatmap_streamlit
from datetime import datetime

np.random.seed(seed)   #Устанавливаем сид (sklearn использует сид от numpy)
tf.random.set_seed(seed) #Устанавливаем сид для нейросетей

if __name__ == "__main__":
    with open("style.css") as f: #Читаем style.css файл
        st.markdown(f.read(), unsafe_allow_html=True) #Это предает стиль всему приложению

    st.title('Тестовое задание РАНХиГС')
    st.markdown(
        """
        ### Прогноз перевода в должности сотрудников компании
        """
    )
    st.image('title_image.jpg')

    st.write("Данное приложение дает возможность создать собственный алгоритм для прогноза будущих карьерных перспектив"
             "сотрудника компании. Вы можете выбрать любой алгоритм из списка предоставленных, а также, при желании, "
             "Вы сможете самостоятельно подбирать гиперпараметры для выбранного Вами алгоритма.")


    st.write("Чтобы обучить алгоритм, необходимо загрузить файл с id сотрудников и их карьерных данных в обучающую"
             "выборку, и также загрузить аналогичный файл в качестве тестовой выборки. Затем, нужно нажать на кнопку "
             "'Обучить'.")

    train_file = st.file_uploader("Загрузите обучающую выборку", key='upload_train_dataset', type=["csv"])
    st.markdown("---")

    possible_algorithms = ['XGBoost', 'Нейросеть', 'Гауссовский классификатор', 'SVM', 'Дерево Решений', 'Случайный Лес']
    selected_algorithm = st.selectbox(
        "Выберите алгоритм для модели: \n (примечание: модель нужно обучать заново после изменения набора столбцов)",
        possible_algorithms, key='selected_algorithm')

    st.write("Настройте гиперпараметры для модели:")
    if selected_algorithm == possible_algorithms[0]:  #XGBoost
        kwargs['n_estimators'] = st.slider("Число деревьев решений", min_value=1, max_value=200, value=100)
        kwargs['GB_max_depth'] = st.slider("Макс. глубина деревьев", min_value=1, max_value=5, value=3)
        learning_rate = st.slider('Скорость обучения', min_value=0.0, max_value=1.0, value=0.5)
        kwargs['GB_learning_rate'] = 10 ** (2 * learning_rate - 2)
        kwargs['min_samples_split'] = st.slider('Минимальное кол-во данных для сплита', min_value=2, max_value=5, value=2)

    elif selected_algorithm == possible_algorithms[1]: #Нейросеть
        num_neurons = []
        kwargs['hidden_layers'] = st.slider("Число скрытых слоев", min_value=0, max_value=5, value=2)
        for i in range(kwargs['hidden_layers']):
            num_neurons.append(st.slider(f'Число нейронов в {i + 1}-m слое', min_value=4, max_value=64, value=32))
        kwargs['neural_network_hidden_neurons'] = num_neurons

        kwargs['num_epochs'] = st.slider("Эпохи обучения", min_value=1, max_value=10, value=5)
        learning_rate = st.slider('Скорость обучения', min_value=0.0, max_value=1.0, value=0.5)
        kwargs['NN_learning_rate'] = 10 ** (4 * learning_rate - 5)
        kwargs['batch_size'] = st.slider('Размер пакета', min_value=16, max_value=256, value=32)
        kwargs['use_class_weights'] = st.checkbox('Приоритизировать более редкие классы в обучении',
                                                  key='class_weights_checkbox')

    elif selected_algorithm == possible_algorithms[2]:  #Гауссовский классификатор
        kwargs['max_iter_predict'] = st.slider("Максимальное число итераций", min_value=10, max_value=500, value=100)
        kwargs['warm_start'] = st.checkbox("Использовать 'тёплое' начало (игнорируйте если не знаете что это такое")

    elif selected_algorithm == possible_algorithms[3]:  #SVM
        kwargs['C'] = st.slider("Регуляризация", min_value=0.1, max_value=5.0, value=1.0)
        kwargs['use_class_weights'] = st.checkbox('Приоритизировать более редкие классы в обучении',
                                                  key='class_weights_checkbox')
        kwargs['kernel'] = st.selectbox("Ядро", ['poly', 'rbf', 'sigmoid'])

    elif selected_algorithm == possible_algorithms[4]:  #Дерево Решений
        max_depth = st.slider("Максимальная глубина дерева (0 = нет ограничений)", min_value=0, max_value=10, value=0)
        kwargs['Tree_max_depth'] = max_depth if max_depth > 0 else None
        kwargs['min_samples_split'] = st.slider("Минимальное число сэмплов для деления дерева", min_value=2,
                                                max_value=5, value=2)
        kwargs['criterion'] = st.selectbox("Функция ошибки", ['gini', 'entropy', 'log_loss'])

    elif selected_algorithm == possible_algorithms[5]:  #Случайный Лес
        kwargs['n_estimators'] = st.slider("Количество деревьев решений", min_value=1, max_value=200, value=100, step=1)
        max_depth = st.slider("Максимальная глубина дерева (0 = нет ограничений)", min_value=0, max_value=10, value=0)
        kwargs['Tree_max_depth'] = max_depth if max_depth > 0 else None
        kwargs['min_samples_split'] = st.slider("Минимальное число сэмплов для деления дерева", min_value=2,
                                                max_value=5, value=2)

    with st.expander("Использовать автоматический подбор гиперпараметров"):
        st.write("""
            TPE (Tree-Structured Parzen Estimator) - алгоритм, который основан на Байесовских методах, и используется для подбора гиперпараметров.
            Этот алгоритм способен автоматический перебирать комбинации гиперпараметров исользуя метод "проба и ошибок".
            
            Если Вы установите этот флажок, программа сама попытается подобрать оптимальный набор гиперпараметров.
            В таком случае, все ранее Вами указанные значения гиперпараметров будут игнорированы.
        """)

        #Возможность использовать Оптуну для подбора гипепараметров
        use_optuna = st.checkbox("Использовать TPE для автоматического подбора гиперпараметров", key='use_optuna')

        if use_optuna:
            optuna_epochs = st.slider("Кол-во эпох для оптимизации алгоритма с помощью TPE (0 = не использовать автоматическую оптимизацию)",
                              min_value=1, max_value=1000, value=50, key='optuna_epochs')

            st.markdown("""
                Рекомендуется использовать не меньше 50 эпох для оптимизации.
                
                **ПРИМЕЧАНИЕ: Программе может потребоваться много времени для выполнения большого количества эпох.**""")

            st.markdown("""
                Также учтите, что TPE только подбирает параметры для выбранного Вами алгоритма модели. 
                Сам алгоритм нужно выбирать выше, в ручную.""")

    # Используем session_state чтобы запомнить нажатье различных кнопок
    if 'train_button_clicked' not in sst:
        sst.train_button_clicked = False #Нужно, чтобы можно было нажать на кнопку "тестировать" после обучения
    if 'model_trained' not in sst:
        sst.model_trained = False  #Нужно, чтобы не обучать модель заново при нажатии кнопки "тестировать"
    if 'algorithm' not in sst:
        sst.algorithm = None       #Нужно, чтобы запомнить алгоритм и его веса после обучения перед тестированием

    #Кнопка для обучения алгоритма
    train_button = st.button(label='Обучить', key='train_button',
                             disabled=True if not train_file else False,
                             on_click=train_buttons_callback)

    #Словарь который переводит введеную строку в класс заданного алгоритма
    str_to_algorithm = {'Нейросеть': NeuralNetwork, 'XGBoost': GradientBoostingAlgorithm, 'SVM': SVMAlgorithm,
                        'Дерево Решений': DecisionTreeAlgorithm, 'Случайный Лес': RandomForestAlgorithm,
                        'Гауссовский классификатор': GaussianAlgorithm}

    #Окно с возможностью выбора столбцов вылезает при загрузки обучающей выборки
    if train_file is not None:
        label_columns = ['type'] #Список всех столбцов, которые подаются в y_train/y_test (всего один такой столбец)
        train_dataframe, tasks_encoder = get_dataframe(train_file) #Извлекаем все возможные столбцы из датасета

        #Составляем список всех столбцов, которые можно выбирать или не выбирать.
        #В этот список не попадают столбцы из label_columns
        possible_columns = [i for i in train_dataframe.columns.values if i not in label_columns]
        if 'dataset_columns' not in sst:
            sst.dataset_columns = possible_columns + label_columns
        selected_columns = st.sidebar.multiselect("Выберите столбцы для обучения нейросети",
                       key='select_dataset_columns', options=possible_columns, default=possible_columns,
                       )

        select_columns_button = st.sidebar.button("Подтвердить выбор столбцов", key='select_columns_button',
                                                  help="Нажмите на кнопку чтобы выбрать указанные столбцы")
        st.sidebar.markdown('---')
        if select_columns_button:
            sst.dataset_columns = selected_columns + label_columns #Добавляем label_columns обратно в датасет

        #Доля валидационной выборки от общей обучающей+валидационной выборки
        val_percentage = st.sidebar.slider("Доля валидационной выборки от общего датасета",
                                   min_value=0.0, max_value=0.9, value=0.2)

        #Кнопка для формирования и вывода матрицы корреляции
        draw_heatmap = st.sidebar.button(label="Вывести матрицу корреляций", key='heatmap_button',
                                         help="Нажмите на кнопку чтобы вывести матрицу корреляции всех выбранных столбцов")


        if draw_heatmap:
            create_heatmap_streamlit(train_dataframe[sst.dataset_columns]) #Функция для вывода матрицы корреляции

    #При нажатии на кпонку "Обучение"
    if sst.train_button_clicked:
        if not sst.model_trained:
            sst.algorithm = str_to_algorithm[selected_algorithm](**kwargs)
        train_dataframe = train_dataframe[sst.dataset_columns] #Обрабатываем датасет и формируем x, y для val и train
        x_train, x_val, y_train, y_val, encoders, scaler = get_train_dataset(
            train_dataframe, tasks_encoder, val_percentage=val_percentage,  # Загружаем обработанный датасет
            scale_data=sst.algorithm.requires_normalization, onehot_encode=sst.algorithm.requires_OHE
        )

        #Если модель ранее не была обучена:
        # (условие нужно, чтобы модель не обучалась заново при нажатии нерелевантных кнопок)
        if not sst.model_trained:
            if use_optuna:  #Оптимизация кода оптуной
                kwargs = optuna_optimization(x_train, y_train, x_val, y_val, sst.algorithm, optuna_epochs)
                #Блок кода, который обнавляет значения в kwargs в зависимости от полученного результата из Оптуны
                if isinstance(sst.algorithm, SVMAlgorithm):
                    kwargs['class_weight'] = calculate_class_weights(y_train) if kwargs['use_class_weights'] else None
                elif isinstance(sst.algorithm, NeuralNetwork):
                    kwargs['neural_network_hidden_neurons'] = [kwargs[f'layer_{i + 1}_size'] for i in
                                                               range(kwargs['hidden_layers'])]
            sst.algorithm.train(x_train, y_train)  # Задаем параметры алгоритма

            # Валидируем на обучающей выборке
            sst.train_score = sst.algorithm.validate(x_train, y_train, subset_type='train')
            # Валидируем на валидационной выборке (только если у нас есть валидационная выборка)
            sst.val_score = sst.algorithm.validate(x_val, y_val) if val_percentage > 0 else None

            # Устанавливаем флажок, который сообщает что модель уже обучена (чтобы не обучать дважды)
            sst.model_trained = True

        #Выводим точности на обучающей и валидационной выборки в стримлит
        st.info(f"Точность модели на обучающей выборке: {round(100*sst.train_score, 3)}%")
        if val_percentage > 0 and sst.val_score is not None:
            st.info(f"Точность модели на валидационной выборке: {round(100*sst.val_score, 3)}%")

        #Создаем загрузщик для тестового файла
        test_file = st.file_uploader("Загрузите тестовую выборку", key='upload_test_dataset', type=["csv"],
                                     on_change=upload_test_dataset)
        st.markdown("---")

        #Кнопка для теста
        test_button = st.button(label='Тестировать', key='test_button',
                                disabled=True if not test_file else False,
                                on_click=test_buttons_callback)

        if test_button: #Когда нажимаем на тестовую кнопку:
            x_test, y_test, data_indices = get_test_dataset(  #Собираем тестовый датасет используя кодировщики из обучающей
                test_file, sst.dataset_columns, encoders, scaler,
                scale_data=sst.algorithm.requires_normalization, onehot_encode=sst.algorithm.requires_OHE
            )

            test_score, predictions = sst.algorithm.test(x_test, y_test) #Тестируем
            # Выводим точность на тестовой выборке в стримлит
            st.info(f"Точность модели на тестовой выборке: {round(100*test_score, 3)}%")

            predictions = [label_to_classname[i] for i in predictions.flatten()]
            predictions_dataframe = pd.DataFrame(data={'ID Сотрудника': data_indices, 'Предсказание': predictions})

            st.write("Предсказания модели: ")
            st.write(predictions_dataframe)
    st.caption("Автор: Никита Серов")



#Стоит отметить, что данную задачу вероятно можно решить как задачу регрессии, и в таком случае можно использовать метрику MSE или MAE.
#Для этого, необходимо выставить все категории в один ряд - сортировать их с условного "худшего" до "лучшего"
#(Например - если 0 это точное понижение в должности, 3 это точное повышение, а 1 и 2 - где-то между повышением и понижением)