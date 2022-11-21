import streamlit as st
import pandas as pd

from streamlit import session_state as sst
from Data_Formation import get_train_dataset, get_test_dataset, get_dataframe
from Optuna_Optimization import optuna_optimization
from models import *
from constants import kwargs
from misc_functions import *
from Data_Analysis import *
from datetime import datetime

np.random.seed(seed)   # Устанавливаем сид (sklearn использует сид от numpy)
tf.random.set_seed(seed)  # Устанавливаем сид для нейросетей

if __name__ == "__main__":
    with open("style.css") as f:  # Читаем style.css файл
        st.markdown(f.read(), unsafe_allow_html=True)  # Это предает стиль всему приложению

    st.title('Тестовое задание РАНХиГС')
    st.markdown(
        """
        ### Прогноз перевода в должности сотрудников компании
        """
    )
    st.image('title_image.jpg')

    st.markdown("---")
    st.markdown("Лабораторная работа *\"Прогноз перевода в должности сотрудников компании\"* помогает на реальном примере понять,"
             "можно ли предсказывать будущие карьерные перспективы сотрудников методами машинного обучения.")

    st.markdown("Лабораторная работа состоит их двух частей:")
    st.markdown("**1.** Анализ данных и выбор релевантных данных для прогноза")
    st.markdown("**2.** Обучение и тестирование алгоритма")

    st.markdown("---")

    st.write("В данном приложении, Вы сможете выбрать один из шести заданных видов алгоритма. Также можно подобрать желаемый"
             "набор гиперпараметров для любого из предоставленных алгоритмов. Если же Вы не желаете сами подбирать гиперпараметры"
             "для Вашего алгоритма, Вы можете либо оставить их как есть (по умолчанию), либо использовать алгоритм TPE для "
             "автоматического подбора гиперпараметров.")

    with st.expander("Прочитать про функционал и особенности различных алгоритмов:"):
        st.markdown("* [Нейронные сети](https://academy.yandex.ru/handbook/ml/article/pervoe-znakomstvo-s-polnosvyaznymi-nejrosetyami) \n"
                    "* [Градиентный бустинг](https://academy.yandex.ru/handbook/ml/article/gradientnyj-busting) \n"
                    "* [Гауссовский Классификатор](https://scikit-learn.ru/1-7-gaussian-processes/) \n"
                    "* [Метод Опорных Векторов](https://machinelearningmastery.ru/the-complete-guide-to-support-vector-machine-svm-f1a820d8af0b/) \n"
                    "* [Случайный Лес](https://academy.yandex.ru/handbook/ml/article/ansambli-v-mashinnom-obuchenii) \n"
                    "* [Дерево Решений](https://loginom.ru/blog/decision-tree-p1)")


    st.write("Чтобы обучить алгоритм, необходимо сперва загрузить файл с id сотрудников и их карьерных данных в обучающую"
             "выборку, и также загрузить аналогичный файл в качестве тестовой выборки. Затем, нужно нажать на кнопку "
             "'Обучить'.")

    train_file = st.file_uploader("Загрузите обучающую выборку", key='upload_train_dataset', type=["csv"])
    st.markdown("---")

    if train_file:
        st.markdown("## Информация о датасете")
        st.markdown("Датасет сформирован на основании ряда файлов, в которых хранятся логи о разных сотрудниках (когда опаздывали, "
                    "когда принимали звонок, с каким образованием пришли на рабочее место, и тд.)")
        st.markdown("Образование и специальность сотрудника - единственные категориальные данные в наборе. Перед подачей в алгоритм, "
                    "каждая категория кодируется в отдельное целое число (либо отдельный one-hot вектор, в зависимости от выбранного алгоритма). "
                    "Все остальные столбцы содержат скалярные данные, где одно число соответствует конкретному показателю "
                    "(например, количество заданий которые были выполнены с опозданием). Эти скалярные данные формируются на основании суммы того, "
                    "что найдено в соответствующих логах (например, если в логах написано, что сотрудник сидел 10 минут, 15 минут, и 5 минут за монитором "
                    "в разные времена, мы суммируем все минуты и получаем одно значение которое приписывается сотруднику - 30 минут.")

        st.markdown('---')
    possible_algorithms = ['XGBoost', 'Нейросеть', 'Гауссовский классификатор', 'Метод Опорных Векторов', 'Дерево Решений', 'Случайный Лес']
    selected_algorithm = st.selectbox(
        "Выберите алгоритм для модели: \n (примечание: модель нужно обучать заново после изменения набора столбцов)",
        possible_algorithms, key='selected_algorithm', on_change=possible_algorithms_callback)

    st.write("Настройте гиперпараметры для модели:")
    if selected_algorithm == possible_algorithms[0]:  # XGBoost
        kwargs['n_estimators'] = st.slider("Число деревьев решений", min_value=1, max_value=200, value=100)
        kwargs['GB_max_depth'] = st.slider("Макс. глубина деревьев", min_value=1, max_value=5, value=3)
        learning_rate = st.slider('Скорость обучения', min_value=0.0, max_value=1.0, value=0.5)
        kwargs['GB_learning_rate'] = 10 ** (2 * learning_rate - 2)
        kwargs['min_samples_split'] = st.slider('Минимальное кол-во данных для сплита', min_value=2, max_value=5, value=2)

    elif selected_algorithm == possible_algorithms[1]:  # Нейросеть
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

    elif selected_algorithm == possible_algorithms[2]:  # Гауссовский классификатор
        kwargs['max_iter_predict'] = st.slider("Максимальное число итераций", min_value=10, max_value=500, value=100)
        kwargs['warm_start'] = st.checkbox("Использовать 'тёплое' начало (игнорируйте если не знаете что это такое)")

    elif selected_algorithm == possible_algorithms[3]:  # Метод Опорных Векторов
        kwargs['C'] = st.slider("Регуляризация", min_value=0.1, max_value=5.0, value=1.0)
        kwargs['use_class_weights'] = st.checkbox('Приоритизировать более редкие классы в обучении',
                                                  key='class_weights_checkbox')
        kwargs['kernel'] = st.selectbox("Ядро", ['poly', 'rbf', 'sigmoid'])

    elif selected_algorithm == possible_algorithms[4]:  # Дерево Решений
        max_depth = st.slider("Максимальная глубина дерева (0 = нет ограничений)", min_value=0, max_value=10, value=0)
        kwargs['Tree_max_depth'] = max_depth if max_depth > 0 else None
        kwargs['min_samples_split'] = st.slider("Минимальное число сэмплов для деления дерева", min_value=2,
                                                max_value=5, value=2)
        kwargs['criterion'] = st.selectbox("Функция ошибки", ['gini', 'entropy', 'log_loss'])
        kwargs['use_class_weights'] = st.checkbox('Приоритизировать более редкие классы в обучении',
                                                  key='class_weights_checkbox')

    elif selected_algorithm == possible_algorithms[5]:  # Случайный Лес
        kwargs['n_estimators'] = st.slider("Количество деревьев решений", min_value=1, max_value=200, value=100, step=1)
        max_depth = st.slider("Максимальная глубина дерева (0 = нет ограничений)", min_value=0, max_value=10, value=0)
        kwargs['Tree_max_depth'] = max_depth if max_depth > 0 else None
        kwargs['min_samples_split'] = st.slider("Минимальное число сэмплов для деления дерева", min_value=2,
                                                max_value=5, value=2)
        kwargs['use_class_weights'] = st.checkbox('Приоритизировать более редкие классы в обучении',
                                                  key='class_weights_checkbox')

    with st.expander("Продвинутые опции"):
        kwargs['random_state'] = st.number_input("Рандомное состояние", value=100)

    st.markdown('---')

    with st.expander("Использовать автоматический подбор гиперпараметров"):
        st.write("""
            TPE (Tree-Structured Parzen Estimator) - алгоритм, который основан на Байесовских методах, и используется для подбора гиперпараметров.
            Этот алгоритм способен автоматический перебирать комбинации гиперпараметров используя метод "проба и ошибок".
            
            Если Вы установите этот флажок, программа сама попытается подобрать оптимальный набор гиперпараметров.
            В таком случае, все ранее Вами указанные значения гиперпараметров будут игнорированы.
        """)

        # Возможность использовать Оптуну для подбора гипепараметров
        use_optuna = st.checkbox("Использовать TPE для автоматического подбора гиперпараметров", key='use_optuna')

        if use_optuna:
            optuna_epochs = st.slider("Кол-во эпох для оптимизации алгоритма с помощью TPE",
                                      min_value=1, max_value=1000, value=100, key='optuna_epochs')

            st.markdown("""
                Рекомендуется использовать не меньше 100 эпох для оптимизации.
                
                **ПРИМЕЧАНИЕ: Программе может потребоваться много времени для выполнения большого количества эпох.**""")

            st.markdown("""
                Также учтите, что TPE только подбирает параметры для выбранного Вами алгоритма модели. 
                Сам алгоритм нужно выбирать выше, в ручную.""")

    # Используем session_state чтобы запомнить нажатье различных кнопок
    if 'train_button_clicked' not in sst:
        sst.train_button_clicked = False  # Нужно, чтобы можно было нажать на кнопку "тестировать" после обучения
    if 'model_trained' not in sst:
        sst.model_trained = False  # Нужно, чтобы не обучать модель заново при нажатии кнопки "тестировать"
    if 'algorithm' not in sst:
        sst.algorithm = None       # Нужно, чтобы запомнить алгоритм и его веса после обучения перед тестированием

    # Кнопка для обучения алгоритма
    train_button = st.button(label='Обучить', key='train_button',
                             disabled=True if not train_file else False,
                             on_click=train_buttons_callback)

    # Словарь, который переводит введенную строку в класс заданного алгоритма
    str_to_algorithm = {'Нейросеть': NeuralNetwork, 'XGBoost': GradientBoostingAlgorithm, 'Метод Опорных Векторов': SVMAlgorithm,
                        'Дерево Решений': DecisionTreeAlgorithm, 'Случайный Лес': RandomForestAlgorithm,
                        'Гауссовский классификатор': GaussianAlgorithm}

    # Окно с возможностью выбора столбцов вылезает при загрузке обучающей выборки
    if train_file is not None:
        label_columns = ['type']  # Список всех столбцов, которые подаются в y_train/y_test (всего один такой столбец)
        train_dataframe = get_dataframe(train_file)  # Извлекаем все возможные столбцы из датасета

        # Составляем список всех столбцов, которые можно выбирать или не выбирать.
        # В этот список не попадают столбцы из label_columns
        possible_columns = [i for i in train_dataframe.columns.values if i not in label_columns]
        if 'dataset_columns' not in sst:
            sst.dataset_columns = possible_columns + label_columns
        selected_columns = st.sidebar.multiselect("Выберите столбцы для обучения нейросети",
                                                  key='select_dataset_columns', options=possible_columns,
                                                  default=possible_columns
                                                  )

        select_columns_button = st.sidebar.button("Подтвердить выбор столбцов", key='select_columns_button',
                                                  help="Нажмите на кнопку чтобы выбрать указанные столбцы")
        st.sidebar.markdown('---')
        if select_columns_button:
            sst.dataset_columns = selected_columns + label_columns  # Добавляем label_columns обратно в датасет

        # Доля валидационной выборки от общей обучающей+валидационной выборки
        val_percentage = st.sidebar.slider("Доля валидационной выборки от общего датасета",
                                           min_value=0.0, max_value=0.9, value=0.2,
                                           on_change=possible_algorithms_callback)

        left, right = st.sidebar.columns(2)  # Метод для отрисовки двух кнопок рядом друг с другом
        with left:
            # Кнопка для формирования и вывода матрицы корреляции
            draw_heatmap = st.button(label="Вывести матрицу корреляций", key='heatmap_button',
                                     help="Нажмите на кнопку чтобы вывести матрицу корреляции всех выбранных столбцов")
        with right:
            # Кнопка для формирования и вывода круговой гиаграммы частотности разных классов
            draw_class_histogram = st.button(
                label="Вывести круговую диаграмму частотности классов",
                key='histogram_button',
                help="Нажмите на кнопку чтобы посмотреть гистограмму классов с столбца 'type'"
            )


        if draw_heatmap:
            # Функция для вывода матрицы корреляции
            st.sidebar.pyplot(create_heatmap_streamlit(train_dataframe[sst.dataset_columns]))
        elif draw_class_histogram:
            # Функция для вывода круговой диаграммы классов
            st.sidebar.pyplot(analyze_class_frequency_streamlit(train_dataframe))

        st.sidebar.markdown('---')
        st.sidebar.subheader("Дополнительная визуализация данных")
        column_for_analysis = st.sidebar.selectbox("Выберите столбец для визуализации данных: ", options=selected_columns)
        show_graph_button = st.sidebar.button("Визуализировать")

        if show_graph_button:
            if column_for_analysis in scalar_columns:
                st.sidebar.pyplot(create_cdf_streamlit(train_dataframe[column_for_analysis]))
            elif column_for_analysis in categorical_columns:
                st.sidebar.pyplot(create_histogram_streamlit(train_dataframe[column_for_analysis]))
            else:
                st.sidebar.error("Неопознанный столбец")

    # При нажатии на кпонку "Обучение"
    if sst.train_button_clicked:
        if not sst.model_trained:
            sst.algorithm = str_to_algorithm[selected_algorithm](**kwargs)
        train_dataframe = train_dataframe[sst.dataset_columns] # Обрабатываем датасет и формируем x, y для val и train
        x_train, x_val, y_train, y_val, label_encoders, onehot_encoders, scaler = get_train_dataset(
            train_dataframe, val_percentage=val_percentage,  # Загружаем обработанный датасет
            scale_data=sst.algorithm.requires_normalization, onehot_encode=sst.algorithm.requires_OHE
        )

        # Если модель ранее не была обучена:
        # (условие нужно, чтобы модель не обучалась заново при нажатии нерелевантных кнопок)
        if not sst.model_trained:
            if use_optuna:  # Оптимизация кода Оптуной
                if val_percentage == 0:
                    st.error("Нужно указать размер валидационной выборки >0 для использования Оптуны")
                    st.stop()
                else:
                    optuna_kwargs = optuna_optimization(x_train, y_train, x_val, y_val, sst.algorithm,
                                                        optuna_epochs, kwargs)
                    kwargs.update(optuna_kwargs)
                    # Блок кода, который обновляет значения в kwargs в зависимости от полученного результата из Оптуны
                    if isinstance(sst.algorithm, SVMAlgorithm):
                        kwargs['class_weight'] = calculate_class_weights(y_train) if kwargs['use_class_weights'] else None
                    elif isinstance(sst.algorithm, NeuralNetwork):
                        kwargs['neural_network_hidden_neurons'] = [kwargs[f'layer_{i + 1}_size'] for i in
                                                                   range(kwargs['hidden_layers'])]

                    # Пересоздаем модель, на этот раз используя гиперпараметры которая подобрала Оптуна
                    sst.algorithm = str_to_algorithm[selected_algorithm](**kwargs)

            sst.algorithm.train(x_train, y_train)  # Обучаем алгоритм

            # Валидируем на обучающей выборке
            sst.train_score = sst.algorithm.validate(x_train, y_train, subset_type='train')
            # Валидируем на валидационной выборке (только если у нас есть валидационная выборка)
            sst.val_score = sst.algorithm.validate(x_val, y_val) if val_percentage > 0 else None

            # Устанавливаем флажок, который сообщает что модель уже обучена (чтобы не обучать дважды)
            sst.model_trained = True

        # Выводим точности на обучающей и валидационной выборки в стримлит
        st.info(f"Точность модели на обучающей выборке: {round(100*sst.train_score, 3)}%")
        if val_percentage > 0 and sst.val_score is not None:
            st.info(f"Точность модели на валидационной выборке: {round(100*sst.val_score, 3)}%")

        # Создаем загрузчик для тестового файла
        test_file = st.file_uploader("Загрузите тестовую выборку", key='upload_test_dataset', type=["csv"],
                                     on_change=test_buttons_callback)
        st.markdown("---")

        # Кнопка для теста
        test_button = st.button(label='Тестировать', key='test_button',
                                disabled=True if not test_file else False,
                                on_click=test_buttons_callback)

        if test_button:  # Когда нажимаем на тестовую кнопку:
            x_test, y_test, data_indices = get_test_dataset(  # Собираем тестовый датасет используя кодировщики из обучающей
                test_file, sst.dataset_columns, label_encoders, onehot_encoders, scaler,
                scale_data=sst.algorithm.requires_normalization, onehot_encode=sst.algorithm.requires_OHE
            )

            # Тестируем
            try:  # Пытаемся выполнить код (получается ValueError если пользователь поменял набор столбцов после трейна)
                test_score, predictions = sst.algorithm.test(x_test, y_test)
            # При возникновении ValueError, сообщаем причину ошибки пользователю и просим переобучить модель
            # При использовании нейросети возникает tf.errors.InvalidArgumentError вместо ValueError - но суть та же
            except (ValueError, tf.errors.InvalidArgumentError):
                st.error("Необходимо заново обучить модель после изменения набора столбцов")
                st.stop()
            # Выводим точность на тестовой выборке в стримлит
            st.info(f"Точность модели на тестовой выборке: {round(100*test_score, 3)}%")

            predictions = [label_to_classname[i] for i in predictions.flatten()]  # Переводим лейблы в названия классов
            y_test_labels = [label_to_classname[i] for i in y_test.flatten()]
            # Создаем датафрейм с ID сотрудников и предсказаниям нейросети
            predictions_dataframe = pd.DataFrame(data={'ID Сотрудника': data_indices,
                                                       'Предсказание': predictions,
                                                       'Правильный ответ': y_test_labels})

            # Выводим то, что предсказала модель
            st.write("Предсказания модели: ")
            st.write(predictions_dataframe)

            st.markdown("---")

            st.markdown("## Выводы по лабораторной работе")
            st.markdown("### Вывод Первый")
            st.markdown("Нужно сперва отметить, что распределение классов в обучающей и валидационной выборки не сбалансированы. "
                        "Примерно 54% от всех сотрудников попадают под одну и ту же категорию - соответственно, модель может предсказывать"
                        "эту категорию для всех сотрудников и получить 54% точности. \n"
                        "То есть, приемлемая точность для модели - это 'значительно' выше 54%. Если же итоговая точность равна или ниже чем 54%, "
                        "можно предположить что модель просто 'угадывает'.")
            st.markdown("### Вывод Второй")
            st.markdown("Следуя из первого вывода, и учитывая что данная модель с трудом и не всегда достигает точность в 60% (на валидационной выборке), "
                        "а также учитывая что матрица корреляции показывает что ни один столбец данных сильно не коррелирует с эталоном, можно прийти к следующему выводу - "
                        "данного набора данных не достаточно для надежного предсказания карьерных перспектив сотрудника. Максимум, чего получается "
                        "добиться, это результат 'чуть-чуть лучше рандома'.")
            st.markdown("Если рассмотреть задачу чисто с идеологической/теоретической точки зрения, можно прийти к мнению что такой вывод логичен. "
                        "Ведь всё таки, карьерные перспективы в основном зависят от очень сложных человеческих факторов, амбиций, и людских отношений. "
                        "Вряд ли эти моменты особо коррелируют с данными вроде, время активности монитора/клавиатуры. А данные, которые наверное по настоящему "
                        "релевантные (кто дружит с начальством, чья работа плохо или хорошо оценивается по субъективным критериям компании, кто попал под сокращение из-за "
                        "внешних обстоятельств, и тд.) вряд ли возможно собрать и запаковать в простой .csv файл.")
            st.markdown("### Вывод Третий")
            st.markdown("Суть данной задачи заключается в многоклассовой классификации. Для такой задачи, использование метрики 'accuracy' является логичным выбором.")
            st.image('metrics.png', caption="Уравнения, по которым считаются различные часто-используемые метрики. "
                                            "(TP = True Positive, TN = True Negative, FP = False Positive, FN = False Negative)")
            st.markdown("Нужно отметить, что при многоклассовой классификации метрики 'accuracy' и 'recall' работают абсолютно также, по сколько "
                        "понятия как True Negative и False Positive существуют только в бинарной классификации (когда возможный ответ - Negative или Positive). ")

            st.markdown("---")

            st.markdown("#### Вопрос для пользователя - зачем нужна валидационная выборка отдельно от тестовой?")
            with st.expander("Щёлкните сюда, чтобы увидеть ответ"):
                st.markdown("---")
                st.markdown("##### Ответ")
                st.write("Мы подбираем параметры нашего алгоритма, ориентируясь на значение метрики на валидационной выборке. "
                         "Таким образом, мы по сути подбираем параметры 'под выборку', и у нас может возникнуть эффект, похожий на переобучение. "
                         "Вспомните как происходит переобучение - обучаемые параметры сети слишком плотно подстраиваются под обучающую выборку, "
                         "в результате чего модель не может обобщить и делать правильные предсказания для новых данных, которые она не видела в обучении. ")
                st.markdown("На валидационной выборке может произойти что-то похожее, только на этот раз с гиперпараметрами, которые мы пытаемся оптимизировать. "
                            "Эти гиперпараметры могут идеально работать на данных, под которые мы пытаемся оптимизировать алгоритм, но при этом могут "
                            "работать значительно хуже на других данных. Практика показывает, что **этот эффект особенно часто происходит при "
                            "автоматическом подборе параметров**, где рассматривается очень много разных комбинаций и точность на валидационной выборке используется "
                            "в качестве целевой функции. Иметь отдельные валидационные и тестовые выборки - это всегда хорошая и правильная практика в машинном обучении, "
                            "но она особенно актуальна и даже критический важна при использовании автоматического подбора параметров.")

    st.caption("Автор: Никита Серов")



# Стоит отметить, что данную задачу вероятно можно решить как задачу регрессии, и в таком случае можно использовать метрику MSE или MAE.
# Для этого, необходимо выставить все категории в один ряд - сортировать их с условного "худшего" до "лучшего"
# (Например - если 0 это точное понижение в должности, 3 это точное повышение, а 1 и 2 - где-то между повышением и понижением)
