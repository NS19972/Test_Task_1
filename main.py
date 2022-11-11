import numpy as np
import sklearn

from Data_Formation import get_train_dataset
from sklearn.ensemble import *
from sklearn.metrics import recall_score
from sklearn.svm import *
from constants import *

np.random.seed(seed)   #Устанавливаем сид (sklearn использует сид от numpy)

if __name__ == "__main__":
    x_train, x_val, y_train, y_val, encoders, scaler = get_train_dataset() #Загружаем обработанный датасет

    model = GradientBoostingClassifier(n_estimators = 20, max_depth=2)     #Задаем параметры алгоритма
    model.fit(x_train, y_train.flatten())                                  #Обучаем алгоритм
    y_pred = model.predict(x_val)                                          #Извлекаем предсказание
    score = recall_score(y_val, y_pred, average='micro')                   #Считаем метрику

    #Выводим полученную точность (метрика recall)
    print(f"Model recall score on validation subset is {score}")

#Стоит отметить, что данную задачу вероятно можно решить как задачу регрессии, и в таком случае можно использовать метрику MSE или MAE.
#Для этого, необходимо выставить все категории в один ряд - сортировать их с условного "худшего" до "лучшего"
#(Например - если 0 это точное понижение в должности, 3 это точное повышение, а 1 и 2 - где-то между повышением и понижением)