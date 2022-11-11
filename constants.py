#ВСЕ КОНСТАНТЫ КОТОРЫЕ ИСПОЛЬЗУЮТСЯ В ПРОЕКТЕ

seed = 100             #Сид (для фиксации результата)
train_size = 0.8       #Коэффициент деления на обучающую/валидационную выборку

#Gradient Boosting
learning_rate = 0.1    #Скорость обучения
num_estimators = 100   #Количество деревьев в алгоритме
max_depth = 3          #Глубина деревьев

#Data Preprocessing
scale_data = False     #Нужно ли скалировать данные?
onehot_encode = False  #Нужно ли кодировать данные в One-Hot?