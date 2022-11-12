#ВСЕ КОНСТАНТЫ КОТОРЫЕ ИСПОЛЬЗУЮТСЯ В ПРОЕКТЕ

seed = 100             #Сид (для фиксации результата)
train_size = 0.8       #Коэффициент деления на обучающую/валидационную выборку
num_classes = 4        #Количество классов в датасете

#Gradient Boosting
GB_learning_rate = 0.1    #Скорость обучения
num_estimators = 100   #Количество деревьев в алгоритме
max_depth = 3          #Глубина деревьев

#Neural Network
neural_network_hidden_neurons = [32, 16, 8]
NN_learning_rate = 1e-3
num_epochs = 10
use_class_weights = True
batch_size = 32

#Data Preprocessing
scale_data = True     #Нужно ли скалировать данные?
onehot_encode = True  #Нужно ли кодировать данные в One-Hot?