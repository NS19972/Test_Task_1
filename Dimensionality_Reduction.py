# Данный файл отображает попытки визуализацию датасета с применением различных методов для декомпозиции матриц
# Отмечается, что никакие чёткие кластеры не наблюдаются ни в одном методе
# Этот факт также указывает на то, что классы не отличимы друг от друга при наличии текущих фичей

import numpy as np
from constants import *
from matplotlib import pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

from Data_Formation import get_dataframe, get_train_dataset

dataframe = get_dataframe('dataset/train.csv')  # Читаем файл с датасетом

# Извлекаем датасет
X, _, Y, _, label_encoders, onehot_encoders, scaler = get_train_dataset(dataframe, val_percentage=0,
                                                                        scale_data=True, onehot_encode=True)

pca = PCA(n_components=2)   # Principal Component Anaylsis
X_PCA = pca.fit_transform(X)

tsne = TSNE(n_components=2, perplexity=10, init='pca')  # T-Distributed Stochastic Neighboring Embedding
X_TSNE = tsne.fit_transform(X)

umap = UMAP(n_components=2)  # Uniform Manifold Approximation and Projection
X_UMAP = umap.fit_transform(X)

fig, ax = plt.subplots(1, 3, figsize=(16, 8))


# Функция для рисования точек из разных классов
def make_plot(num, data):
    color_list = ['blue', 'red', 'green', 'black']
    for i in np.unique(Y):
        class_indices = (Y == i).nonzero()[0]
        class_i = data[class_indices]
        ax[num].scatter(class_i[:, 0], class_i[:, 1], color=color_list[i])


make_plot(0, X_PCA)   # Рисуем результат PCA в первое полотно
make_plot(1, X_TSNE)  # Рисуем результат t-SNE во второе полотно
make_plot(2, X_UMAP)  # Рисуем результат UMAP в третье полотно

# Добавляем заголовки
ax[0].set_title("PCA", fontsize=16)
ax[0].set_xlabel("Principal Component 1")
ax[2].set_ylabel("Principal Component 2")

ax[1].set_title("t-SNE", fontsize=16)
ax[1].set_xlabel("Principal Component 1")
ax[2].set_ylabel("Principal Component 2")

ax[2].set_title("U-MAP", fontsize=16)
ax[2].set_xlabel("Principal Component 1")
ax[2].set_ylabel("Principal Component 2")

# Отображаем результат
plt.legend()
plt.show()

