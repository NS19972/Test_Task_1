import numpy as np


def calculate_class_weights(y):
    unique_categories, counts = np.unique(y, return_counts=True)
    counts = counts/counts.sum()
    weights_dict = {i: j for i, j in enumerate(counts)}
    return weights_dict