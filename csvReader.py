import numpy as np


def read_csv(filename):
    data = np.genfromtxt(filename, delimiter=",")

    normalized_data = np.ndarray(data.shape)
    normalized_data.fill(0)
    max_data = data.max()
    min_data = data.min()
    for idx, value in enumerate(data):
        normal_value = (value - min_data) / (max_data - min_data) * 255
        normalized_data[idx] = normal_value
    return normalized_data
