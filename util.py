import numpy as np


def return_matrix_of_instance_values(instance):
    values = [[1.0]]
    for key, value in zip(instance.keys(), instance.values):
        if 'class' not in key:
            values.append([value])
    return np.matrix(values)
