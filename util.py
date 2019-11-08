import numpy as np


def return_matrix_of_instance_values(instance):
    values = [[1.0]]
    for key, value in zip(instance.keys(), instance.values()):
        if 'class' not in key:
            values.append([value])
    return np.matrix(values)


def printMatrix(m, rowName='m', logger=None):

    # if not isinstance(m, np.matrix):
    #     raise ValueError("Argument m is not a numpy matrix")

    for i, row in enumerate(m):

        s = "{}{}: {}".format(rowName, i+1, row.T)

        if logger:
            logger.debug(s)
        else:
            print(s)