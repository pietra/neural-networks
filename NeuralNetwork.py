import math

import numpy as np

from util import return_matrix_of_instance_values


class NeuralNetWork():
    def __init__(self, initial_weights):
        self.layers_matrices = []
        self.create_layers_matrixes(initial_weights)

    def create_layers_matrixes(self, initial_weights):
        for layer in initial_weights.values():
            matrix = np.matrix([neuron for neuron in layer.values()])
            self.layers_matrices.append(matrix)

    def propagate_instance_through_network(self, instance):
        matrix_values_instance = return_matrix_of_instance_values(instance)
        for layer_index, layer_matrix in enumerate(self.layers_matrices):
            if layer_index == 0:
                matrices_product = np.dot(layer_matrix, matrix_values_instance)
            else:
                matrices_product = np.dot(layer_matrix, matrices_product)

            matrices_product = self.calculate_sigmoide(matrices_product)

            if layer_index < len(self.layers_matrices) - 1:
                matrices_product = self.add_bias_term(matrices_product)

        return matrices_product

    def calculate_sigmoide(self, product_matrix):
        result = np.matrix([[self.sigmoide(value.item(0, 0))]
                            for value in product_matrix])
        return result

    def sigmoide(self, z):
        return 1.0 / (1 + math.exp(-z))

    def add_bias_term(self, matrices_product):
        return np.vstack([[1.0], matrices_product])
