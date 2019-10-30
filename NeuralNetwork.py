import math

import numpy as np
import logging
from util import return_matrix_of_instance_values
from data import Data


# Initialize logging
log = logging.getLogger('neural-network')
log.setLevel(logging.DEBUG)
# ch = logging.FileHandler('neural-network.log')
ch = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)
# log.info('New log ---------------------------------------')


class NeuralNetWork():
    def __init__(self, initial_weights, regFactor=0):
        self.layers_matrices = []
        self.create_layers_matrixes(initial_weights)
        self.regularization_factor = regFactor

    def create_layers_matrixes(self, initial_weights):
        for layer in initial_weights.values():
            matrix = np.matrix([neuron for neuron in layer.values()])
            self.layers_matrices.append(matrix)
        
        # log.debug('Resulting neural network:\n {}'.format(self.layers_matrices))

    def propagate_instance_through_network(self, instance, debug=False):
        matrix_values_instance = return_matrix_of_instance_values(instance)
        # log.debug('Propagating: {}'.format(matrix_values_instance))
        activation_matrix = list()
        for layer_index, layer_matrix in enumerate(self.layers_matrices):
            if layer_index == 0:
                matrices_product = np.dot(layer_matrix, matrix_values_instance)
            else:
                matrices_product = np.dot(layer_matrix, matrices_product)

            matrices_product = self.calculate_sigmoide(matrices_product)

            if layer_index < len(self.layers_matrices) - 1:
                matrices_product = self.add_bias_term(matrices_product)

            # Append matrix product as a new row
            activation_matrix.append(matrices_product.transpose().tolist()[0])
            
        prediction = matrices_product.item((0,0))
        # log.debug("Activation matrix: {}".format(activation_matrix))
        # log.debug("Prediction: {}".format(prediction))
        return (prediction, activation_matrix)

    def calculate_sigmoide(self, product_matrix):
        result = np.matrix([[self.sigmoide(value.item(0, 0))]
                            for value in product_matrix])
        return result

    def sigmoide(self, z):
        return 1.0 / (1 + math.exp(-z))

    def add_bias_term(self, matrices_product):
        return np.vstack([[1.0], matrices_product])

    def train(self, data, batchSize=1):

        for instance in data.instances:
            (f, activation_matrix) = self.propagate_instance_through_network(instance)
            y = instance[data.className]

            log.debug("f = {}, y = {}".format(f, y))

