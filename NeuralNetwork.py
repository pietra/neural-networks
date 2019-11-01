import math
import logging

from functools import reduce
import numpy as np

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

        prediction = matrices_product.A1
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

    def calculate_cost_function(self, instances):
        j_value = 0
        number_of_instances = len(instances)
        for instance in instances:
            prediction, activation_matrix = self.propagate_instance_through_network(
                instance)
            vector_of_classes = self.generate_vector_of_classes(instance)
            for predicted_class, correct_class in zip(prediction, vector_of_classes):
                j_value = j_value + \
                    self.j_function(predicted_class, correct_class)

        j_value = j_value / number_of_instances
        sum_weights = self.calculate_sum_network_weights(number_of_instances)

        return j_value + sum_weights

    def j_function(self, predicted_value, real_value):
        return - real_value * math.log(predicted_value) - \
            (1 - real_value) * math.log(1 - predicted_value)

    def calculate_sum_network_weights(self, number_of_instances):
        return (self.regularization_factor /
                (2*number_of_instances)) * self.sum_network_weights()

    def sum_network_weights(self):
        sum_weights = 0
        for layer in self.layers_matrices:
            for neuron in layer:
                for weight_index, weight in enumerate(neuron.A1):
                    if weight_index == 0:
                        continue
                    else:
                        sum_weights += math.pow(weight, 2)
        return sum_weights

    def generate_vector_of_classes(self, instance):
        return [instance[attribute]
                for attribute in instance if 'class' in attribute]
