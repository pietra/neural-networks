import math
import logging

from functools import reduce
import numpy as np
import logging
from util import printMatrix
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

    def propagate_instance_through_network(self, instance_matrix, debug=False):

        # log.debug('Propagating: {}'.format(instance_matrix))
        activation_matrix = [instance_matrix]
        for layer_index, layer_matrix in enumerate(self.layers_matrices):
            if layer_index == 0:
                matrices_product = np.dot(layer_matrix, instance_matrix)
            else:
                matrices_product = np.dot(layer_matrix, matrices_product)

            matrices_product = self.calculate_sigmoide(matrices_product)

            if layer_index < len(self.layers_matrices) - 1:
                matrices_product = self.add_bias_term(matrices_product)

            # Append matrix product
            activation_matrix.append(matrices_product)

        printMatrix(activation_matrix, 'a', log)
        return activation_matrix

    def calculate_sigmoide(self, product_matrix):
        result = np.matrix([[self.sigmoide(value.item(0, 0))]
                            for value in product_matrix])
        return result

    def sigmoide(self, z):
        return 1.0 / (1 + math.exp(-z))

    def add_bias_term(self, matrices_product):
        return np.vstack([[1.0], matrices_product])

    def train(self, data, batchSize=1):

        totalLayers = len(self.layers_matrices) + 1
        delta = [0]*totalLayers
        log.debug("Number of layers: {}".format(totalLayers))
        log.debug("Regularization factor: {}".format(self.regularization_factor))

        gradients = [0]*(totalLayers-1)

        for instanceIndex in range(len(data.instances)):
            # Calculate delta for the output layer
            instance = data.getAttrMatrix(instanceIndex)
            a = self.propagate_instance_through_network(instance)
            y = data.getResultMatrix(instanceIndex)
            curDelta = a[-1] - y
            delta[totalLayers-1] = curDelta

            # (1.3) Calculate delta for the hidden layers
            for i in range(totalLayers-2, 0, -1):
                
                step1 = self.layers_matrices[i].T*delta[i+1]
                step2 = np.multiply(step1, a[i])
                step3 = np.multiply(step2, 1-a[i])
                # Remove delta value for bias neuron (first row)
                delta[i] = np.delete(step3, 0, axis=0)

            # (1.4) Update gradients for every layer based on the current example
            for i in range(totalLayers-2, -1, -1):
                gradients[i] = gradients[i] + delta[i+1]*a[i].T
    
        # (2) Calculate the final gradients for every layer
        numExamples = len(data.instances)
        for i in range(totalLayers-2, -1, -1):
            tempTheta = self.layers_matrices[i]
            # (2.1) Remove thetas for bias neurons (first column) and apply reg.
            # curP = np.delete(self.regularization_factor*tempTheta, 0, axis=1)
            curP = self.regularization_factor*tempTheta
            curP[:,0] = 0
            # (2.2) Combine gradients and regularization, calculate mean grad.
            gradients[i] = (1/numExamples)*(gradients[i] + curP)

        # (4) Update the weights(thetas) for each layer
        newThetas = [0]*len(self.layers_matrices)
        alpha = 1
        for i in range(totalLayers-2, -1, -1):
            newThetas[i] = self.layers_matrices[i] - alpha*gradients[i]

        log.debug("f = {}, y = {}".format(a[-1], y))
        # log.debug("Resulting delta: {}".format(delta))
        # log.debug("Resulting gradients: {}".format(gradients))
        # log.debug("Resulting thetas: {}".format(newThetas))
        # log.debug("Original thetas: {}".format(self.layers_matrices))

        for i in range(totalLayers-1):

            s = "Gradiente numerico de Theta{}:".format(i)
            print(s)
            for row in gradients[i]:
                print(row)

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
