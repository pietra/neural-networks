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
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)
# log.info('New log ---------------------------------------')


class NeuralNetWork():

    def __init__(self, initial_weights, regFactor=0):
        self.layers_matrices = []
        self.create_layers_matrixes(initial_weights)
        self.regularization_factor = regFactor
        # Set decimals cases shown as output for numpy
        np.set_printoptions(precision=3)

    def create_layers_matrixes(self, initial_weights):
        for layer in initial_weights.values():
            matrix = np.matrix([neuron for neuron in layer.values()])
            self.layers_matrices.append(matrix)

        # log.debug("------ Neural network structure:")
        # for layerIndex in range(len(self.layers_matrices)):
        #     log.debug("Theta{}".format(layerIndex+1))
        #     for row in self.layers_matrices[layerIndex]:
        #         log.debug(row)

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

        # printMatrix(activation_matrix, 'a', log)
        return activation_matrix

    def calculate_sigmoide(self, product_matrix):
        result = np.matrix([[self.sigmoide(value.item(0, 0))]
                            for value in product_matrix])
        return result

    def sigmoide(self, z):
        return 1.0 / (1 + math.exp(-z))

    def add_bias_term(self, matrices_product):
        return np.vstack([[1.0], matrices_product])

    def train(self, data, batchSize=1, checkGradients=False):
        """
        Trains the neural network using the backpropagation algorithm.
        """

        totalLayers = len(self.layers_matrices) + 1
        delta = [0]*totalLayers
        log.debug("Number of layers: {}".format(totalLayers))
        log.debug("Regularization factor: {}".format(self.regularization_factor))

        gradients = [0]*(totalLayers-1)

        for instanceIndex, instance in enumerate(data.instances):
            # Calculate delta for the output layer
            attrMatrix = data.getAttrMatrix(instance)
            a = self.propagate_instance_through_network(attrMatrix)
            y = data.getResultMatrix(instance)
            curDelta = a[-1] - y
            delta[totalLayers-1] = curDelta

            error = self.calculate_cost_function([instance])
            # log.debug("Error for instance {}: {}".format(instanceIndex+1, error))

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

        if checkGradients:
            self.numeric_gradient_check(data, backprop=gradients)

        # (4) Update the weights(thetas) for each layer
        newThetas = [0]*len(self.layers_matrices)
        alpha = 1
        for i in range(totalLayers-2, -1, -1):
            newThetas[i] = self.layers_matrices[i] - alpha*gradients[i]

        # Calculate error for validation
        error = self.calculate_cost_function(data.instances, applyReg=True)
        # log.debug("Error for all instances: {}".format(error))
        # log.debug("f = {}, y = {}".format(a[-1], y))
        # log.debug("Resulting delta: {}".format(delta))
        # log.debug("Resulting gradients: {}".format(gradients))
        # log.debug("Resulting thetas: {}".format(newThetas))
        # log.debug("Original thetas: {}".format(self.layers_matrices))

        log.debug("---- Numeric gradients calculated via backpropagation")
        for i in range(totalLayers-1):
            s = "Numeric gradient for Theta{}:".format(i)
            log.debug(s)
            for row in gradients[i]:
                log.debug(row)

        return gradients

    def numeric_gradient_check(self, data, epsilon=0.000001, backprop=None):
        """
        Numerically calculates the gradients for the entire network.
        :param backprop: gradients calculated via backpropagation, if none is 
        given, the difference will not be calculated
        """

        gradients = []
        # Iterate over every weight in the neural network
        for layerIndex, layer in enumerate(self.layers_matrices):

            (rows, cols) = layer.shape
            gradMatrix = np.matrix([[0.0]*cols]*rows)
            for rowInd in range(rows):
                for colInd in range(cols):

                    curTheta = self.layers_matrices[layerIndex][rowInd,colInd]
                    # Calculate error for theta + epsilon
                    self.layers_matrices[layerIndex][rowInd,colInd] = curTheta + epsilon
                    j_plus = self.calculate_cost_function(data.instances, applyReg=True)
                    # Calculate error for theta - epsilon
                    self.layers_matrices[layerIndex][rowInd,colInd] = curTheta - epsilon
                    j_minus = self.calculate_cost_function(data.instances, applyReg=True)
                    # Calculate gradient
                    gradMatrix.itemset((rowInd, colInd), (j_plus - j_minus)/(2*epsilon))
                
                    # Rewrite original network weight
                    self.layers_matrices[layerIndex][rowInd,colInd] = curTheta
                
            gradients.append(gradMatrix)

        log.debug("----- Numeric gradient estimate (epsilon={})".format(epsilon))
        for layerIndex in range(len(gradients)):
            log.debug("Numeric gradient estimate for Theta{}".format(layerIndex+1))
            for row in gradients[layerIndex]:
                log.debug(row)

        if backprop:
            # Calculate gradient difference
            log.debug("----- Running gradient diff")
            for i in range(len(gradients)):

                diff = backprop[i] -  gradients[i]

                log.debug("Gradient diff for layer {}".format(i+1))
                for row in diff:
                    log.debug(row)

        return gradients

    def calculate_cost_function(self, instances, applyReg=False):
        """
        Calcultes the cost for a list of instances.
        applyReg: if true, the sum of network weights will be used (regulariza.)
        """
        j_value = 0
        number_of_instances = len(instances)
        for instance in instances:
            attrMatrix = Data.getAttrMatrix(instance)
            activation_matrix = self.propagate_instance_through_network(
                attrMatrix)
            prediction = activation_matrix[-1]
            vector_of_classes = self.generate_vector_of_classes(instance)
            for predicted_class, correct_class in zip(prediction, vector_of_classes):
                j_value = j_value + \
                    self.j_function(predicted_class, correct_class)

        j_value = j_value / number_of_instances
        if applyReg:
            sum_weights = self.calculate_sum_network_weights(number_of_instances)
            return j_value + sum_weights
        else:
            return j_value

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
