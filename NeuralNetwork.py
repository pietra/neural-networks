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
        # 1.J=0 // inicializa a variável que irá acumular o erro total da rede
        # 2.Para cada exemplo (x(i), y(i)) no conjunto de treinamento:
        # 1.Propaga x(i) e obtém as saídas fθ(x(i)) preditas pela rede
        # 2.Calcula o vetor J(i) com o custo associado à cada saída da rede para o exemplo atual
        # J(i) = -y(i) .* log(fθ(x(i))) - (1-y(i)) .* log(1 - fθ(x(i)))
        # J = J + sum(J(i)) // soma os elementos do vetor J(i) e acumula o resultado em J
        # 3.J = J / n // divide o erro total calculado pelo número de exemplos
        # 4.S = eleva cada peso da rede ao quadrado (exceto os pesos de bias) e os soma
        # 5.S = (λ/(2n)) * S // calcula o termo de regularização
        # 6.Retorna o custo regularizado J+S
        j_value = 0
        number_of_instances = len(instances)
        for instance in instances:
            prediction, activation_matrix = self.propagate_instance_through_network(
                instance)
            for element in prediction:
                j_value = j_value + self.j_function(element, instance['class'])

        j_value = j_value / number_of_instances

        s = self.sum_network_weights(activation_matrix)
        s = (self.regularization_factor/2*number_of_instances) * s

        return j_value + s

    def sum_network_weights(self, activation_matrix):
        s = 0
        for layer in activation_matrix:
            for j in range(1, len(layer)):
                s = s + math.pow(layer[j], 2)

        return s

    def j_function(self, predicted_value, real_value):
        j = - real_value * math.log(predicted_value) - \
            (1 - real_value) * math.log(1 - predicted_value)
        return j
