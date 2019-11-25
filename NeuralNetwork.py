import math
import logging
import random
from functools import reduce
import numpy as np
import logging
from util import printMatrix
from data import Data
import matplotlib.pyplot as plt


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

    gradient_output_filename = "backprop_gradients.out"
    numpy_precision = 5

    def __init__(self, structure, initial_weights=[], regFactor=0):
        self.layers_matrices = []
        self.create_layers_matrixes(structure, initial_weights)
        self.regularization_factor = regFactor
        self.structure = structure
        # Set decimals cases shown as output for numpy
        np.set_printoptions(precision=self.numpy_precision)

    def create_layers_matrixes(self, structure, initial_weights=[]):

        if len(initial_weights) > 0:
            # Set pre defined initial weights
            for layer in initial_weights.values():
                matrix = np.matrix([neuron for neuron in layer.values()])
                self.layers_matrices.append(matrix)

            # Check if weight matrix matches the structure specified
            for i in range(len(self.layers_matrices)):
                (numRows, numCols) = self.layers_matrices[i].shape
                # +1 represents the bias neuron
                if numCols != structure[i]+1 or numRows != structure[i+1]:
                    raise ValueError("Network structure and initial weights do not" + \
                                     "match for layer {}".format(i+1))

            log.debug("Weight matrixes initialized with pre-defined values")
        else:
            # Initialize weights with random values
            for i in range(len(structure) - 1):
                # numRows is the number of neurons for the current layer
                numCols = structure[i] + 1  # Add bias neuron
                # numCols is the number of outputs for each neuron
                numRows = structure[i+1]
                self.layers_matrices.append(np.matrix([[0.0]*numCols]*numRows))
                for j in range(numRows):
                    for k in range(numCols):
                        buf = random.uniform(-1, 1)
                        buf = 0.001 if buf == 0 else buf
                        self.layers_matrices[-1][j, k] = buf

            log.debug("Weight matrixes initialized with random values between -1 and 1")
                        

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

    def classify(self, instance):

        attrMatrix = Data.getAttrMatrix(instance)
        a = self.propagate_instance_through_network(attrMatrix)
        f = a[-1]
        y = Data.getResultMatrix(instance)

        # Find out if data has been splitted
        classValues = []
        for key in instance.keys():
            if 'class_' in key:
                classValues.append(key)
        
        if len(classValues) > 0:
            # Prediction is the class with the highest score
            highest = 0
            for i in range(len(f)):
                if f[i,0] > f[highest,0]:
                    highest = i

            # print("f: '%s', y: '%s" % (f, y))
            # print("Predicted class: '%s', expected: '%s'" % (classValues[highest], y[highest,0]))
            return classValues[highest].replace('class_', '')
        else:
            # There is only on class
            if len(f) > 1:
                raise SystemError("Logic error, there should be only one class")

            return f[0,0]              

    def train(self, data, batchSize=0, checkGradients=False, alpha=1, 
              plotError=True):
        """
        Trains the neural network using the backpropagation algorithm.
        """
        totalLayers = len(self.layers_matrices) + 1
        delta = [0]*totalLayers
        numInstances = len(data.instances)
        log.debug("----- Training neural network with:")
        log.debug("Layers: {} ({} in total)".format(self.structure, totalLayers))
        log.debug("Regularization factor: {}".format(self.regularization_factor))
        log.debug("Alpha: {}".format(alpha))

        # Check batch size
        if batchSize < 0:
            raise ValueError("Batch size can`t be negative")
        elif batchSize == 0:
            # Default batch size
            batchSize = numInstances
        elif batchSize > numInstances:
            raise ValueError("Batch size ({}) is greater than the number of instances ({})".\
                format(batchSize, numInstances))
        elif numInstances % batchSize > 0:
            print("Batch size {} and num of instances {} do not match, {} instances will be left out.".\
                format(batchSize, numInstances, numInstances%batchSize))
        
        numBatches = int(numInstances/batchSize)
                    
        log.debug("Batch size: {}".format(batchSize))
        log.debug("Number of instances: {}".format(numInstances))

        # Start training
        errorHistory = []
        learningGraphX = []
        learningGraphY = []
        trainingDone = False
        iterCount = 0
        lastMean = 0
        newMean = 0
        trainingRuns = 0
        print("Training in progress...")
        while not trainingDone:

            for iterCount in range(numBatches):
                startIndex = iterCount*batchSize
                batchInstances = data.instances[startIndex:startIndex+batchSize]
                #log.debug("Training network with batch {}/{}...".\
                #    format(iterCount+1, int(numInstances/batchSize)))
                gradients = [0]*(totalLayers-1)
                for instanceIndex, instance in enumerate(batchInstances):
                    # Calculate delta for the output layer
                    attrMatrix = data.getAttrMatrix(instance)
                    a = self.propagate_instance_through_network(attrMatrix)
                    y = data.getResultMatrix(instance)
                    curDelta = a[-1] - y
                    delta[totalLayers-1] = curDelta

                    error = self.calculate_cost_function([instance])

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
                for i in range(totalLayers-2, -1, -1):
                    tempTheta = self.layers_matrices[i]
                    # (2.1) Remove thetas for bias neurons (first column) and apply reg.
                    curP = self.regularization_factor*tempTheta
                    curP[:,0] = 0
                    # (2.2) Combine gradients and regularization, calculate mean grad.
                    gradients[i] = (1/batchSize)*(gradients[i] + curP)

                # Calculate error for validation
                curError = self.calculate_cost_function(batchInstances, 
                                                        applyReg=True)

                log.debug("Error: %.5f" % curError)
                learningGraphY.append(curError)
                learningGraphX.append(trainingRuns)

                # Calculate the mean value for the last few error values
                errorHistory.append(curError)
                num = 5
                if len(errorHistory) > num:
                    # Calculate mean
                    newMean = 0
                    for value in errorHistory[-num:]:
                        newMean += value
                    newMean = newMean/num

                    if abs(newMean - lastMean) < 0.0001:
                        trainingDone = True
                        log.debug("Training done with mean error diff = {:.5f}".\
                            format(abs(newMean - lastMean)))
                    else:
                        trainingDone = False                  
                else:
                    trainingDone = False

                if checkGradients:
                    trainingDone = True
                    plotError = False
                    self.numeric_gradient_check(data, gradients)
                    if trainingRuns == 0:
                        self.output_gradients_to_file(gradients)

                # log.debug("Error history: {}".format(errorHistory))
                # log.debug("lastMean: {}, newMean: {} -> diff: {}".\
                #     format(lastMean, newMean, abs(newMean-lastMean)))

                lastMean = newMean

                # (4) Update the weights(thetas) for each layer
                for i in range(totalLayers-2, -1, -1):
                    self.layers_matrices[i] = self.layers_matrices[i] - alpha*gradients[i]

            trainingRuns += 1

        if plotError:
            plt.plot(learningGraphX, learningGraphY)
            plt.title("Learning curve")
            plt.ylabel("Error")
            plt.xlabel("Runs")
            plt.grid()
            plt.show()

        # log.debug("f = {}, y = {}".format(a[-1], y))
        # log.debug("Resulting delta: {}".format(delta))
        # log.debug("Resulting gradients: {}".format(gradients))
        # log.debug("Resulting thetas: {}".format(newThetas))
        # log.debug("Original thetas: {}".format(self.layers_matrices))

        # log.debug("---- Numeric gradients calculated via backpropagation")
        # for i in range(totalLayers-1):
        #     s = "Numeric gradient for Theta{}:".format(i)
        #     log.debug(s)
        #     for row in gradients[i]:
        #         log.debug(row)

    def numeric_gradient_check(self, data, backprop, epsilon=0.000001):
        """
        Numerically calculates the gradients for the entire network.
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
        
        # Calculate gradient difference
        log.debug("\n----- Difference between numeric and backpropagation gradients")
        for i in range(len(gradients)):
            diff = backprop[i] -  gradients[i]
            # diff = gradients[i] -  backprop[i]
            
            numRows, numCols = diff.shape
            # mean = 0
            # for rowInd in range(numRows):
            #     for colInd in range(numCols):
            #         mean += diff[rowInd, colInd]
            # mean = mean/(numRows*numCols)
            # log.debug("Gradient diff for layer {} = {}".format(i+1, mean))
            for row in diff:
                log.debug(row)

        # Print backpropagation gradients
        log.debug("\n----- Gradients calculated via backpropagation")
        for layerIndex in range(len(backprop)):
            log.debug("Gradients for Theta{}".format(layerIndex+1))
            (numRows, numCols) = backprop[layerIndex].shape
            for rowInd in range(numRows):
                log.debug(backprop[layerIndex][rowInd, :])
        
        # Print numeric gradients
        log.debug("\n----- Gradients calculated numerically")
        for layerIndex in range(len(gradients)):
            log.debug("Gradients for Theta{}".format(layerIndex+1))
            (numRows, numCols) = gradients[layerIndex].shape
            for rowInd in range(numRows):
                log.debug(gradients[layerIndex][rowInd, :])
                

        return gradients

    def output_gradients_to_file(self, gradients):

        # Write output file in the same format as initial_weights
        fh = open(self.gradient_output_filename, 'w')
        # log.debug("----- Gradients calculated via backpropagation")
        for layerIndex in range(len(gradients)):
            # log.debug("Gradients for Theta{}".format(layerIndex+1))
            (numRows, numCols) = gradients[layerIndex].shape
            line = ""
            for rowInd in range(numRows):
                # log.debug(gradients[layerIndex][rowInd, :])
                for colInd in range(numCols):
                    line += "%.5f" %(gradients[layerIndex][rowInd, colInd])
                    # Coma separated row values
                    if colInd < numCols - 1:
                        line += ", "
                # Semicolumn separated rows
                if rowInd < numRows-1:
                    line += "; "
                
            fh.write(line + '\n')
        fh.close()

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
                j_value += self.j_function(predicted_class, correct_class)

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

        # GAMBI to know if classes have been splitted
        splitted = False

        for attr in instance:
            if 'class_' in attr:
                splitted = True

        if splitted:
            substr = 'class_'
        else:
            substr = 'class'
    
        return [instance[attribute] 
                for attribute in instance if substr in attribute]
