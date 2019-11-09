#!/usr/bin/python3
import sys
from files_reader import read_network_file, read_initial_weights_file, read_dataset_file
from NeuralNetwork import NeuralNetWork
from data import Data


def main():
    print("Starting Neural Networks Algorithm...")

    # For tests
    sys.argv.append('entry_files/network_2.txt')
    sys.argv.append('entry_files/initial_weights_2.txt')
    sys.argv.append('datasets/test_2.csv')

    # 1st parameter: network.txt
    regularization_factor, networks_layers_size = read_network_file(
        sys.argv[1])

    # 2nd parameter: initial_weights.txt
    initial_weights = read_initial_weights_file(sys.argv[2])

    # 3rd parameter: dataset.csv
    dataset = Data(categoricalVars=[])
    dataset.parseFromFile(sys.argv[3])
    dataset.normalizeAttributes()

    neural_network = NeuralNetWork(initial_weights, 
                                   regFactor=regularization_factor)

    realGrads = neural_network.train(dataset)
    estimateGrads = neural_network.numeric_gradient_estimate(dataset)
    # neural_network.gradient_difference(realGrads, estimateGrads)
    # j_value = neural_network.calculate_cost_function(dataset.instances)

def evaluatePerformance():
    # For tests
    sys.argv.append('entry_files/network_2.txt')
    sys.argv.append('entry_files/initial_weights_2.txt')
    sys.argv.append('datasets/test_2.csv')

    # 1st parameter: network.txt
    regularization_factor, networks_layers_size = read_network_file(
        sys.argv[1])

    # 2nd parameter: initial_weights.txt
    initial_weights = read_initial_weights_file(sys.argv[2])

    # 3rd parameter: dataset.csv
    dataset = Data(categoricalVars=[])
    dataset.parseFromFile(sys.argv[3])
    dataset.normalizeAttributes()

    #Creates neural network form files
    neural_network = NeuralNetWork(initial_weights, regularization_factor)

    #Generates testing/training folds
    num_folds = 2
    folds = dataset.generateFolds(num_folds)
    for i in range(num_folds):
        print("")
        print("----- Test run {} of {} -----".format(i+1, num_folds))
        print("")
        testing_data = Data()
        for instance in folds[i]:   #Takes one fold for testing
            testing_data.addInstance(instance)

        training_data = Data()
        for fold in folds:
            if folds.index(fold) != i:  #Skips the fold used for testing
                for instance in fold:
                    training_data.addInstance(instance)

        #Trains network
        neural_network.train(training_data)
        neural_network.numeric_gradient_estimate(training_data)

        #v---v THE REST OF THIS FUNCTION IS UNTESTED AS THE PROPAGATION DOESN'T WORK YET v---v

        #Propagates testing data through the network
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        for instance in testing_data.instances:
            output = neural_network.propagate_instance_through_network(instance)
            #TODO: compare network output with correct result
            #TODO: update TP, FP, TN, FN numbers


if __name__ == "__main__":
    main()
    #evaluatePerformance()
