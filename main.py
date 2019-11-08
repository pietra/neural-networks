#!/usr/bin/python3
import sys
from files_reader import read_network_file, read_initial_weights_file, read_dataset_file
from NeuralNetwork import NeuralNetWork
from data import Data


def main():
    print("Starting Neural Networks Algorithm...")

    # For tests
    sys.argv.append('entry_files/network.txt')
    sys.argv.append('entry_files/initial_weights.txt')
    sys.argv.append('datasets/test.csv')

    # 1st parameter: network.txt
    regularization_factor, networks_layers_size = read_network_file(
        sys.argv[1])

    # 2nd parameter: initial_weights.txt
    initial_weights = read_initial_weights_file(sys.argv[2])

    # 3rd parameter: dataset.csv
    dataset = Data(categoricalVars=[])
    dataset.parseFromFile(sys.argv[3])

    neural_network = NeuralNetWork(initial_weights, 
                                   regFactor=regularization_factor)

    neural_network.train(dataset)
    # j_value = neural_network.calculate_cost_function(dataset.instances)


if __name__ == "__main__":
    main()
