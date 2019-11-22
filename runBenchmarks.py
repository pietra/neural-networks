#!/usr/bin/python3
import getopt, sys
from files_reader import read_network_file, read_initial_weights_file, read_dataset_file
from NeuralNetwork import NeuralNetWork
from data import Data
from util import return_matrix_of_instance_values


def generateIonosphere() -> (Data, NeuralNetWork):

    checkGradients = False
    initial_weights = []

    networkFile = 'entry_files/ionosphere.txt'
    datasetFile = 'datasets/ionosphere.csv'

    # 1st parameter: network.txt
    regularization_factor, networks_layers_size = read_network_file(networkFile)  
    
    # 3rd parameter: dataset.csv
    dataset = Data(categoricalVars=['class'])
    dataset.parseFromFile(datasetFile)

    # print("Dataset instances before normalization")
    # for i in range(1, 10):
    #     print(dataset.instances[i]) 

    dataset.normalizeAttributes(normalizeClass=False)
    # print("Dataset instances after normalization")
    # for i in range(1, 10):
    #     print(dataset.instances[i]) 

    # neural_network = NeuralNetWork(networks_layers_size, initial_weights=initial_weights, regFactor=regularization_factor)

    # neural_network.train(dataset, batchSize=0, checkGradients=checkGradients)
    # j_value = neural_network.calculate_cost_function(dataset.instances)
