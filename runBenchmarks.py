#!/usr/bin/python3
import getopt, sys
from files_reader import read_network_file, read_initial_weights_file, read_dataset_file
from NeuralNetwork import NeuralNetWork
from data import Data
from util import return_matrix_of_instance_values


def generateIonosphere() -> (Data, NeuralNetWork):
    """
    https://archive.ics.uci.edu/ml/datasets/Ionosphere
    34 atributos, 351 exemplos, 2 classes
    """

    checkGradients = False
    batchSize = 0

    networkFile = 'entry_files/network_ionosphere.txt'
    datasetFile = 'datasets/ionosphere.csv'

    # 1st parameter: network.txt
    regularization_factor, networks_layers_size = read_network_file(networkFile)  
    
    # 3rd parameter: dataset.csv
    dataset = Data(categoricalVars=['class'])
    dataset.parseFromFile(datasetFile)
    dataset.normalize()
    dataset.splitClasses()

    neural_network = NeuralNetWork(networks_layers_size, regFactor=regularization_factor)
    neural_network.train(dataset, batchSize=batchSize, checkGradients=checkGradients)

    errorCount = 0
    totalCount = 0
    for inst in dataset.instances:
        pred = neural_network.classify(inst)
        # print("Prediction: '%s', Y: '%s'" % (pred, inst['class']))
        totalCount += 1
        if pred != inst['class']:
            errorCount += 1

    print("Result: %d hits and %d misses" % (totalCount-errorCount, errorCount))

    return (dataset, neural_network)


def generatePima(batchSize=0) -> (Data, NeuralNetWork):
    """
    https://github.com/EpistasisLab/penn-ml-benchmarks/tree/master/datasets/classification/pima
    8 atributos, 768 exemplos, 2 classes
    """
    networkFile = 'entry_files/network_pima.txt'
    datasetFile = 'datasets/pima.csv'

    # 1st parameter: network.txt
    regularization_factor, networks_layers_size = read_network_file(networkFile)
    
    # 3rd parameter: dataset.csv
    dataset = Data(categoricalVars='class')
    dataset.parseFromFile(datasetFile)
    dataset.normalize()
    dataset.splitClasses()

    # Train and classify
    neural_network = NeuralNetWork(networks_layers_size, regFactor=regularization_factor)
    neural_network.train(dataset, batchSize=batchSize, alpha=1, plotError=False)

    errorCount = 0
    totalCount = 0
    for inst in dataset.instances:
        pred = neural_network.classify(inst)
        # print("Prediction: '%s', Y: '%s'" % (pred, inst['class']))
        totalCount += 1
        if pred != inst['class']:
            errorCount += 1

    print("Result: %d hits and %d misses" % (totalCount-errorCount, errorCount))     

    return (dataset, neural_network)


def generateWine(batchSize=0) -> (Data, NeuralNetWork):
    """
    https://archive.ics.uci.edu/ml/datasets/wine
    13 atributos, 178 exemplos, 3 classes
    """
    networkFile = 'entry_files/network_wine.txt'
    datasetFile = 'datasets/wine.csv'

    # 1st parameter: network.txt
    regularization_factor, networks_layers_size = read_network_file(networkFile)
    
    # 3rd parameter: dataset.csv
    dataset = Data(categoricalVars=['class'])
    dataset.parseFromFile(datasetFile)

    dataset.normalize()
    dataset.splitClasses()

    neural_network = NeuralNetWork(networks_layers_size, regFactor=regularization_factor)

    neural_network.train(dataset, batchSize=batchSize, alpha=1)

    errorCount = 0
    totalCount = 0
    for inst in dataset.instances:
        pred = neural_network.classify(inst)
        # print("Prediction: '%s', Y: '%s'" % (pred, inst['class']))
        totalCount += 1
        if pred != inst['class']:
            errorCount += 1

    print("Result: %d hits and %d misses" % (totalCount-errorCount, errorCount))

    return (dataset, neural_network)


def generateWdbc(batchSize=0) -> (Data, NeuralNetWork):
    """
    https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
    32 atributos, 569 exemplos, 2 classes
    """
    networkFile = 'entry_files/network_wdbc.txt'
    datasetFile = 'datasets/wdbc2.csv'

    # 1st parameter: network.txt
    regularization_factor, networks_layers_size = read_network_file(networkFile)
    
    # 3rd parameter: dataset.csv
    dataset = Data(categoricalVars=['class'])
    dataset.parseFromFile(datasetFile)
    dataset.normalize()
    dataset.splitClasses()

    neural_network = NeuralNetWork(networks_layers_size, regFactor=regularization_factor)
    neural_network.train(dataset, batchSize=batchSize, alpha=1)

    errorCount = 0
    totalCount = 0
    for inst in dataset.instances:
        pred = neural_network.classify(inst)
        # print("Prediction: '%s', Y: '%s'" % (pred, inst['class']))
        totalCount += 1
        if pred != inst['class']:
            errorCount += 1

    print("Result: %d hits and %d misses" % (totalCount-errorCount, errorCount))

    return (dataset, neural_network)
