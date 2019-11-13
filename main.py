#!/usr/bin/python3
import sys
from files_reader import read_network_file, read_initial_weights_file, read_dataset_file
from NeuralNetwork import NeuralNetWork
from data import Data
from util import return_matrix_of_instance_values


def main():
    print("Starting Neural Networks Algorithm...")

    checkGradients = False

    # For tests
    sys.argv.append('entry_files/network_2.txt')
    sys.argv.append('entry_files/initial_weights_2.txt')
    sys.argv.append('datasets/test_2.csv')

    # 1st parameter: network.txt
    regularization_factor, networks_layers_size = read_network_file(sys.argv[1])

    # 2nd parameter: initial_weights.txt
    initial_weights = read_initial_weights_file(sys.argv[2])

    # 3rd parameter: dataset.csv
    dataset = Data(categoricalVars=[])
    dataset.parseFromFile(sys.argv[3])
    # dataset.normalizeAttributes()

    # neural_network = NeuralNetWork(networks_layers_size, initial_weights=initial_weights, regFactor=regularization_factor)
    neural_network = NeuralNetWork(networks_layers_size, regFactor=regularization_factor)

    neural_network.train(dataset, batchSize=2, checkGradients=False)
    # j_value = neural_network.calculate_cost_function(dataset.instances)


def evaluate_performance():
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

    #Analyzes network performance using each fold as testing data once
    allPerformances = []
    allPrecisions = []
    allRecalls = []
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

        print(neural_network.propagate_instance_through_network(return_matrix_of_instance_values(testing_data.instances[0])))

    '''
        #v---v THE REST OF THIS FUNCTION IS UNTESTED AND A WORK IN PROGRESS v---v

        runPerformances = []
        runPrecisions = []
        runRecalls = []
        #TODO: loop this taking every possible output as the positive class once
            true_positives = 0
            false_positives = 0
            true_negatives = 0
            false_negatives = 0
            for instance in testing_data.instances:
                output = neural_network.propagate_instance_through_network(instance)
                #TODO: compare network output with correct result
                #TODO: update TP, FP, TN, FN numbers

            #Adds results to this run's results
            runPerformances.append((truePositives+trueNegatives)/len(predictions))    #Right guesses
            runPrecisions.append(truePositives / (truePositives + falsePositives))    #Right guesses from instances guessed positive
            runRecalls.append(truePositives / (truePositives + falseNegatives))   #Right guesses from instances that were supposed to be positive

        #Calculates average performance, recall and precision for this run
        runAvgPerformance = sum(runPerformances)/len(runPerformances)
        runAvgPrecision = sum(runPrecisions)/len(runPrecisions)
        runAvgRecall = sum(runRecalls)/len(runRecalls)

        print("Network performance: {:.2f}% of guesses (precision: {:.2f}% / recall: {:.2f}%)".format(runAvgPerformance*100, runAvgPrecision*100, runAvgRecall*100))

        allPerformances.append(runAvgPerformance)   #Adds this run's performance to the list
        allPrecisions.append(runAvgPrecision)   #Adds this run's precision to the list
        allRecalls.append(runAvgRecall)   #Adds this run's recall to the list

    #Calculates averages for all runs
    avgPerformance = sum(allPerformances)/len(allPerformances)
    avgPrecision = sum(allPrecisions)/len(allPrecisions)
    avgRecall = sum(allRecalls)/len(allRecalls)
    #Calculates F1-Measure for all runs
    f1 = (2*avgPrecision*avgRecall) / (avgPrecision+avgRecall)

    print("")
    print("----- FINAL PERFORMANCE -----")
    print("")
    print("Network's average performance: {:.2f}% (precision: {:.2f}% / recall: {:.2f}%)".format(avgPerformance*100, avgPrecision*100, avgRecall*100))
    print("F1-measure from averages: {:.2f}%".format(f1*100))
    '''



if __name__ == "__main__":
    main()
    #evaluate_performance()
