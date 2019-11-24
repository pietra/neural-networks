#!/usr/bin/python3
import getopt, sys
import runBenchmarks
from files_reader import read_network_file, read_initial_weights_file, read_dataset_file
from NeuralNetwork import NeuralNetWork
from data import Data
from util import return_matrix_of_instance_values


def usage():
    s = "Backpropagation usage:\n" +\
        "\t./backpropagation network.txt initial_weights.txt dataset.txt\n" +\
        "Or\n" +\
        "\t./backpropagation network.txt dataset.txt\n" +\
        "Other options:\n" +\
        "\t-g: Run a numeric gradient check and output the gradients\n" +\
        "\tcalculated via backpropagation to a file named \n" +\
        "\tbackprop_gradients.out\n" +\
        "\t-h or --help: Show this message"

    print(s)

def main():

    checkGradients = False
    inputFiles = []

    # For tests
    sys.argv.append('entry_files/network_2.txt')
    sys.argv.append('entry_files/initial_weights_2.txt')
    sys.argv.append('datasets/test_2.csv')

    # Separate network.txt, initial_weights.txt and dataset.txt
    for arg in sys.argv[1:]:
        if arg.endswith('.txt') or arg.endswith('.csv'):
            inputFiles.append(arg)

    # Validate input files
    networkFile = initialWeightsFile = datasetFile = None
    if len(inputFiles) == 3:
        # network, initial_weights and dataset specified
        networkFile = inputFiles[0]
        initialWeightsFile = inputFiles[1]
        datasetFile = inputFiles[2]
    elif len(inputFiles) == 2:
        # network and dataset specified
        networkFile = inputFiles[0]
        datasetFile = inputFiles[1]
    else:
        print("Error parsing input files")
        usage()
        sys.exit()      

    # Parse other parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hg", ["help"])
    except getopt.GetoptError as err:
        print("Error parsing parameters")
        usage()
        sys.exit()
    for o, a in opts:
        if o == '-g':
            checkGradients = True
        elif o in ('-h', '--help'):
            usage()

    print("Starting Neural Networks Algorithm...")
    # 1st parameter: network.txt
    regularization_factor, networks_layers_size = read_network_file(networkFile)

    # 2nd parameter: initial_weights.txt
    if initialWeightsFile:
        initial_weights = read_initial_weights_file(initialWeightsFile)
    else:
        initial_weights = []

    # 3rd parameter: dataset.csv
    dataset = Data(categoricalVars=[])
    dataset.parseFromFile(datasetFile)
    # dataset.normalize()

    neural_network = NeuralNetWork(networks_layers_size,
                                   regFactor=regularization_factor)

    neural_network.train(dataset, batchSize=0)
    # j_value = neural_network.calculate_cost_function(dataset.instances)

def evaluate_performance():
    if(sys.argv[2] == "ionosphere"):
        print("Using Ionosphere dataset.")
        networkFile = 'entry_files/network_ionosphere.txt'
        datasetFile = 'datasets/ionosphere.csv'
    elif(sys.argv[2] == "pima"):
        print("Using Pima dataset.")
        networkFile = 'entry_files/network_pima.txt'
        datasetFile = 'datasets/pima.csv'
    elif(sys.argv[2] == "wine"):
        print("Using Wine dataset.")
        networkFile = 'entry_files/network_wine.txt'
        datasetFile = 'datasets/wine.csv'
    elif(sys.argv[2] == "wdbc"):
        print("Using WDBC dataset.")
        networkFile = 'entry_files/network_wdbc.txt'
        datasetFile = 'datasets/wdbc2.csv'
    else:
        print("Invalid dataset! Please use 'ionosphere', 'pima', 'wine' or 'wdbc'. Defaulting to 'wine'.")
        networkFile = 'entry_files/network_wine.txt'
        datasetFile = 'datasets/wine.csv'

    checkGradients = False
    batchSize = 0

    # 1st parameter: network.txt
    regularization_factor, networks_layers_size = read_network_file(networkFile)
    
    # 3rd parameter: dataset.csv
    dataset = Data(categoricalVars='class')
    dataset.parseFromFile(datasetFile)
    dataset.normalize()
    dataset.splitClasses()

    #Generates testing/training folds
    num_folds = 10
    folds = dataset.generateFolds(num_folds)

    #Analyzes network performance using each fold as testing data once
    allPerformances = []
    allPrecisions = []
    allRecalls = []
    print("Running tests...")
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

        # Training
        neural_network = NeuralNetWork(networks_layers_size, regFactor=regularization_factor)
        neural_network.train(training_data, batchSize=batchSize, alpha=1, plotError=False)

        runPerformances = []
        runPrecisions = []
        runRecalls = []
        for class_value in dataset.listClassValues():
            positive_class = class_value
            true_positives = 0
            false_positives = 0
            true_negatives = 0
            false_negatives = 0
            predictions = 0

            for instance in testing_data.instances:
                output = neural_network.classify(instance)
                predictions += 1
                print("Prediction {}, is {}".format(output, instance['class']))

                if (output == positive_class) and (instance['class'] == positive_class):
                    true_positives += 1
                elif (output == positive_class) and (instance['class'] != positive_class):
                    false_positives += 1
                elif (output != positive_class) and (instance['class'] != positive_class):
                    true_negatives += 1
                elif (output != positive_class) and (instance['class'] == positive_class):
                    false_negatives += 1

            #print("TP: {}, FP: {}, TN: {}, FN: {}".format(true_positives, false_positives, true_negatives, false_negatives))
            #Adds results to this run's results
            runPerformances.append((true_positives+true_negatives)/predictions)    #Right guesses
            runPrecisions.append(true_positives / (true_positives + false_positives))    #Right guesses from instances guessed positive
            runRecalls.append(true_positives / (true_positives + false_negatives))   #Right guesses from instances that were supposed to be positive

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


if __name__ == "__main__":
    if (sys.argv[1] == "-p"):
        evaluate_performance()
    else:
        main()
    # main()
    #evaluate_performance()
    # runBenchmarks.generateIonosphere()
    # runBenchmarks.generatePima()
    # runBenchmarks.generateWine()
    # runBenchmarks.generateWdbc()
