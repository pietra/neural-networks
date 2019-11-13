import pandas


def read_network_file(filepath):

    with open(filepath) as f:
        content = f.readlines()
        content = [float(line.strip()) for line in content]

    regularization_factor = content[0]
    networks_layers_size = [int(x) for x in content[1:]]

    return regularization_factor, networks_layers_size


def read_initial_weights_file(filepath):
    with open(filepath) as f:
        content = f.readlines()
        layers_weights = [line.strip() for line in content]

    network_weights = {}
    for layer_index, layer in enumerate(layers_weights):
        network_weights[layer_index] = {}
        neurons = layer.split(';')

        for neuron_index, neuron in enumerate(neurons):
            network_weights[layer_index][neuron_index] = []

            neuron = neuron.split(',')
            neuron_weights = [float(weight) for weight in neuron]

            for weight in neuron_weights:
                network_weights[layer_index][neuron_index].append(
                    weight)

    return network_weights


def read_dataset_file(filepath):
    return pandas.read_csv(filepath)
