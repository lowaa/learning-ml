from typing import List, NamedTuple, Callable

import numpy as np

np.random.seed(123)


def calc_layer_z(weight: np.array,
                 previous_activation: np.array,
                 bias: np.array) -> np.array:
    return np.matmul(weight, previous_activation) + bias


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))


TrainResult = NamedTuple('TrainResult', [
    ('layer_z', np.array),
    ('layer_activation', np.array),
    ('error', np.array),
])


class Layer(object):
    weights: np.array
    biases: np.array

    def __init__(self, num_layer_inputs, num_layer_outputs):
        if num_layer_inputs < 1:
            raise ValueError('num_layer_inputs must be >= 1')
        if num_layer_outputs < 1:
            raise ValueError('num_layer_outputs must be >= 1')

        # Initialise with random weightings

        # Each weight matrix will be the ...
        # number of neurons in the previous layer : width
        # number of neurons in the current layer : height
        self.weights = np.random.rand(num_layer_outputs, num_layer_inputs)

        # The bias for each layer.
        # The last index is for output layer. Work backwards from that.
        # For each matrix...
        # Width = 1
        # Height = number of neurons in the current layer
        self.biases = np.random.rand(num_layer_outputs, 1)


class NeuralNetwork(object):
    # Public so you can change the values
    layers: List[Layer]

    # Activation function
    _activation_fn: Callable[[float], float]
    # Derivative of activation function
    _activation_der_fn: Callable[[float], float]

    def __init__(self, num_inputs: int,
                 num_outputs: int,
                 learning_rate: float,
                 hidden_layer_sizes: List[int] = None,
                 activation_fn=None):

        if num_inputs < 1:
            raise ValueError('num_inputs must be >= 1')
        if num_outputs < 1:
            raise ValueError('num_outputs must be >= 1')

        self._num_inputs = num_inputs
        self._num_outputs = num_outputs
        self._learning_rate = learning_rate

        if activation_fn is None:
            self._activation_fn = sigmoid
            self._activation_der_fn = sigmoid_der

        # The weights for each layer L connected to the previous layer, L-1.
        # i.e., in an array of length 2 there will be 1 hidden layer.
        # index 0 will be the weights for input_layer -> hidden_layer
        # index 1 will be the weights for hidden_layer -> output_layer

        self.layers = []

        if hidden_layer_sizes is None:
            hidden_layer_sizes = []

        # Initialise the inputs...
        temp_inputs = num_inputs

        for sz in hidden_layer_sizes:
            self.layers.append(Layer(num_layer_inputs=temp_inputs, num_layer_outputs=sz))
            # The next input size is the previous layer's output size
            temp_inputs = sz

        self.layers.append(Layer(num_layer_inputs=temp_inputs, num_layer_outputs=num_outputs))

    def train(self, features: np.array, labels: np.array, epochs: int):
        """
        Trains a single epoch
        :param features:
        :param labels:
        :return: dict with debug data
        """

        if epochs <= 0:
            raise ValueError('epochs must be > 0')

        layer_z = []
        layer_activation = []

        error = None

        for i in range(0, epochs):

            # Initialise activation for forward propagation...
            current_layer_activation = features
            # Begin forward propagation...
            for layer in self.layers:
                current_layer_z = np.dot(layer.weights, current_layer_activation) + layer.biases
                layer_z.append(current_layer_z)
                current_layer_activation = self._activation_fn(current_layer_z)
                layer_activation.append(current_layer_activation)

                # Transpose in preparation for the next time around...
                current_layer_activation = current_layer_activation.T

            # Forward propagation done, calc error
            # All the errors for all of the input feature sets...
            error = layer_activation[-1] - labels

            # This is fixed for the duration of the back prop.
            d_cost_d_activation = 2 * error

            # Backwards propagation...
            this_layer_activation = features
            # Work your way forward through the layers so we don't change weights before we need them...
            for idx, layer in enumerate(self.layers):

                # From the current layer to the output...
                chain = None
                for inner_idx in range(idx, len(self.layers)):
                    # Step 1
                    if chain is None:
                        chain = this_layer_activation
                    else:
                        chain = chain * self.layers[inner_idx].weights

                    # Step 2
                    chain = np.dot(chain, self._activation_der_fn(layer_activation[inner_idx]))
                    # d_activation_d_z = self._activation_der_fn(layer_activation[idx])

                # d_z_d_w = this_layer_activation
                # d_cost_d_w = chain *   d_cost_d_activation * d_activation_d_z * d_z_d_w
                d_cost_d_w = np.dot(chain, d_cost_d_activation)

                print('d_cost_d_w', d_cost_d_w)

                # Transpose the matrix so we can add the things together...
                layer.weights -= self._learning_rate * d_cost_d_w.T

                # Prepare for the next stage of back propagation
                this_layer_activation = layer_activation[idx]

        # Mainly for test assertion
        return TrainResult(
            error=error,
            layer_z=layer_z,
            layer_activation=layer_activation
        )

    def evaluate(self, features: np.array, labels: np.array):
        pass

    def predict(self, features: np.array, labels: np.array):
        pass
