from typing import List, NamedTuple, Callable

import numpy as np

np.random.seed(123)


def log(msg):
    print(msg)


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
    _weights: np.array
    _biases: np.array

    def __init__(self, num_layer_inputs, num_layer_outputs):
        if num_layer_inputs < 1:
            raise ValueError('num_layer_inputs must be >= 1')
        if num_layer_outputs < 1:
            raise ValueError('num_layer_outputs must be >= 1')

        # Initialise with random weightings

        # Each weight matrix will be the ...
        # number of neurons in the previous layer : width
        # number of neurons in the current layer : height
        self._weights = np.random.rand(num_layer_outputs, num_layer_inputs)

        # The bias for each layer.
        # The last index is for output layer. Work backwards from that.
        # For each matrix...
        # Width = 1
        # Height = number of neurons in the current layer
        self._biases = np.random.rand(num_layer_outputs, 1)

    @property
    def weights(self) -> np.array:
        return self._weights

    @weights.setter
    def weights(self, value: np.array):
        if value.shape != self._weights.shape:
            raise ValueError('weights shapes do not match. old: {}, new: {}'.format(
                self._weights.shape, value.shape
            ))
        self._weights = value

    @property
    def biases(self) -> np.array:
        return self._biases

    @biases.setter
    def biases(self, value: np.array):
        if value.shape != self._biases.shape:
            raise ValueError('biases shapes do not match. old: {}, new: {}'.format(
                self._biases.shape, value.shape
            ))
        self._biases = value


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
            # Make sure to clear this before each epoch
            layer_activation = []

            # Initialise activation for forward propagation...
            log('------------------- Begin forward prop')
            current_layer_activation = features.T
            # Begin forward propagation...
            for idx, layer in enumerate(self.layers):
                log('calc layer z (layer {}):\nw={}\na={}\nb={}'.format(idx, layer.weights, current_layer_activation,
                                                                        layer.biases))
                current_layer_z = np.dot(layer.weights, current_layer_activation) + layer.biases
                layer_z.append(current_layer_z)
                current_layer_activation = self._activation_fn(current_layer_z)
                layer_activation.append(current_layer_activation)

                # Transpose in preparation for the next time around...
                # current_layer_activation = current_layer_activation

            log('activations {}'.format(layer_activation))

            # Forward propagation done, calc error
            # All the errors for all of the input feature sets...
            error = layer_activation[-1] - labels.T

            # This is fixed for the duration of the back prop.
            d_cost_d_activation = 2 * error

            log('------------------- Begin backward prop')

            # Backwards propagation...
            # Work your way forward through the layers so we don't change weights before we need them...
            for idx, layer in enumerate(self.layers):
                log('*** backwards prop to idx {}'.format(idx))

                # Work backwards from the output layer through to the layer
                # for which we are calculating the partial derivatives
                chain = None
                for inner_idx in reversed(range(idx, len(self.layers))):
                    # Step 1
                    if chain is None:
                        log('chain step 1: init chain: {}'.format(d_cost_d_activation))
                        chain = d_cost_d_activation
                    else:
                        weights = self.layers[inner_idx].weights
                        log(f'chain step 1: do chain {weights} x {chain}')
                        chain = np.dot(weights, chain)

                    # Step 2
                    d_activation_d_z = self._activation_der_fn(layer_z[inner_idx])
                    log(f'chain step 2: idx={inner_idx}\n{chain} x {d_activation_d_z}')
                    # Don't use dot product here. We want element-wise multiplication!
                    chain = chain * d_activation_d_z

                log('End of chain function operations: {}'.format(chain))

                # If this is the first layer, it means the layer activation is the
                # original input.
                # Otherwise, look up the layer activation
                previous_layer_activation = features.T if idx == 0 else layer_activation[idx - 1]

                log(f'd_cost_d_w calc: {chain} x {previous_layer_activation.T}')
                d_cost_d_w = np.dot(chain, previous_layer_activation.T)

                log('d_cost_d_w {}'.format(d_cost_d_w))

                weight_adjustments = self._learning_rate * d_cost_d_w

                log('weight_adj {}'.format(weight_adjustments))
                log('current weights {}'.format(layer.weights))

                # Transpose the matrix so we can add the things together...
                layer.weights -= weight_adjustments

                # derivative of cost with respect to bias.
                # No need to dot product with anything since the bias has no
                # mutiplier in the form: z = (a * w) + b
                for partial_bias_result in chain.T:
                    # This is pretty ugly and probably inefficient
                    d_cost_d_b = self._learning_rate * np.atleast_2d(partial_bias_result).T
                    log('d_cost_d_b {}'.format(d_cost_d_b))
                    log('current biases {}'.format(layer.biases))
                    layer.biases -= d_cost_d_b

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
