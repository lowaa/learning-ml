import unittest

import numpy as np
from hamcrest import assert_that, equal_to, close_to

from neural_network import calc_layer_z, NeuralNetwork, TrainResult

DELTA = 0.0000001


class NeuralNetworkTestCase(unittest.TestCase):

    def test_simple_layer_z(self):
        r0 = calc_layer_z(
            weight=np.array([0.2]),
            previous_activation=np.array([0.5]),
            bias=np.array([1.0])
        )
        assert_that(r0, equal_to(1.1))

    def test_more_complicated_layer_z(self):
        r0 = calc_layer_z(
            weight=np.array([[0.2, 0.1], [0.3, 0.4]]),
            previous_activation=np.array([0.5, 0.4]),
            bias=np.array([1.0, 2.0])
        )
        assert_that(r0[0], close_to(1.14, DELTA))
        assert_that(r0[1], equal_to(2.31), DELTA)

    def test_matrix_mult(self):
        # Just making sure the dimensions work the way I think they do
        r0 = np.matmul(np.array([[1, 2]]), np.array([[1], [1]]))
        assert_that(r0[0][0], equal_to(3))

    def test_training_single_layer_2_in_1_out(self):
        # Single epoch training

        subject = NeuralNetwork(num_inputs=2, num_outputs=1, learning_rate=0.05)
        # Force our weights to be a certain thing...
        subject.layers[0].weights = np.array([[0.25, 0.4]])
        subject.layers[0].biases = np.array([[1.0]])

        features = np.array([[1, 0.1]])
        labels = np.array([[0.6]])

        train_result: TrainResult = subject.train(features=features, labels=labels, epochs=1)

        assert_that(train_result.error[0], close_to(0.18414719, DELTA))

        assert_that(subject.layers[0].weights[0][0], close_to(0.2468831178638474, DELTA))
        assert_that(subject.layers[0].weights[0][1], close_to(0.39968831178638475, DELTA))

        assert_that(subject.layers[0].biases[0][0], close_to(0.9968831178638474, DELTA))

    def test_training_single_layer_1_in_2_out(self):
        # Single epoch training

        subject = NeuralNetwork(num_inputs=1, num_outputs=2, learning_rate=0.05)
        # Force our weights to be a certain thing...
        subject.layers[0].weights = np.array([[0.25], [0.4]])
        subject.layers[0].biases = np.array([[1.0], [2.0]])

        features = np.array([[0.5]])
        labels = np.array([[0.6, 0.2]])

        train_result: TrainResult = subject.train(features=features, labels=labels, epochs=1)

        assert_that(train_result.error[0][0], close_to(0.1549149, DELTA))
        assert_that(train_result.error[1][0], close_to(0.7002495, DELTA))

        assert_that(subject.layers[0].weights[0][0], close_to(0.248566895, DELTA))
        assert_that(subject.layers[0].weights[1][0], close_to(0.396855868, DELTA))

        assert_that(subject.layers[0].biases[0][0], close_to(0.99713379, DELTA))
        assert_that(subject.layers[0].biases[1][0], close_to(1.99371173708, DELTA))

    def test_training_hidden_layer(self):
        # Very simple single epoch training

        subject = NeuralNetwork(num_inputs=2, num_outputs=2, learning_rate=0.05,
                                hidden_layer_sizes=[3])
        # Force our weights to be a certain thing...
        subject.layers[0].weights = np.array([[0.25, 0.4], [0.25, 0.4], [0.25, 0.4]])
        subject.layers[0].biases = np.array([[1], [0.7], [0.6]])
        subject.layers[1].weights = np.array([[0.25, 0.4, 0.25], [0.4, 0.25, 0.4]])
        subject.layers[1].biases = np.array([[1], [0.7]])

        features = np.array([[0.8, 0.1]])
        labels = np.array([[0.6, 0.3]])

        train_result: TrainResult = subject.train(features=features, labels=labels, epochs=1)

        # TODO: hand-calc result for assertion

    def test_training_single_layer_2_in_1_out_multiple_input_samples(self):
        # Single epoch training

        subject = NeuralNetwork(num_inputs=2, num_outputs=1, learning_rate=0.05)
        # Force our weights to be a certain thing...
        subject.layers[0].weights = np.array([[0.25, 0.4]])
        subject.layers[0].biases = np.array([[1.0]])

        features = np.array([[1, 0.1], [1, 0.1]])
        labels = np.array([[0.6], [0.6]])

        train_result: TrainResult = subject.train(features=features, labels=labels, epochs=1)

        assert_that(train_result.error[0][0], close_to(0.18414719, DELTA))
        assert_that(train_result.error[0][1], close_to(0.18414719, DELTA))

        assert_that(subject.layers[0].weights[0][0], close_to(0.24376623551740695, DELTA))
        assert_that(subject.layers[0].weights[0][1], close_to(0.3993766235517407, DELTA))

        assert_that(subject.layers[0].biases[0][0], close_to(0.993766235517407, DELTA))

    def test_training_single_layer_1_in_2_out_multiple_input_samples(self):
        # Single epoch training

        subject = NeuralNetwork(num_inputs=1, num_outputs=2, learning_rate=0.05)
        # Force our weights to be a certain thing...
        subject.layers[0].weights = np.array([[0.25], [0.4]])
        subject.layers[0].biases = np.array([[1.0], [2.0]])

        features = np.array([[0.5], [0.5]])
        labels = np.array([[0.6, 0.2], [0.6, 0.2]])

        train_result: TrainResult = subject.train(features=features, labels=labels, epochs=1)

        assert_that(train_result.error[0][0], close_to(0.1549149, DELTA))
        assert_that(train_result.error[0][1], close_to(0.1549149, DELTA))
        assert_that(train_result.error[1][0], close_to(0.7002495, DELTA))
        assert_that(train_result.error[1][1], close_to(0.7002495, DELTA))

        assert_that(subject.layers[0].weights[0][0], close_to(0.247133790856, DELTA))
        assert_that(subject.layers[0].weights[1][0], close_to(0.3937117371, DELTA))

        assert_that(subject.layers[0].biases[0][0], close_to(0.994267581712, DELTA))
        assert_that(subject.layers[0].biases[1][0], close_to(1.9874234742, DELTA))
