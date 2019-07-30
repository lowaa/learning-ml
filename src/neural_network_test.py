import unittest

import numpy as np
from hamcrest import assert_that, equal_to, close_to

from neural_network import calc_layer_z, NeuralNetwork, TrainResult

DELTA = 0.000000001


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

    def test_training_single_layer(self):
        # Single epoch training

        subject = NeuralNetwork(num_inputs=2, num_outputs=1, learning_rate=0.05)
        # Force our weights to be a certain thing...
        subject.layers[0].weights = np.array([[0.25, 0.4]])
        subject.layers[0].biases = np.array([[1]])

        features = np.array([[1], [0.1]])
        labels = np.array([[0.6]])

        train_result: TrainResult = subject.train(features=features, labels=labels, epochs=1)

        assert_that(train_result.error[0], close_to(0.18414719, 0.0000001))

        assert_that(subject.layers[0].weights[0][0], close_to(0.246037333, 0.00001))
        assert_that(subject.layers[0].weights[0][1], close_to(0.399603733, 0.00001))

    def test_training_hidden_layer(self):
        # Very simple single epoch training

        subject = NeuralNetwork(num_inputs=1, num_outputs=2, learning_rate=0.05,
                                hidden_layer_sizes=[2])
        # Force our weights to be a certain thing...
        subject.layers[0].weights = np.array([[0.25, 0.4]])
        subject.layers[0].biases = np.array([[1, 0.7]])

        features = np.array([[1], [0.1]])
        labels = np.array([[0.6, 0.3]])

        train_result: TrainResult = subject.train(features=features, labels=labels, epochs=1)

        assert_that(train_result.error[0], close_to(0.18414719, 0.0000001))

        assert_that(subject.layers[0].weights[0][0], close_to(0.246037333, 0.00001))
        assert_that(subject.layers[0].weights[0][1], close_to(0.399603733, 0.00001))
