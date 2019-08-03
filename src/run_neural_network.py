import numpy as np

from neural_network import NeuralNetwork


def run_neural_network():
    feature_set = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]])
    labels = np.array([[1], [0], [0], [1], [1]])

    nn = NeuralNetwork(learning_rate=0.05,
                       num_inputs=3,
                       num_outputs=1)

    nn.train(features=feature_set, labels=labels, epochs=10000)
    print(nn.evaluate(features=feature_set, labels=labels))


if __name__ == '__main__':
    run_neural_network()
