import numpy as np



class NeuralNetwork:

    @staticmethod
    def relu(X: np.ndarray) -> np.ndarray:
        return (X > 0) * X

    @staticmethod
    def sigmoid(X: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-1 * X))

    def __init__(self) -> None:
        self.biases: np.ndarray = np.ndarray(shape=(2,), dtype=float)
        self.weights: np.ndarray = np.ndarray(shape=(2,), dtype=float)
        self.activation_function: function = self.relu

    def feed_forward(self) -> None:
        pass

    def get_network_outputs(self) -> None:
        pass
