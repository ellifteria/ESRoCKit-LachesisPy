import numpy as np
from xml.dom.minidom import parse
from io import StringIO

class NeuralNetwork:

    @staticmethod
    def relu(X: np.ndarray) -> np.ndarray:
        return (X > 0) * X

    @staticmethod
    def sigmoid(X: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-1 * X))
    
    @staticmethod
    def interpret_ennf_file(file_path: str) -> tuple:
        ennf_doc = parse(file_path)

        sensor_neurons = {}
        motor_neurons = {}
        network_weights = {}
        network_biases = {}

        sensor_neuron_list = ennf_doc.getElementsByTagName("sensor_neuron")
        for neuron in sensor_neuron_list:
            sensor_neurons[int(neuron.getAttribute("index"))] = neuron.getAttribute("link")

        motor_neuron_list = ennf_doc.getElementsByTagName("motor_neuron")
        for neuron in motor_neuron_list:
            motor_neurons[int(neuron.getAttribute("index"))] = neuron.getAttribute("joint")

        network_layers = ennf_doc.getElementsByTagName("network_layer")
        for layer in network_layers:
            network_weights[int(layer.getAttribute("index"))] = np.loadtxt(
                StringIO(layer.getAttribute("weight_matrix").replace("; ", "\n"))
            )
            network_biases[int(layer.getAttribute("index"))] = np.loadtxt(
                StringIO(layer.getAttribute("bias_matrix").replace("; ", "\n"))
            )

        return (
            sensor_neurons,
            motor_neurons,
            network_weights,
            network_biases
        )


    def __init__(self, file_path: str, activation_function: str = "relu") -> None:
        sensor_neurons, motor_neurons, network_weights, network_biases = self.interpret_ennf_file(file_path)
        self.sensor_neurons: dict[int, str] = sensor_neurons
        self.motor_neurons: dict[int, str] = motor_neurons
        self.biases: dict[int, np.ndarray] = network_biases
        self.weights: dict[int, np.ndarray] = network_weights

        self.output: np.ndarray = np.ndarray(shape=(len(motor_neurons),), dtype=float)

        match activation_function:
            case "relu":
                temp_activation_function = self.relu
            case "sigmoid":
                temp_activation_function = self.sigmoid
            case _:
                raise ValueError("Activation function must be a supported LachesisPy NeuralNetwork activation function")
        self.activation_function: function = temp_activation_function

    def feed_forward(self, inputs: np.ndarray) -> None:
        a: np.ndarray = inputs

        for (w, b) in zip(self.weights.values(), self.biases.values()):
            z = (w * a) + b
            a = self.activation_function(z)

        self.output = a

    def get_sensor_neurons(self) -> dict[int, str]:
        return self.sensor_neurons
    
    def get_motor_neurons(self) -> dict[int, str]:
        return self.motor_neurons

    def get_network_output(self) -> np.ndarray:
        return self.output
