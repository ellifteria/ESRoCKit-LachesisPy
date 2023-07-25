import numpy as np
from xml.dom.minidom import parse
from io import StringIO
import warnings
from collections.abc import Callable

class NeuralNetwork:

    # STATIC METHODS
    @staticmethod
    def relu(X: np.ndarray) -> np.ndarray:
        return (X > 0) * X

    @staticmethod
    def sigmoid(X: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-1 * X))
    
    @staticmethod
    def interpret_ennf_file(file_path: str) -> tuple:
        ennf_doc = parse(file_path)

        sensor_neurons  = {}
        motor_neurons   = {}
        network_weights = {}
        network_biases  = {}

        sensor_neuron_list = ennf_doc.getElementsByTagName("sensor")
        for neuron in sensor_neuron_list:
            sensor_neurons[int(neuron.getAttribute("index"))] = neuron.getAttribute("link")

        motor_neuron_list = ennf_doc.getElementsByTagName("motor")
        for neuron in motor_neuron_list:
            motor_neurons[int(neuron.getAttribute("index"))] = neuron.getAttribute("joint")

        network_layers = ennf_doc.getElementsByTagName("network_layer")
        for layer in network_layers:
            network_weights[int(layer.getAttribute("index"))] = np.loadtxt(
                StringIO(layer.getAttribute("weight_matrix").replace("; ", "\n"))
            )
            network_biases[int(layer.getAttribute("index"))] = np.loadtxt(
                StringIO(layer.getAttribute("bias_matrix").replace(", ", "\n"))
            )

        return (
            sensor_neurons,
            motor_neurons,
            network_weights,
            network_biases
        )

    # PRIVATE METHODS
    def _validate_network(self) -> None:
        if len(self.sensor_neurons) != np.shape(self.weights[0])[1]:
            warnings.warn("Non-fatal incorrect size: initial layer; number of columns does not match number of sensor neurons")
        
        if len(self.motor_neurons) != np.shape(self.weights[max(self.weights.keys())])[0]:
            warnings.warn("Non-fatal incorrect size: final layer; number of rows does not match number of motor neurons")

        for i in range(len(self.weights) - 1):
            first_layer = self.weights[i]
            second_layer = self.weights[i + 1]

            first_shape = np.shape(first_layer)[0]
            second_shape = np.shape(second_layer)[1]

            if first_shape != second_shape:
                raise ValueError(f"Fatal incorrect size: layers {i}, {i+1} sizes ({first_shape}, {second_shape}) do not match properly for feed-forward matrix multiplication")

    # CONSTRUCTOR
    def __init__(self, file_path: str, activation_function: str = "relu") -> None:
        sensor_neurons, motor_neurons, network_weights, network_biases = self.interpret_ennf_file(file_path)

        self.sensor_neurons:    dict[int, str]          = sensor_neurons
        self.motor_neurons:     dict[int, str]          = motor_neurons
        self.biases:            dict[int, np.ndarray]   = network_biases
        self.weights:           dict[int, np.ndarray]   = network_weights

        self._validate_network()

        match activation_function:
            case "relu":
                temp_activation_function = self.relu
            case "sigmoid":
                temp_activation_function = self.sigmoid
            case _:
                raise ValueError("Activation function must be a supported LachesisPy NeuralNetwork activation function")
        self.activation_function: Callable[[np.ndarray], np.ndarray] = temp_activation_function

        self._output_has_been_calculated: bool = False

    # ACCESS METHODS
    def get_sensor_neurons(self) -> dict[int, str]:
        return self.sensor_neurons
    
    def get_motor_neurons(self) -> dict[int, str]:
        return self.motor_neurons

    def get_network_output(self) -> np.ndarray:
        if not self._output_has_been_calculated:
            raise RuntimeError("No output exists; you must feed forward through the network first")
        return self.output
    
    def get_joint_targets(self) -> tuple[list[str], list[float]]:
        network_output: np.ndarray = self.get_network_output()
        output_len = len(self.motor_neurons)
        joints_list = [''] * output_len
        values_list = [0.0] * output_len
        for i in range(output_len):
            joints_list[i] = self.motor_neurons[i]
            values_list[i] = network_output[i]
        return (joints_list, values_list)

    # ACTION METHODS
    def feed_forward(self, inputs: np.ndarray) -> None:
        a: np.ndarray = inputs

        for (w, b) in zip(self.weights.values(), self.biases.values()):
            z = (w * a) + b
            a = self.activation_function(z)

        self.output: np.ndarray = a
        self._output_has_been_calculated = True
