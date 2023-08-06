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
    def linear(X: np.ndarray) -> np.ndarray:
        return X
    
    @staticmethod
    def interpret_ennf_file(file_path: str) -> tuple:
        ennf_doc = parse(file_path)

        sensor_neurons  = {}
        motor_neurons   = {}
        network_weights = {}
        network_biases  = {}

        sensor_neuron_list = ennf_doc.getElementsByTagName("sensor")
        for neuron in sensor_neuron_list:
            sensor_neurons[neuron.getAttribute("link")] = int(neuron.getAttribute("index"))

        motor_neuron_list = ennf_doc.getElementsByTagName("motor")
        for neuron in motor_neuron_list:
            motor_neurons[int(neuron.getAttribute("index"))] = neuron.getAttribute("joint")

        network_layers = ennf_doc.getElementsByTagName("network_layer")
        for layer in network_layers:
            index = int(layer.getAttribute("index"))
            network_weights[index] = np.loadtxt(
                StringIO(layer.getAttribute("weight_matrix").replace("; ", "\n"))
            )
            network_biases[index] = np.loadtxt(
                StringIO(layer.getAttribute("bias_matrix").replace(", ", "\n"))
            )
            
            network_biases[index] = np.reshape(
                network_biases[index],
                (np.size(network_biases[index]), 1)) 
            
            if len(np.shape(network_weights[index])) == 1:
                if np.size(network_biases[index]) == 1:
                    network_weights[index] = np.reshape(
                        network_weights[index],
                        (np.size(network_weights[index]), 1))
                else:
                    network_weights[index] = np.reshape(
                        network_weights[index],
                        (1, np.size(network_weights[index])))

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

        self.sensor_neurons:    dict[str, int]          = sensor_neurons
        self.motor_neurons:     dict[int, str]          = motor_neurons
        self.biases:            dict[int, np.ndarray]   = network_biases
        self.weights:           dict[int, np.ndarray]   = network_weights

        self._validate_network()

        match activation_function:
            case "relu":
                temp_activation_function = self.relu
            case "sigmoid":
                temp_activation_function = self.sigmoid
            case "linear":
                temp_activation_function = self.linear
            case _:
                raise ValueError("Activation function must be a supported LachesisPy NeuralNetwork activation function")
        self.activation_function: Callable[[np.ndarray], np.ndarray] = temp_activation_function

        self._output_has_been_calculated: bool = False

    # ACCESS METHODS
    def get_sensor_neurons(self) -> dict[str, int]:
        return self.sensor_neurons
    
    def get_motor_neurons(self) -> dict[int, str]:
        return self.motor_neurons

    def get_network_output(self) -> np.ndarray:
        if not self._output_has_been_calculated:
            raise RuntimeError("No output exists; you must feed forward through the network first")
        return self.output
    
    def get_joint_targets(self) -> dict[str, float]:
        network_output: np.ndarray  = self.get_network_output()
        output_len:     int         = len(self.motor_neurons)
        joints_list:    list[str]   = [''] * output_len
        values_list:    list[float] = [0.0] * output_len

        for i in range(output_len):
            joints_list[i] = self.motor_neurons[i]
            values_list[i] = float(network_output[i])
        
        targets = {joint: value for (joint, value) in zip(joints_list, values_list)}
            
        return targets
    
    # MODIFIER METHODS
    def update_sensor_neurons(self, new_sensor_neurons: dict[str, int]) -> None:
        warnings.warn("WARNING: Use of this method is not recommended for most cases and can prevent LachesisPy from being able to simulate properly. \nDo not use this methods unless you know what you are doing!")
        
        if len(new_sensor_neurons) != len(self.sensor_neurons):
            warnings.warn("WARNING: The provided sensor neuron dictionary does NOT properly match the size of the current sensor neurons dictionary. This discrepancy may prevent LachesisPy from being able to simulate properly.\nDo not use this method unless you know what you are doing!")
        
        self.sensor_neurons = new_sensor_neurons
    
    def update_motor_neurons(self, new_motor_neurons: dict[int, str]) -> None:
        warnings.warn("WARNING: Use of this method is not recommended for most cases and can prevent LachesisPy from being able to simulate properly. \nDo not use this methods unless you know what you are doing!")
        
        if len(new_motor_neurons) != len(self.motor_neurons):
            warnings.warn("WARNING: The provided motor neuron dictionary does NOT properly match the size of the current motor neurons dictionary. This discrepancy may prevent LachesisPy from being able to simulate properly.\nDo not use this method unless you know what you are doing!")
        
        self.motor_neurons = new_motor_neurons

    # ACTION METHODS
    def feed_forward_raw(self, inputs: np.ndarray) -> None:
        a: np.ndarray = inputs

        for (w, b) in zip(self.weights.values(), self.biases.values()):
            z = (w @ a) + b
            a = self.activation_function(z)

        self.output: np.ndarray = a
        self._output_has_been_calculated = True

    def feed_forward(self, inputs: dict[str, float]) -> None:
        if isinstance(inputs, np.ndarray):
            warnings.warn("Use of feed_forward method with np.ndarray parameter is being depreciated. For feed_forward, please provide an argument of type dict[str, float] or use feed_forward_raw")
            self.feed_forward_raw(inputs)
            return

        raw_inputs = np.zeros((len(self.sensor_neurons), 1))

        for neuron_name in inputs:
            index = self.sensor_neurons[neuron_name]
            raw_inputs[index] = inputs[neuron_name]

        self.feed_forward_raw(raw_inputs)
