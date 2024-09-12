import math
import numpy as np
from typing import List, Union, Callable

# Activation functions
def identity(x: float) -> float:
    return x

def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))

def tanh(x: float) -> float:
    return math.tanh(x)

def relu(x: float) -> float:
    return max(0, x)

ACTIVATION_FUNCTIONS = {
    'identity': identity,
    'sigmoid': sigmoid,
    'tanh': tanh,
    'relu': relu
}

class Node:
    """
    Constructor for a node in the neural network.

    """
    def __init__(
        self,
        id: int,
        node_type: str,
        bias: float = 0,
        activation_function: str = 'identity'
    ):
        self.id = id
        self.type = node_type

        self.incoming_synapses: List[Node] = []
        self.outgoing_synapses: List[Node] = []

        self.activation_function = ACTIVATION_FUNCTIONS["identity"]
        self.bias = 0.0

        self.value = 0.0

    def set_id(self, id: int):
        self.id = id

    def set_activation_function(self, activation_function: str):
        self.activation_function = ACTIVATION_FUNCTIONS[activation_function]
    
    def rswitch_activation_function(self):
        current = self.activation_function
        while self.activation_function == current:
            self.set_activation_function(np.random.choice(list(ACTIVATION_FUNCTIONS.keys())))

    def set_bias(self, bias: float):
        self.bias = bias
    
    def set_value(self, value: float):
        self.value = value

    def _check_integrity(self):
        pass  # Placeholder for integrity check implementation in subclasses
    
    def to_dict(self):
        # This is used to save the genome to a file
        return {
            "id": self.id,
            "type": self.type,
            "bias": self.bias,
            "activation_function": self.activation_function.__name__
        }
    
    @classmethod
    def from_dict(cls, data):
        # This is used to load the genome from a file
        node = cls(data["id"], data["type"])
        node.set_bias(data["bias"])
        node.set_activation_function(data["activation_function"])
        return node

    def __str__(self):
        return f"Node(id={self.id}, type={self.type}, f={self.activation_function.__name__}, b={self.bias:.4f})"

    def __repr__(self):
        return self.__str__()

class InputNode(Node):
    def __init__(self, id: int, bias: float = 0, activation_function: str = 'identity'):
        super().__init__(id, "input", bias, activation_function)

    def _check_integrity(self):
        # Check that input node does not have incoming synapses
        if self.incoming_synapses:
            raise ValueError(f"InputNode cannot have incoming synapses ({len(self.incoming_synapses)} found).")

        # Check that input node has at least one outgoing synapse
        if not self.outgoing_synapses:
            print(f"Warning: InputNode {self.id} must have at least one outgoing synapse.")
            raise ValueError(f"InputNode {self.id} must have at least one outgoing synapse.")

class OutputNode(Node):
    def __init__(self, id: int, bias: float = 0, activation_function: str = 'sigmoid'):
        super().__init__(id, "output", bias, activation_function)

    def _check_integrity(self):
        # Check that output node does not have outgoing synapses
        if self.outgoing_synapses:
            raise ValueError(f"OutputNode {self.id} cannot have outgoing synapses ({len(self.outgoing_synapses)} found).")
        
        # Check that output node has at least one incoming synapse  
        if not self.incoming_synapses:
            raise ValueError(f"OutputNode {self.id} must have at least one incoming synapse.")

class HiddenNode(Node):
    def __init__(self, id: int, bias: float = 0, activation_function: str = 'relu'):
        super().__init__(id, "hidden", bias, activation_function)

    def _check_integrity(self):
        # Check that hidden node has at least one incoming and one outgoing synapses
        if not self.incoming_synapses or not self.outgoing_synapses:
            raise ValueError(f"HiddenNode {self.id} must have at least 1 incoming & 1 outgoing synapses.")
