import math
from typing import Callable

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
    def __init__(
            self,
            id: int,
            type: str,
            activation_function: str = "identity"
        ):
        self.id = id
        self.type = type
        self.activation_function = ACTIVATION_FUNCTIONS[activation_function]
        self.bias = 0.0

        self.value = 0.0
    
    def set_bias(self, bias: float):
        self.bias = bias
        return self
    
    def set_activation_function(self, activation_function: str):
        self.activation_function = ACTIVATION_FUNCTIONS[activation_function]
        return self
    
    def to_dict(self):
        return {
            "id": self.id,
            "type": self.type,
            "bias": self.bias,
            "activation_function": self.activation_function.__name__
        }

    @classmethod
    def from_dict(cls, data):
        node = cls(data["id"], data["type"])
        node.set_bias(data["bias"])
        node.set_activation_function(data["activation_function"])
        return node

    def __str__(self):
        return f"Node(id={self.id}, type={self.type}, f={self.activation_function.__name__}, b={self.bias:.4f})"

    def __repr__(self):
        return self.__str__()


