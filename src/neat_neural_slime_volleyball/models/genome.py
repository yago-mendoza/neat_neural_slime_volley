import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Union, Tuple, Callable
from ..utils.pretty_dict import PrettyDict

# do outputs have a bias?

class Node:
    def __init__(
            self,
            id: int,
            type: str,
            activation_function: Callable = None
        ):
        self.id = id
        self.type = type
        self.activation_function = activation_function or self.identity
        self.bias = np.random.randn()
    
    @staticmethod
    def identity(x):
        return x
    
    def __str__(self):
        return f"Node(id={self.id}, type={self.type}, f={self.activation_function.__name__}, b={self.bias:.4f})"

    def __repr__(self):
        return self.__str__()
    
class Connection:
    def __init__(
            self,
            connection_id: int,
            from_node: int,
            to_node: int
        ):

        self.id = connection_id
        self.from_node = from_node
        self.to_node = to_node
    
    def _set_weight(self, input_size: int, output_size: int):
        # Scale weights to maintain stable variance (Xavier/Glorot)
        limit = np.sqrt(6 / (input_size + output_size))
        self.weight = np.random.uniform(-limit, limit)

    def __str__(self):
        return f"Connection(id={self.id}, from={self.from_node}, to={self.to_node}, w={self.weight:.4f})"

    def __repr__(self):
        return self.__str__()
    
class Genome:
    def __init__(
            self,
            input_size: int,
            output_size: int
        ):

        self.input_size = input_size
        self.output_size = output_size
        self.nodes: Dict[int, Node] = PrettyDict()
        self.connections: Dict[int, Connection] = PrettyDict()
        
        self._init_input_layer()
        self._init_output_layer()

        self._input_to_output_batch_connections()

    def add_hidden_node(
            self,
            from_node: Union[int, Node],
            to_node: Union[int, Node]
        ) -> Tuple[int, Node]:

        from_node_id = from_node.id if isinstance(from_node, Node) else from_node
        to_node_id = to_node.id if isinstance(to_node, Node) else to_node

        if from_node_id not in self.nodes or to_node_id not in self.nodes:
            raise ValueError("Invalid node IDs")
        
        from_node = self.nodes[from_node_id]
        to_node = self.nodes[to_node_id]
        
        if from_node.type == to_node.type in ("input", "output"):
            raise ValueError(f"Cannot add hidden node between 2 {from_node.type} nodes.")
        
        hidden_node_id, hidden_node = self._add_node(type="hidden")
        self._add_connection(from_node, hidden_node)
        self._add_connection(hidden_node, to_node)

        return hidden_node_id, hidden_node
    
    def _add_node(
            self,
            type: str = "hidden"
        ) -> Tuple[int, Node]:

        id = self._find_next_available_id(type="node")
        node = Node(id=id, type=type)
        self.nodes[id] = node
        return id, node
    
    def _add_connection(
            self,
            from_node: Union[int, Node],
            to_node: Union[int, Node]
        ) -> Tuple[int, Connection]:

        from_node_id = from_node if isinstance(from_node, int) else from_node.id
        to_node_id = to_node if isinstance(to_node, int) else to_node.id
        
        if from_node_id not in self.nodes or to_node_id not in self.nodes:
            raise ValueError("Invalid node IDs")
        
        connection_id = self._find_next_available_id(type="connection")
        connection = Connection(connection_id, from_node_id, to_node_id)
        connection._set_weight(input_size=self.input_size, output_size=self.output_size)
        self.connections[connection_id] = connection
        return connection_id, connection
    
    def _remove_connection(
            self,
            connection_id: int
        ) -> None:

        if connection_id not in self.connections:
            raise ValueError(f"Connection with ID {connection_id} does not exist.")
        
        del self.connections[connection_id]
    
    def remove_node(
            self,
            id: int
        ) -> None:

        if id not in self.nodes:
            raise ValueError(f"Node with ID {id} does not exist.")
        
        node = self.nodes[id]

        if node.type in ["input", "output"]:
            raise ValueError(f"Cannot remove {node.type} node with ID {id} since it is part of the I/O structure.")
        
        connections_to_remove = [
            conn_id for conn_id, conn in self.connections.items()
            if conn.from_node == id or conn.to_node == id
        ]
        
        for conn_id in connections_to_remove:
            self._remove_connection(conn_id)
        
        del self.nodes[id]

    def _init_input_layer(self):
        for _ in range(self.input_size):
            self._add_node(type="input")

    def _init_output_layer(self):
        for _ in range(self.output_size):
            self._add_node(type="output")

    def _input_to_output_batch_connections(self):

        input_nodes = [n for n in self.nodes.values() if n.type == "input"]
        output_nodes = [n for n in self.nodes.values() if n.type == "output"]
        
        for input_node in input_nodes:
            for output_node in output_nodes:
                self._add_connection(input_node, output_node)

    def _find_next_available_id(self, type: str = "node") -> int:
        """Find the smallest positive integer not used as a node or connection ID."""
        if type == "node":
            lst = list(self.nodes.keys())
        elif type == "connection":
            lst = list(self.connections.keys())
        else:
            raise ValueError("Invalid type. Must be 'node' or 'connection'.")
        return next(i for i in range(1, len(lst) + 2) if i not in set(lst))

    def __str__(self):
        return f"Genome:\nNodes:\n{self.nodes}\nConnections:\n{self.connections}"

    def __repr__(self):
        return self.__str__()
  

    
    

    

    # def forward(self, inputs: np.ndarray) -> np.ndarray:
    #     if inputs.shape[0] != self.input_size:
    #         raise ValueError(f"Expected {self.input_size} inputs, got {inputs.shape[0]}")

    #     node_values = {i: val for i, val in enumerate(inputs)}

    #     sorted_nodes = self._topological_sort()
    #     for node_id in sorted_nodes[self.input_size:]:
    #         node = self.nodes[node_id]
    #         incoming = [conn for conn in self.connections.values() if conn.to_node == node_id]
    #         node_input = sum(node_values[conn.from_node] * conn.weight for conn in incoming)
    #         node_values[node_id] = node.activation_function(node_input + node.bias)

    #     return np.array([node_values[i] for i in range(self.input_size, self.input_size + self.output_size)])

    # def _topological_sort(self) -> List[int]:
    #     # Implement topological sort algorithm here
    #     # This ensures correct order of node evaluation
    #     pass

    # def save(self, folder: str) -> None:
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     save_folder = os.path.join(folder, f"network_{timestamp}")
    #     os.makedirs(save_folder, exist_ok=True)

    #     network_data = {
    #         "input_size": self.input_size,
    #         "output_size": self.output_size,
    #         "nodes": [{"id": n.id, "bias": n.bias} for n in self.nodes.values()],
    #         "connections": [{"from": c.from_node, "to": c.to_node, "weight": c.weight} for c in self.connections]
    #     }

    #     with open(os.path.join(save_folder, "network_structure.json"), "w") as f:
    #         json.dump(network_data, f, indent=2)

    # @classmethod
    # def load(cls, folder: str) -> 'Genome':
    #     with open(os.path.join(folder, "network_structure.json"), "r") as f:
    #         network_data = json.load(f)

    #     nn = cls(network_data["input_size"], network_data["output_size"])
    #     nn.nodes = {n["id"]: Node(n["id"]) for n in network_data["nodes"]}
    #     for node_data in network_data["nodes"]:
    #         nn.nodes[node_data["id"]].bias = node_data["bias"]

    #     nn.connections = {}
    #     for conn_data in network_data["connections"]:
    #         connection = Connection(len(nn.connections), conn_data["from"], conn_data["to"])
    #         connection.weight = conn_data["weight"]
    #         nn.connections[connection.id] = connection

    #     return nn

    # # Additional methods for setting activation functions, etc.